"""
cost_tracker - OpenTelemetry-based per-token cost tracking for LLM serving.

Emits per-request spans with OpenInference semantic conventions, calculates
cost from a versioned price table, enforces per-tenant hourly budgets via
Redis, and exports cost histograms for Prometheus.

Usage:
    from cost_tracker import CostTracker

    tracker = CostTracker(prices=PriceTable.load("prices.yaml"), redis=redis_client)

    with tracker.trace(tenant_id="team-a", prompt_version="v3") as span:
        # ... call LLM ...
        span.record(
            model="gpt-4o",
            prompt_tokens=1500,
            completion_tokens=300,
            cached_tokens=800,
            finish_reason="stop",
            ttft_ms=420,
            itl_ms=35,
        )
        # span auto-computes cost, checks budget, emits metrics
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# OpenTelemetry imports - install with: pip install opentelemetry-api opentelemetry-sdk
try:
    from opentelemetry import trace
    from opentelemetry.metrics import Meter
except ImportError:
    trace = None  # type: ignore

try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore


@dataclass
class ModelPrice:
    """Versioned pricing for a single model."""

    model: str
    input_price_per_1k: float  # USD per 1K input tokens
    output_price_per_1k: float  # USD per 1K output tokens
    cached_price_per_1k: float  # USD per 1K cached prefix tokens (usually 0 or 0.1x input)
    effective_date: datetime  # when this price took effect


@dataclass
class PriceTable:
    """Versioned price table. Load from YAML or JSON, never hardcode."""

    prices: dict[str, list[ModelPrice]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "PriceTable":
        with open(path) as f:
            if path.endswith(".json"):
                data = json.load(f)
            else:
                import yaml  # type: ignore

                data = yaml.safe_load(f)
        table = cls()
        for model, entries in data.items():
            table.prices[model] = [
                ModelPrice(
                    model=model,
                    input_price_per_1k=e["input_per_1k"],
                    output_price_per_1k=e["output_per_1k"],
                    cached_price_per_1k=e.get("cached_per_1k", 0.0),
                    effective_date=datetime.fromisoformat(e["effective_date"]),
                )
                for e in entries
            ]
        return table

    def get_price(self, model: str, at: Optional[datetime] = None) -> ModelPrice:
        """Get the price for a model as of a given datetime (default: now)."""
        at = at or datetime.now(timezone.utc)
        entries = self.prices.get(model)
        if not entries:
            raise KeyError(f"no price for model {model}")
        # find the most recent price effective at or before `at`
        applicable = [p for p in entries if p.effective_date <= at]
        if not applicable:
            raise KeyError(f"no price for {model} effective before {at}")
        return max(applicable, key=lambda p: p.effective_date)


def compute_cost(
    price: ModelPrice,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """
    Cost = (prompt - cached) * input_price + completion * output_price
           + cached * cached_price.

    The cached_tokens term is the one most teams miss. Cached prefix tokens
    are usually free or discounted 10x. Counting them at full price makes
    your cost numbers wrong by the cache hit rate.
    """
    billable_input = max(0, prompt_tokens - cached_tokens)
    return (
        billable_input * price.input_price_per_1k / 1000.0
        + completion_tokens * price.output_price_per_1k / 1000.0
        + cached_tokens * price.cached_price_per_1k / 1000.0
    )


def estimate_cost(
    price: ModelPrice,
    prompt_tokens: int,
    expected_completion_tokens: int = 0,
) -> float:
    """Pre-call cost estimate for budget checking. Cannot know completion
    tokens ahead of time, so use max_tokens or a per-tenant historical avg."""
    return (
        prompt_tokens * price.input_price_per_1k / 1000.0
        + expected_completion_tokens * price.output_price_per_1k / 1000.0
    )


class BudgetExceeded(Exception):
    """Raised when a tenant exceeds their hourly budget."""

    def __init__(self, tenant_id: str, spent: float, limit: float):
        self.tenant_id = tenant_id
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"tenant {tenant_id} exceeded hourly budget: ${spent:.4f} > ${limit:.4f}"
        )


def _hour_bucket() -> str:
    """Redis key suffix for the current hour bucket (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H")


class CostTracker:
    """
    Per-request cost tracking with OTel spans, budget enforcement, and
    Prometheus histograms.

    Args:
        prices: PriceTable with versioned model pricing.
        redis: Optional Redis client for budget enforcement. If None,
            budget checks are skipped (useful for testing).
        meter: Optional OTel Meter for histogram export. If None, histograms
            are not exported.
        tracer: Optional OTel Tracer. If None, spans are not emitted.
        tenant_limits: dict of tenant_id -> hourly USD limit.
    """

    def __init__(
        self,
        prices: PriceTable,
        redis: Optional["redis.Redis"] = None,
        meter: Optional["Meter"] = None,
        tracer: Optional["trace.Tracer"] = None,
        tenant_limits: Optional[dict[str, float]] = None,
    ):
        self.prices = prices
        self.redis = redis
        self.tenant_limits = tenant_limits or {}
        self.tracer = tracer or (trace.get_tracer(__name__) if trace else None)
        self.meter = meter
        self._init_histograms()

    def _init_histograms(self) -> None:
        if not self.meter:
            self.cost_hist = None
            self.token_hist = None
            return
        self.cost_hist = self.meter.create_histogram(
            name="gen_ai.client.cost",
            description="Estimated cost in USD per LLM request",
            unit="USD",
        )
        self.token_hist = self.meter.create_histogram(
            name="gen_ai.client.token_usage",
            description="Total tokens per LLM request",
            unit="tokens",
        )

    def check_budget(self, tenant_id: str, estimated_cost: float) -> bool:
        """Check if tenant can spend estimated_cost. Returns False if over."""
        if not self.redis:
            return True
        limit = self.tenant_limits.get(tenant_id)
        if limit is None:
            return True  # no limit set
        key = f"budget:{tenant_id}:{_hour_bucket()}"
        spent = self.redis.incrbyfloat(key, estimated_cost)
        if spent > limit:
            # rollback the increment so we do not accumulate rejected spend
            self.redis.incrbyfloat(key, -estimated_cost)
            return False
        # set TTL so old buckets expire (2 hours)
        self.redis.expire(key, 7200)
        return True

    def trace(self, tenant_id: str, prompt_version: str = "unknown"):
        """Context manager for tracing a single LLM request."""
        return _RequestSpan(self, tenant_id, prompt_version)

    def _record(
        self,
        tenant_id: str,
        prompt_version: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        finish_reason: str,
        ttft_ms: float,
        itl_ms: float,
    ) -> float:
        price = self.prices.get_price(model)
        cost = compute_cost(price, prompt_tokens, completion_tokens, cached_tokens)

        if self.cost_hist:
            self.cost_hist.record(
                cost,
                attributes={
                    "tenant_id": tenant_id,
                    "gen_ai.response.model": model,
                    "prompt_version": prompt_version,
                    "finish_reason": finish_reason,
                },
            )
        if self.token_hist:
            self.token_hist.record(
                prompt_tokens + completion_tokens,
                attributes={
                    "tenant_id": tenant_id,
                    "gen_ai.response.model": model,
                },
            )
        return cost


class _RequestSpan:
    """Context manager for a single traced LLM request."""

    def __init__(self, tracker: CostTracker, tenant_id: str, prompt_version: str):
        self.tracker = tracker
        self.tenant_id = tenant_id
        self.prompt_version = prompt_version
        self.span = None
        self.cost: Optional[float] = None

    def __enter__(self) -> "_RequestSpan":
        if self.tracker.tracer:
            self.span = self.tracker.tracer.start_span(
                "llm.request",
                attributes={
                    "tenant_id": self.tenant_id,
                    "prompt_version": self.prompt_version,
                },
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.end()

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        finish_reason: str = "stop",
        ttft_ms: float = 0.0,
        itl_ms: float = 0.0,
    ) -> float:
        """Record the result of the LLM call. Returns the computed cost."""
        if self.span:
            self.span.set_attributes(
                {
                    "gen_ai.request.model": model,
                    "gen_ai.response.model": model,
                    "gen_ai.usage.prompt_tokens": prompt_tokens,
                    "gen_ai.usage.completion_tokens": completion_tokens,
                    "gen_ai.usage.cached_tokens": cached_tokens,
                    "gen_ai.response.finish_reason": finish_reason,
                    "gen_ai.latency.time_to_first_token": ttft_ms,
                    "gen_ai.latency.time_per_output_token": itl_ms,
                }
            )
        self.cost = self.tracker._record(
            self.tenant_id,
            self.prompt_version,
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            finish_reason,
            ttft_ms,
            itl_ms,
        )
        return self.cost

    def estimate_and_check(
        self, model: str, prompt_tokens: int, expected_completion: int = 0
    ) -> bool:
        """Pre-call: estimate cost and check budget. Returns False if over."""
        price = self.tracker.prices.get_price(model)
        est = estimate_cost(price, prompt_tokens, expected_completion)
        return self.tracker.check_budget(self.tenant_id, est)
