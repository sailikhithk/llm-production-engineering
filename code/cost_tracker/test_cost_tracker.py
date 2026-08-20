"""Tests for cost_tracker. Run with: python -m pytest test_cost_tracker.py"""

import sys
import os
from datetime import datetime, timezone

# Add parent dir so we can import the cost_tracker package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_tracker import (
    PriceTable,
    ModelPrice,
    compute_cost,
    estimate_cost,
    CostTracker,
    BudgetExceeded,
)


def test_compute_cost_basic():
    """No cache: cost = prompt * input + completion * output."""
    price = ModelPrice(
        model="test",
        input_price_per_1k=0.005,
        output_price_per_1k=0.015,
        cached_price_per_1k=0.0025,
        effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    # 1000 input, 500 output, 0 cached
    cost = compute_cost(price, prompt_tokens=1000, completion_tokens=500, cached_tokens=0)
    assert cost == 0.005 + 0.0075  # 0.0125
    assert abs(cost - 0.0125) < 1e-9


def test_compute_cost_with_cache():
    """With cache: cached tokens billed at cached_price, not input_price."""
    price = ModelPrice(
        model="test",
        input_price_per_1k=0.005,
        output_price_per_1k=0.015,
        cached_price_per_1k=0.0025,  # half of input
        effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    # 1000 input, 800 cached, 500 output
    # billable_input = 200, cached = 800
    cost = compute_cost(price, prompt_tokens=1000, completion_tokens=500, cached_tokens=800)
    expected = 200 * 0.005 / 1000 + 500 * 0.015 / 1000 + 800 * 0.0025 / 1000
    assert abs(cost - expected) < 1e-9


def test_compute_cost_free_cache():
    """When cached_price is 0, cached tokens are free."""
    price = ModelPrice(
        model="test",
        input_price_per_1k=0.005,
        output_price_per_1k=0.015,
        cached_price_per_1k=0.0,
        effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    cost = compute_cost(price, prompt_tokens=1000, completion_tokens=500, cached_tokens=1000)
    # all input cached, only output billed
    assert abs(cost - 0.0075) < 1e-9


def test_estimate_cost():
    """Pre-call estimate uses only prompt + expected completion."""
    price = ModelPrice(
        model="test",
        input_price_per_1k=0.005,
        output_price_per_1k=0.015,
        cached_price_per_1k=0.0,
        effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    est = estimate_cost(price, prompt_tokens=1000, expected_completion_tokens=500)
    assert abs(est - 0.0125) < 1e-9


def test_price_table_versioning():
    """PriceTable returns the correct price for a given datetime."""
    table = PriceTable(prices={
        "test-model": [
            ModelPrice("test-model", 0.010, 0.030, 0.0, datetime(2024, 1, 1, tzinfo=timezone.utc)),
            ModelPrice("test-model", 0.005, 0.015, 0.0, datetime(2024, 6, 1, tzinfo=timezone.utc)),
        ]
    })
    # before the price cut
    old = table.get_price("test-model", at=datetime(2024, 3, 1, tzinfo=timezone.utc))
    assert old.input_price_per_1k == 0.010
    # after the price cut
    new = table.get_price("test-model", at=datetime(2024, 9, 1, tzinfo=timezone.utc))
    assert new.input_price_per_1k == 0.005


def test_budget_enforcement_no_redis():
    """Without Redis, budget checks always pass (no enforcement)."""
    tracker = CostTracker(
        prices=PriceTable(),
        redis=None,
        tenant_limits={"team-a": 1.0},
    )
    assert tracker.check_budget("team-a", 100.0) is True  # no redis, no limit


class FakeRedis:
    """Minimal Redis stub for budget tests."""

    def __init__(self):
        self.store = {}

    def incrbyfloat(self, key, amount):
        self.store[key] = self.store.get(key, 0.0) + amount
        return self.store[key]

    def expire(self, key, ttl):
        return True


def test_budget_enforcement_allows_under_limit():
    redis = FakeRedis()
    tracker = CostTracker(
        prices=PriceTable(),
        redis=redis,
        tenant_limits={"team-a": 1.0},
    )
    assert tracker.check_budget("team-a", 0.5) is True
    assert tracker.check_budget("team-a", 0.4) is True  # total 0.9, under 1.0


def test_budget_enforcement_rejects_over_limit():
    redis = FakeRedis()
    tracker = CostTracker(
        prices=PriceTable(),
        redis=redis,
        tenant_limits={"team-a": 1.0},
    )
    assert tracker.check_budget("team-a", 0.6) is True  # 0.6
    assert tracker.check_budget("team-a", 0.5) is False  # would be 1.1, over 1.0
    # rejected spend should be rolled back - bucket stays at 0.6 (with float tolerance)
    assert round(list(redis.store.values())[0], 9) == 0.6


def test_full_request_flow():
    """End-to-end: trace, record, get cost back."""
    table = PriceTable.load(os.path.join(os.path.dirname(__file__), "prices.yaml"))
    tracker = CostTracker(prices=table, redis=FakeRedis(), tenant_limits={"team-a": 1.0})

    with tracker.trace(tenant_id="team-a", prompt_version="v3") as span:
        # pre-call budget check
        ok = span.estimate_and_check(model="gpt-4o", prompt_tokens=1500, expected_completion=300)
        assert ok is True
        # record the actual result
        cost = span.record(
            model="gpt-4o",
            prompt_tokens=1500,
            completion_tokens=300,
            cached_tokens=800,
            finish_reason="stop",
            ttft_ms=420,
            itl_ms=35,
        )
        # gpt-4o (post Oct 2024): input 0.0025, output 0.010, cached 0.00125
        # billable_input = 700, cached = 800, output = 300
        expected = 700 * 0.0025 / 1000 + 300 * 0.010 / 1000 + 800 * 0.00125 / 1000
        assert abs(cost - expected) < 1e-9
        assert span.cost == cost


if __name__ == "__main__":
    # Run without pytest for quick verification
    tests = [
        test_compute_cost_basic,
        test_compute_cost_with_cache,
        test_compute_cost_free_cache,
        test_estimate_cost,
        test_price_table_versioning,
        test_budget_enforcement_no_redis,
        test_budget_enforcement_allows_under_limit,
        test_budget_enforcement_rejects_over_limit,
        test_full_request_flow,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
