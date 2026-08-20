# Section 01 - Cost Tracking

> Your LLM serving bill is probably 10x what it should be, and you do not
> know it because you are not tracking cost per token correctly.

## The problem

Most teams track LLM cost at the billing level: a monthly invoice from
OpenAI, Anthropic, or AWS. By the time that invoice arrives, the damage is
done. A single prompt template that grew by 500 tokens, a model downgrade
that silently failed and fell back to the expensive model, a caching layer
that stopped hitting - any of these can multiply your spend by 10x in a
day, and you will not know until the finance team asks why the bill doubled.

This is not a hypothetical. I have seen each of these happen in production.

The fix is per-request, per-tenant cost tracking with OpenTelemetry, exposed
as a histogram that you can alert on. Not a billing dashboard. A real-time
cost signal that fires before the invoice arrives.

## What you need to track

### Per-request metrics

Every LLM call should emit these attributes on a trace span:

| Attribute | Why |
|-----------|-----|
| `gen_ai.request.model` | Which model handled the request (cost varies by model) |
| `gen_ai.usage.prompt_tokens` | Input tokens, drives input cost |
| `gen_ai.usage.completion_tokens` | Output tokens, drives output cost |
| `gen_ai.usage.cached_tokens` | Cached prefix tokens, usually free or discounted |
| `gen_ai.latency.time_to_first_token` | TTFT, user-visible latency |
| `gen_ai.latency.time_per_output_token` | ITL, inter-token latency |
| `gen_ai.response.finish_reason` | stop, length, tool_calls, content_filter |
| `tenant_id` (low cardinality) | Per-tenant attribution for chargeback |
| `route` or `gateway` | Which model gateway or fallback path was used |
| `prompt_version` | Which prompt template version generated this call |

The `gen_ai.*` attributes follow the OpenInference semantic conventions
(Arize AI, now part of the OpenTelemetry GenAI working group). Use them
instead of inventing your own - your observability backend will understand
them natively.

### Aggregated metrics (histograms)

From the per-request spans, export these histograms:

```
gen_ai.client.cost            # USD per request, histogram
gen_ai.client.token_usage     # tokens per request, histogram
gen_ai.client.operation.duration  # end-to-end duration, histogram
gen_ai.client.ttft            # time to first token, histogram
gen_ai.client.itl             # inter-token latency, histogram
```

Histograms, not counters, because you need percentiles. Average cost per
token is useless - you need p50, p95, p99 to understand the distribution.
A few expensive requests can dominate spend while the average looks fine.

## The cost calculation

Cost per request is not just `tokens * price`. It is:

```
cost = (prompt_tokens - cached_tokens) * input_price
     + completion_tokens * output_price
     + cached_tokens * cache_price          # usually 0 or 0.1x input price
```

The `cached_tokens` term is the one most teams miss. Anthropic, OpenAI, and
vLLM's prefix caching all discount or free cached prefix tokens. If you
count them at full price, your cost numbers are wrong by the cache hit rate
(often 30-60% for RAG workloads).

### Price snapshots

Model prices change. OpenAI has changed pricing multiple times in 2024-2025.
Your cost tracker needs a price table that is:

1. Versioned (each price has an effective date)
2. Loaded from a config file, not hardcoded
3. Updated when providers announce changes

Do not hardcode prices in your tracking code. I have seen teams ship cost
dashboards that were wrong for 3 months because they hardcoded GPT-4
pricing and OpenAI cut the price.

## Budget enforcement

Tracking cost is necessary but not sufficient. You also need to enforce
budgets before the request hits the model.

### Per-tenant hourly budgets

```python
# Pseudocode - see code/cost_tracker/ for full impl
def check_budget(tenant_id: str, estimated_cost: float) -> bool:
    spent = redis.incrbyfloat(
        f"budget:{tenant_id}:{hour_bucket()}",
        estimated_cost
    )
    if spent > tenant_limit(tenant_id):
        return False  # reject or downgrade
    return True
```

The key design decisions:

1. **Hourly buckets, not monthly.** Monthly budgets let a tenant burn the
   whole budget in 1 hour. Hourly buckets spread spend evenly.
2. **Estimate before the call.** Use `prompt_tokens * input_price` as the
   estimate. You cannot know completion tokens ahead of time, so estimate
   based on max_tokens or a per-tenant historical average.
3. **Reject or downgrade, do not just alert.** If you only alert, the spend
   already happened. Downgrade to a cheaper model instead of rejecting when
   possible.

### Chargeback

Per-tenant cost attribution enables chargeback: each team or customer sees
their actual LLM spend. This is the single most effective tool for cost
control I have seen. When teams see their own spend in real time, they
optimize their prompts. When they do not, they send 10K-token contexts to
GPT-4 without thinking.

## The 10x spend traps

These are the patterns I have seen multiply LLM spend by 10x in production.
Each is preventable with the tracking above.

### 1. Prompt template bloat

A prompt template that includes a large context document "just in case" can
add 2K tokens to every request. At 10K requests/day on GPT-4-class pricing,
that is roughly $200/day of waste. The fix: track `prompt_tokens` per
`prompt_version` and alert when a new version adds significant tokens.

### 2. Silent fallback to expensive model

Your model gateway falls back to the expensive model when the cheap one is
unavailable. If the cheap model is down for an hour, all traffic routes to
the expensive one and your spend spikes 5-10x. The fix: track `route` and
alert on fallback rate. Set a fallback budget separate from normal budget.

### 3. Cache miss storm

Your prefix cache stops hitting because the prompt prefix changed (someone
added a timestamp to the system prompt). Cache hit rate drops from 60% to
0%, and your input token cost jumps 2.5x. The fix: track
`cached_tokens / prompt_tokens` ratio and alert on sudden drops.

### 4. Multi-turn explosion

A bug triggers multi-turn conversations where single-turn was expected. Each
turn re-sends the full history, so a 5-turn conversation costs 15x a
single-turn. The fix: track `turn_count` per session and alert on
distribution shifts.

### 5. Streaming truncation

Responses get truncated (finish_reason: length) but still count as
successful API calls. The user retries, doubling cost. The fix: track
`finish_reason` distribution and alert on `length` rate increases.

## Reference implementation

See [code/cost_tracker/](../../code/cost_tracker/) for a working
OpenTelemetry-based cost tracker that implements:

- Per-request span emission with OpenInference attributes
- Price table with versioned model pricing
- Per-tenant hourly budget enforcement via Redis
- Cost histogram export for Prometheus
- Alert rules for the 5 spend traps above

## Sources

- OpenTelemetry GenAI semantic conventions
- "How to Track Token Usage, Prompt Costs, and Model Latency with
  OpenTelemetry" - OneUptime blog
- "LLM Observability & Monitoring" - CalibreOS
- "LLM API Observability: Metrics, Traces, Logs, and Cost" - flatkey.ai
- "What You Cannot See Will Break Your LLM App" - DevOps.com
- Production experience: cost tracking across multi-model LLM gateways
