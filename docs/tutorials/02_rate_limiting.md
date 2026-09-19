# Work within provider rate limits

Provider quotas vary by account, endpoint, and date. Treat the provider's current documentation and
account dashboard as authoritative. The defaults in `ml4t-data` are conservative client-side
request pacing, not a statement of the account's total quota.

## Understand the two limits

A provider may enforce both:

- a short-window request rate, such as calls within a minute; and
- a daily, monthly, credit, or cost budget.

Client pacing can enforce the first constraint. It cannot infer the second from elapsed time, so a
recurring workflow must also track its account usage and requested data volume.

## Use provider defaults

Providers derived from `BaseProvider` initialize a shared rate limiter. Instances with the same
provider name share that limiter within the process, which prevents separate objects from each
using the complete short-window allowance.

```python
from ml4t.data.providers import SyntheticProvider

provider = SyntheticProvider(seed=42)
data = provider.fetch_ohlcv("SYNTH", "2024-01-01", "2024-01-10", "daily")
```

The synthetic provider performs no network request. Network-backed providers acquire their limiter
before issuing a request.

## Override pacing when the account requires it

A provider constructor accepts `rate_limit=(calls, period_seconds)` through `BaseProvider`:

```python
from ml4t.data.providers import YahooFinanceProvider

provider = YahooFinanceProvider(rate_limit=(10, 60.0))
```

This example permits ten acquisitions in a rolling 60-second period. It does not grant provider
access or increase an external quota. Choose the value from the current provider terms for the
account and endpoint in use.

## Reduce avoidable requests

Use the storage-backed workflow in [incremental updates](03_incremental_updates.md) instead of
requesting complete history repeatedly. It determines the missing date range before calling the
provider.

Before scheduling a large universe:

1. Confirm the provider's current quota and billing unit.
2. Test one symbol at the required frequency and date range.
3. Measure how the provider paginates that request.
4. Estimate total requests or credits for the universe.
5. Store successful responses and request only missing ranges.
6. Leave capacity for retries and operational checks.

## Handle failures by type

A rate-limit response is different from authentication failure, unavailable data, or a transient
network error. Catch `RateLimitError` when the workflow can reschedule from its `retry_after`
value. Treat `AuthenticationError` and `DataNotAvailableError` as input or account problems. Retry
`NetworkError` only when its `retryable` attribute permits it.

Do not turn authentication or unavailable-data errors into automatic retries. Repeating those
requests consumes quota without changing the outcome. The [exception reference](../api/index.md)
defines these error types.

## Continue

- [Incremental updates](03_incremental_updates.md)
- [Provider selection](../getting-started/provider-selection.md)
- [Provider reference](../providers/index.md)
