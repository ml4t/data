# utils/ - 1.6k Lines

Rate limiting, gap detection, and retry utilities.

## Modules

| File | Purpose |
|------|---------|
| rate_limit.py | Per-provider rate limits |
| async_rate_limit.py | Async rate limiting |
| global_rate_limit.py | Global rate coordinator |
| retry.py | Retry with backoff |
| gaps.py | Gap detection |
| gap_optimizer.py | Gap merge optimization |
| locking.py | File locking |
| format.py | Data formatting |

## Key

`RateLimiter`, `detect_gaps()`, `retry_with_backoff()`
