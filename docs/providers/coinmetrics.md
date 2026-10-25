# Coin Metrics spot-market candles

`CoinMetricsProvider` acquires OHLCV candles for one explicit Coin Metrics spot market through
[`/v4/timeseries/market-candles`](https://docs.coinmetrics.io/api/v4/#operation/getTimeseriesMarketCandles).
It implements the standard provider API and the exact `TimestampRangeProvider` capability used by
[stored-gap recovery](../user-guide/incremental-updates.md#download-real-observations-for-a-selected-report).

## Access and identity

Install `ml4t-data`; no additional SDK is required. Pass `api_key` or set `COINMETRICS_API_KEY`.
The registry treats candle access as credentialed. Account entitlements determine available
markets, frequencies, historical coverage and quotas; a valid key alone does not guarantee access.
An instance without a key uses the community API, whose candle catalog was empty when checked
on October 23, 2026. Unauthorized candle requests raise `AuthenticationError`.

Use the exact lower-case `exchange-base-quote-spot` identifier, for example
`coinbase-btc-usd-spot`. Wildcards, lists, asset tickers and other instrument types are rejected
before acquisition. Canonical output retains this complete market identity in uppercase `symbol`.
Asset metrics, reference rates and other Coin Metrics schemas are separate datasets and are not
converted into candles.

## Request contract

| Operation or parameter | Contract |
|---|---|
| `fetch_ohlcv(symbol, start, end, frequency="daily")` | Inclusive UTC date strings; the end date includes its entire day |
| `fetch_ohlcv_range(symbol, start, end, frequency="minute")` | Aware datetime bounds, normalized to UTC; `start <= timestamp < end` |
| Frequencies | `1m`, `5m`, `10m`, `15m`, `30m`, `1h`, `4h`, `1d`; standard minute/hourly/daily aliases are accepted |
| Candle grid | UTC interval starts, left labels, no custom daily offset |
| `page_size` | 1-10000, default 1000 |
| Client pacing | Conservative default 60 requests/minute; account quotas still apply |
| Pagination | Follow the service's continuation token from `next_page_url`; preserve market, bounds and frequency; reject changed filters, foreign endpoints and repeated tokens |
| Output | UTC timestamp, symbol, open, high, low, close, base-asset volume; sorted, validated and bounded |
| Empty results | Typed empty frame; no fabricated price or volume, no retry merely because observations remain absent |

The provider reuses shared HTTP retries for transient failures and rate limits. Conflicting
duplicate candles and malformed or mismatched source records fail validation. Account errors
are reported without echoing credentials. Exact and daily range semantics also apply when using
`DataManager` with `provider="coinmetrics"`.
The shared `Frequency` values `5minute`, `15minute`, `30minute` and `hour`, and the `10min`
metadata spelling, map to the corresponding native candle intervals.

Coin Metrics generates candles only for intervals containing trades. A calendar gap therefore
identifies an absent scheduled observation for investigation; it does not prove a recoverable
vendor outage. Inspect the recovery result's `errors`, `rows_added` and `remaining_gaps`.

## Qualification boundary

Offline regressions exercise the public manager and the real storage/report/backfill workflow
using Coin Metrics' published response examples, including exact bounds, pagination, empty and
partial responses, identity preservation and residual gaps. Community access denial is verified
live. Successful authenticated acquisition against a particular paid dataset requires its account
entitlement and is not asserted by these tests.
