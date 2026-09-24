# Massive provider

`MassiveProvider` retrieves aggregate bars for stocks, options, futures, crypto, and foreign
exchange. It also retrieves stock financial statements and company ratios. A Massive account and
API key are required; accessible datasets, history, and request quotas depend on the account.

## Configure access

Set `MASSIVE_API_KEY` or pass the key directly:

```bash
export MASSIVE_API_KEY=your_api_key
```

Accounts created under the former Polygon.io name can continue to use `POLYGON_API_KEY`. If both
variables are set, `MASSIVE_API_KEY` takes precedence.

## Fetch aggregate bars

```python
from ml4t.data.providers import MassiveProvider

provider = MassiveProvider()
try:
    bars = provider.fetch_ohlcv(
        "AAPL",
        "2024-01-01",
        "2024-01-31",
        frequency="daily",
    )
finally:
    provider.close()
```

The provider accepts daily, weekly, monthly, hourly, and minute aliases. Symbols select the asset
route as follows:

| Asset class | Symbol example | Routing rule |
|---|---|---|
| Stocks | `AAPL` | Default for an unprefixed symbol |
| Options | `O:SPY240119C00480000` | `O:` prefix |
| Futures | `F:ESM6` | `F:` or `FUT:` prefix |
| Crypto | `X:BTCUSD` | `X:` prefix |
| Foreign exchange | `C:EURUSD` | `C:` prefix |

An unprefixed futures symbol is ambiguous with an equity symbol. Pass
`asset_class="futures"` or add the `F:` prefix.

## Fetch fundamentals

`fetch_financials()` returns the shared long-form statement schema. Supported statements are
`income`, `balance`, and `cashflow`; supported periods are `annual` and `quarterly`. Income and cash
flow statements also accept `ttm`.

```python
provider = MassiveProvider()
try:
    income = provider.fetch_financials(
        "AAPL",
        statement="income",
        period="quarterly",
        limit=8,
    )
    ratios = provider.fetch_company_metrics("AAPL")
finally:
    provider.close()
```

Results are ordered most recent first, and `limit` caps the number of statement periods. The
`filed_at` field is the most recent SEC filing that included a period. A later filing can repeat or
restate an earlier period, so this field does not establish when a value first became available.
Do not treat these results as point-in-time data without an independent filing-history check.

## Rate limiting and failures

The default client pace is five calls per minute. This is a conservative local setting, not a
statement about the service plan. Pass `rate_limit=(calls, period_seconds)` to use a lower pace
required by an account or workload.

Authentication failures raise `AuthenticationError`; HTTP rate limits raise `RateLimitError` with
the reported retry delay; missing symbols raise `SymbolNotFoundError`; other transport and response
failures use the shared provider exception types.

See the current [Massive API documentation](https://massive.com/docs) and
[account plans](https://massive.com/pricing) for service-side coverage and limits.
