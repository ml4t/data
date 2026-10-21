# Yahoo Finance Provider

**Provider**: `YahooFinanceProvider`
**API Wrapper**: [yfinance](https://pypi.org/project/yfinance/)
**API Key**: Not required
**Access**: Public service with provider-imposed limits

---

## Overview

Yahoo Finance provides free access to US equities, ETFs, crypto, and forex data. It's the recommended starting point for learning and prototyping.

**Best For**: Quick start, US equities, personal research

**Limitations**:
- Personal use only (per Yahoo Terms of Service)
- Intraday availability depends on interval and source access limits
- Unofficial API (may change without notice)

---

## Quick Start

```python
from ml4t.data.providers import YahooFinanceProvider

provider = YahooFinanceProvider()

# Daily data
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01", frequency="daily")

# Hourly data; availability depends on the source's current history window
df = provider.fetch_ohlcv("AAPL", "2024-12-15", "2024-12-20", frequency="hourly")

provider.close()
```

---

## Supported Frequencies

| Provider frequency | SDK interval |
|---|---|
| `minute` | `1m` |
| `5minute` | `5m` |
| `15minute` | `15m` |
| `hourly` | `1h` |
| `daily` | `1d` |
| `weekly` | `1wk` |

History windows are imposed upstream, not guaranteed by Data. See the
[SDK download contract](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
for interval and request semantics; unavailable ranges can fail.

---

## Symbol Format

| Asset Class | Format | Examples |
|-------------|--------|----------|
| US Stocks | TICKER | AAPL, MSFT, GOOGL |
| ETFs | TICKER | SPY, QQQ, IWM |
| Crypto | TICKER-USD | BTC-USD, ETH-USD |
| Forex | PAIR=X | EURUSD=X, GBPUSD=X |
| Indices | ^TICKER | ^GSPC, ^DJI, ^IXIC |

---

## Data Adjustments

- **Split Adjusted**: Yes (automatic)
- **Dividend Adjusted**: Yes (automatic)

---

## Rate Limits

- No official rate limit documentation
- Reuse provider instances and the shared rate-limit/recovery path for bulk downloads

## Adjusted-close acquisition and legacy bounds

```python
with YahooFinanceProvider() as provider:
    prices = provider.fetch_adjusted_close("SPY", start="2024-01-02", end="2024-01-06")
    provenance = provider.last_fetch_metadata
```

This daily scalar route requests `keepna=True` so reported all-missing price rows are not hidden
before return derivation. Existing OHLCV behavior remains unchanged. It forwards date/datetime bounds to
the SDK unchanged, including timezone-aware bounds; its end is exclusive. In contrast,
`fetch_ohlcv` retains its existing inclusive date end, translated to the next SDK day.
Both use `auto_adjust=True, actions=False`. Close is already adjusted for splits/dividends;
do not subtract dividends again or label it raw plus an independently validated action table.
Metadata records the actual SDK index name/unit, price basis and SDK version. A download time is
not historical publication evidence.

The [Empyrical acquisition adapter](../getting-started/migration.md#empyrical-acquisition-utilities)
returns the legacy one-column pandas return frame or named Series. It derives simple returns
without filling prices and rejects undefined infinite percentage returns.

---

## Terms of Service

**Important**: Yahoo Finance is intended for personal use only.

- [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
- [API Terms](https://policies.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.htm)

yfinance is NOT affiliated with Yahoo, Inc.

---

## Not Yet Implemented

These features are available via yfinance but not yet in ml4t-data:

| Feature | Priority | Notes |
|---------|----------|-------|
| Options chains | HIGH | Coming soon |
| Earnings data | MEDIUM | |
| Analyst recommendations | LOW | |
| Holders data | LOW | |

Financial statements and company metrics are already available through `fetch_financials` and
`fetch_company_metrics`; they are not historical benchmark-return data.

---

## See Also

- [Equity](equities.md) and [ETF](etfs.md) source references
- [Provider reference](index.md) - All providers
