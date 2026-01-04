# Yahoo Finance Provider

**Provider**: `YahooFinanceProvider`
**API Wrapper**: [yfinance](https://pypi.org/project/yfinance/)
**API Key**: Not required
**Free Tier**: Unlimited (personal use only)

---

## Overview

Yahoo Finance provides free access to US equities, ETFs, crypto, and forex data. It's the recommended starting point for learning and prototyping.

**Best For**: Quick start, US equities, personal research

**Limitations**:
- Personal use only (per Yahoo Terms of Service)
- Intraday data limited to 7 days
- Unofficial API (may change without notice)

---

## Quick Start

```python
from ml4t.data.providers import YahooFinanceProvider

provider = YahooFinanceProvider()

# Daily data
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01", frequency="daily")

# Intraday (last 7 days only)
df = provider.fetch_ohlcv("AAPL", "2024-12-15", "2024-12-20", frequency="1h")

provider.close()
```

---

## Supported Frequencies

| Frequency | History Limit | Notes |
|-----------|---------------|-------|
| `1m` | 7 days | Last 7 trading days only |
| `5m` | 60 days | |
| `15m` | 60 days | |
| `1h` | 730 days | |
| `daily` | 50+ years | |
| `weekly` | 50+ years | |

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

- ~2,000 requests/hour (informal, IP-based)
- No official rate limit documentation
- Recommend 0.5-1 second delay between calls for bulk downloads

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
| Financial statements | HIGH | Coming soon |
| Earnings data | MEDIUM | |
| Analyst recommendations | LOW | |
| Holders data | LOW | |

---

## See Also

- [Provider README](README.md) - All providers
- [Provider Audit](PROVIDER_AUDIT.md) - Detailed capabilities
