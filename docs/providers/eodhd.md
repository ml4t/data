# EODHD Provider

**Provider**: `EODHDProvider`
**Website**: [eodhd.com](https://eodhd.com)
**API Key**: Required
**Free Tier**: 20 API calls/day

---

## Overview

EODHD (End of Day Historical Data) provides affordable access to global equities across 60+ exchanges with 30+ years of history.

**Best For**: Global equities, affordable production data

**Pricing**:
| Tier | Price | Features |
|------|-------|----------|
| Free | $0/mo | 20 calls/day, 1 year history |
| Paid plans | See current pricing | Expanded history, markets, and endpoints |

---

## Quick Start

```python
import os
os.environ["EODHD_API_KEY"] = "your_key_here"

from ml4t.data.providers import EODHDProvider

provider = EODHDProvider()

# US stocks use .US suffix
df = provider.fetch_ohlcv("AAPL.US", "2024-01-01", "2024-12-01")

# International stocks
df = provider.fetch_ohlcv("VOW3.DE", "2024-01-01", "2024-12-01")  # Volkswagen (Germany)
df = provider.fetch_ohlcv("7203.T", "2024-01-01", "2024-12-01")   # Toyota (Japan)

provider.close()
```

---

## Symbol Format

| Exchange | Suffix | Examples |
|----------|--------|----------|
| US (NYSE, NASDAQ) | .US | AAPL.US, MSFT.US |
| Germany (XETRA) | .DE | SAP.DE, VOW3.DE |
| UK (LSE) | .LSE | HSBA.LSE, BP.LSE |
| Japan (TSE) | .T | 7203.T, 9984.T |
| Hong Kong | .HK | 0700.HK, 9988.HK |
| Australia | .AU | BHP.AU, CBA.AU |

See [EODHD Exchange List](https://eodhd.com/financial-apis/exchanges-api-list-of-tickers-and-டexchange-codes) for all 60+ exchanges.

---

## Supported Frequencies

| Frequency | Availability | Tier Required |
|-----------|--------------|---------------|
| `daily` | ✅ | Free |
| `weekly` | ✅ | Free |
| `monthly` | ✅ | Free |
| `1m`, `5m`, `1h` | ✅ | A plan with intraday access |

---

## Coverage

- **Exchanges**: 60+ worldwide
- **Tickers**: 150,000+
- **History**: 30+ years
- **Asset Types**: Stocks, ETFs, Mutual Funds, Forex, Crypto

---

## API Key Setup

```bash
# Environment variable
export EODHD_API_KEY=your_api_key_here
```

Get your API key at [eodhd.com/register](https://eodhd.com/register).

---

## Rate Limits

| Tier | Limit |
|------|-------|
| Free | 20 calls/day |
| Paid | 100,000 calls/day |

---

## Not Yet Implemented

| Feature | Tier Required | Priority |
|---------|---------------|----------|
| Fundamentals | Paid endpoint access | HIGH |
| Intraday data | Paid endpoint access | MEDIUM |
| Options (Marketplace) | Marketplace access | HIGH |
| Earnings data | Included | MEDIUM |
| Insider transactions | Included | LOW |

---

## See Also

- [EODHD Pricing](https://eodhd.com/pricing)
- [Provider reference](index.md)
