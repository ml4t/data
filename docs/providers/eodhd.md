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
| EOD All World | $19.99/mo | 100k calls/day, 30+ year history |
| EOD+Intraday | $29.99/mo | + Intraday data |
| Fundamentals | $59.99/mo | + Stock fundamentals |
| ALL-IN-ONE | $99.99/mo | Everything included |

**Note**: 50% student discount available.

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
| `1m`, `5m`, `1h` | ✅ | EOD+Intraday ($29.99) |

---

## Coverage

- **Exchanges**: 60+ worldwide
- **Tickers**: 150,000+
- **History**: 30+ years
- **Asset Types**: Stocks, ETFs, Mutual Funds, Forex, Crypto

---

## API Key Setup

```bash
# .env file
EODHD_API_KEY=your_api_key_here
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
| Fundamentals | $59.99/mo | HIGH |
| Intraday data | $29.99/mo | MEDIUM |
| Options (Marketplace) | $29.99/mo | HIGH |
| Earnings data | Included | MEDIUM |
| Insider transactions | Included | LOW |

---

## See Also

- [EODHD Pricing](https://eodhd.com/pricing)
- [Provider README](README.md)
- [Provider Audit](PROVIDER_AUDIT.md)
