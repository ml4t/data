# Polygon / Massive Provider

**Provider**: `PolygonProvider`
**Website**: [massive.com](https://massive.com) (formerly polygon.io)
**API Key**: Required
**Free Tier**: 5 API calls/minute

---

## Overview

Polygon (now Massive) provides comprehensive US market data including stocks, options, and crypto with institutional-quality tick data.

**Best For**: US equities, options research, tick data

**Note**: Polygon.io rebranded to Massive.com in 2025.

**Pricing**:
| Tier | Price | Features |
|------|-------|----------|
| Basic (Free) | $0/mo | 5 calls/min, 2yr history, EOD |
| Starter | $29/mo | Unlimited calls, 5yr history, 15min delayed |
| Developer | $79/mo | 10yr history, trades data |
| Advanced | $199/mo | 20yr+ history, real-time, quotes, financials |

---

## Quick Start

```python
import os
os.environ["POLYGON_API_KEY"] = "your_key_here"

from ml4t.data.providers import PolygonProvider

provider = PolygonProvider()

# Daily data
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01", frequency="daily")

# Minute data (requires paid tier)
df = provider.fetch_ohlcv("AAPL", "2024-12-01", "2024-12-15", frequency="1m")

provider.close()
```

---

## Supported Frequencies

| Frequency | Free Tier | Paid Tier |
|-----------|-----------|-----------|
| `daily` | ✅ (2yr) | ✅ (20yr+) |
| `1h` | ❌ | ✅ |
| `1m` | ❌ | ✅ |
| Tick | ❌ | ✅ (Developer+) |

---

## Coverage

- **Stocks**: All US exchanges (NYSE, NASDAQ, etc.)
- **Options**: Full OPRA data (Advanced tier)
- **Crypto**: Major cryptocurrencies
- **Forex**: Major pairs
- **Indices**: Major US indices

---

## API Key Setup

```bash
# .env file
POLYGON_API_KEY=your_api_key_here
```

Get your API key at [massive.com](https://massive.com).

---

## Rate Limits

| Tier | Limit |
|------|-------|
| Basic | 5 calls/minute |
| Starter | Unlimited |
| Developer+ | Unlimited |

---

## Not Yet Implemented

| Feature | Tier Required | Priority |
|---------|---------------|----------|
| Options chains | Advanced | HIGH |
| Options Greeks | Advanced | HIGH |
| Financials | Advanced | HIGH |
| Trades (tick) | Developer | MEDIUM |
| Quotes (NBBO) | Developer | MEDIUM |
| WebSockets | Any | NOT PLANNED |

---

## See Also

- [Massive Pricing](https://massive.com/pricing)
- [Provider README](README.md)
- [Provider Audit](PROVIDER_AUDIT.md)
