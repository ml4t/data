# Massive Provider

**Provider**: `MassiveProvider`
**Legacy Alias**: `PolygonProvider`
**Website**: [massive.com](https://massive.com)
**API Key**: Required
**Free Tier**: 5 API calls/minute

---

## Overview

Massive, formerly Polygon.io, provides comprehensive market data across stocks,
options, futures, forex, and crypto with institutional-quality tick, quote,
reference, and aggregate bar data.

**Best For**: US equities, options research, futures reference data, tick data,
and multi-asset research workflows.

**Compatibility**: Existing Polygon.io API keys remain valid. New code should use
`MassiveProvider` and `MASSIVE_API_KEY`; `PolygonProvider` and `POLYGON_API_KEY`
remain supported for backward compatibility.

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
os.environ["MASSIVE_API_KEY"] = "your_key_here"

from ml4t.data.providers import MassiveProvider

provider = MassiveProvider()

# Stocks
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01", frequency="daily")

# Options
options = provider.fetch_ohlcv(
    "O:SPY240119C00480000",
    "2024-01-01",
    "2024-01-19",
    frequency="daily",
)

# Futures use Massive's futures endpoint; use a prefix or explicit asset class
futures = provider.fetch_ohlcv(
    "F:ESM6",
    "2024-01-01",
    "2024-01-31",
    frequency="daily",
)

# Crypto and forex
btc = provider.fetch_ohlcv("X:BTCUSD", "2024-01-01", "2024-01-31", frequency="daily")
eurusd = provider.fetch_ohlcv("C:EURUSD", "2024-01-01", "2024-01-31", frequency="daily")

provider.close()
```

---

## Asset Classes

| Asset Class | Symbol Format | REST Route |
|-------------|---------------|------------|
| Stocks | `AAPL` | aggregate bars |
| Options | `O:SPY240119C00480000` | aggregate bars |
| Futures | `F:ESM6` or `asset_class="futures"` | futures aggregate bars |
| Crypto | `X:BTCUSD` | aggregate bars |
| Forex | `C:EURUSD` | aggregate bars |

Unprefixed futures tickers can be ambiguous with equities. Use the `F:` prefix
or pass `asset_class="futures"` when calling `fetch_ohlcv()`.

---

## Supported Frequencies

| Frequency | Free Tier | Paid Tier |
|-----------|-----------|-----------|
| `daily` | Yes (2yr) | Yes (20yr+) |
| `1h` | No | Yes |
| `1m` | No | Yes |
| Tick | No | Yes (Developer+) |

---

## Coverage

- **Stocks**: All US exchanges (NYSE, NASDAQ, etc.)
- **Options**: Full OPRA data (Advanced tier)
- **Futures**: CME, CBOT, COMEX, NYMEX contracts, products, schedules, and bars
- **Crypto**: Major cryptocurrencies
- **Forex**: Major pairs
- **Indices**: Major US indices

---

## API Key Setup

```bash
# .env file
MASSIVE_API_KEY=your_api_key_here
```

Get your API key at [massive.com](https://massive.com).

Existing users can continue to use:

```bash
POLYGON_API_KEY=your_existing_polygon_key
```

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
- [Massive REST Docs](https://massive.com/docs)
- [Polygon compatibility note](polygon.md)
- [Provider Audit](PROVIDER_AUDIT.md)
