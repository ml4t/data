# Binance Bulk Provider

**Provider**: `BinanceBulkProvider`
**Website**: [data.binance.vision](https://data.binance.vision)
**API Key**: Not required
**Free Tier**: Unlimited

---

## Overview

BinanceBulkProvider downloads bulk historical data from Binance's public data repository. No API key or authentication required.

**Best For**: Bulk crypto historical downloads, no geo-restrictions

---

## Quick Start

```python
from ml4t.data.providers import BinanceBulkProvider

provider = BinanceBulkProvider()

# Download historical data
df = provider.fetch_ohlcv("BTCUSDT", "2020-01-01", "2024-12-01", frequency="daily")
df = provider.fetch_ohlcv("ETHUSDT", "2023-01-01", "2024-12-01", frequency="1h")

provider.close()
```

---

## Data Source

Data is downloaded from [data.binance.vision](https://data.binance.vision), Binance's public data archive:

- Pre-aggregated OHLCV data
- No rate limits
- No geo-restrictions
- Monthly/daily zip files

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `1m` | ✅ |
| `5m` | ✅ |
| `15m` | ✅ |
| `1h` | ✅ |
| `daily` | ✅ |

---

## Symbol Format

Same as Binance:
- `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, etc.

---

## Advantages Over BinanceAPIProvider

| Feature | Binance Bulk | Binance API |
|---------|---------------|---------|
| Geo-restrictions | None | Yes (some countries) |
| Rate limits | None | 1,200/min |
| Bulk downloads | Optimized | Standard API |
| Authentication | Not required | Optional |

---

## See Also

- [Binance Data Portal](https://data.binance.vision)
- [Binance API Provider](binance_api.md)
- [Provider Index](index.md)
