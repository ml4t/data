# Binance Provider

**Provider**: `BinanceProvider`
**Website**: [binance.com](https://www.binance.com)
**API Key**: Not required for market data
**Free Tier**: Generous rate limits

---

## Overview

Binance provides comprehensive crypto market data for spot and futures markets.

**Best For**: Crypto spot and futures, high-frequency data

**Note**: May have geo-restrictions in certain countries (US).

---

## Quick Start

```python
from ml4t.data.providers import BinanceProvider

provider = BinanceProvider()

# USDT pairs
df = provider.fetch_ohlcv("BTCUSDT", "2024-01-01", "2024-12-01", frequency="daily")
df = provider.fetch_ohlcv("ETHUSDT", "2024-01-01", "2024-12-01", frequency="1h")

provider.close()
```

---

## Symbol Format

| Type | Format | Examples |
|------|--------|----------|
| Spot | BASEUSDT | BTCUSDT, ETHUSDT |
| Spot | BASEBUSD | BTCBUSD |
| Futures | BASEUSDT | BTCUSDT (on futures API) |

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `1m` | ✅ |
| `5m` | ✅ |
| `15m` | ✅ |
| `1h` | ✅ |
| `daily` | ✅ |
| `weekly` | ✅ |

---

## Rate Limits

Binance has generous rate limits for market data:
- 1,200 requests/minute for most endpoints
- Weight-based system (check API docs)

---

## Geo-Restrictions

Binance may block access from certain countries. Consider using `BinancePublicProvider` for bulk historical downloads without restrictions.

---

## See Also

- [Binance API](https://www.binance.com/en/binance-api)
- [BinancePublic Provider](binance_public.md)
- [Provider README](README.md)
