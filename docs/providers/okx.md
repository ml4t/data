# OKX Provider

**Provider**: `OKXProvider`
**Website**: [okx.com](https://www.okx.com)
**API Key**: Not required for public market data
**Free Tier**: No geo-restrictions

---

## Overview

OKX provides cryptocurrency perpetual swap data with funding rates. Unlike Binance, OKX has no geo-restrictions, making it accessible globally.

**Best For**: Crypto perpetuals, funding rate analysis

---

## Quick Start

```python
from ml4t.data.providers import OKXProvider

provider = OKXProvider()

# Perpetual swap OHLCV
df = provider.fetch_ohlcv("BTC-USDT-SWAP", "2024-01-01", "2024-06-30", frequency="daily")

# Funding rates
rates = provider.fetch_funding_rates("BTC-USDT-SWAP", "2024-01-01", "2024-06-30")

provider.close()
```

---

## Symbol Format

| Type | Format | Examples |
|------|--------|----------|
| Perpetual Swap | BASE-QUOTE-SWAP | BTC-USDT-SWAP, ETH-USDT-SWAP |

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `1m` | Yes |
| `5m` | Yes |
| `15m` | Yes |
| `1h` | Yes |
| `daily` | Yes |

---

## Async Support

OKX uses native async via `httpx.AsyncClient`:

```python
async with OKXProvider() as provider:
    df = await provider.fetch_ohlcv_async("BTC-USDT-SWAP", "2024-01-01", "2024-06-30")
```

---

## See Also

- [OKX API v5 Documentation](https://www.okx.com/docs-v5/en/)
- [Provider README](README.md)
