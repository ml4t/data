# CoinGecko Provider

**Provider**: `CoinGeckoProvider`
**Website**: [coingecko.com](https://www.coingecko.com)
**API Key**: Not required (Demo key available)
**Free Tier**: 50 calls/minute

---

## Overview

CoinGecko provides comprehensive cryptocurrency market data with broad coverage and no authentication required.

**Best For**: Crypto market overview, simple API

---

## Quick Start

```python
from ml4t.data.providers import CoinGeckoProvider

provider = CoinGeckoProvider()

# Use CoinGecko IDs (not ticker symbols)
df = provider.fetch_ohlcv("bitcoin", "2024-01-01", "2024-12-01", frequency="daily")
df = provider.fetch_ohlcv("ethereum", "2024-01-01", "2024-12-01", frequency="daily")

provider.close()
```

---

## Symbol Format

CoinGecko uses coin IDs, not ticker symbols:

| Coin | CoinGecko ID |
|------|--------------|
| Bitcoin | `bitcoin` |
| Ethereum | `ethereum` |
| Solana | `solana` |
| Cardano | `cardano` |

Find IDs at [coingecko.com/api/documentation](https://www.coingecko.com/api/documentation).

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `daily` | ✅ |
| Intraday | ❌ |

---

## Rate Limits

- Free: 50 calls/minute
- Demo API key: 50 calls/minute (recommended)
- Pro: $129+/mo for higher limits

---

## See Also

- [CoinGecko API](https://www.coingecko.com/en/api/pricing)
- [Provider README](README.md)
