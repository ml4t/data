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
from datetime import UTC, datetime, timedelta

from ml4t.data.providers import CoinGeckoProvider

provider = CoinGeckoProvider()
end = datetime.now(UTC).date() - timedelta(days=1)
start = end - timedelta(days=6)

# Use CoinGecko IDs (not ticker symbols)
df = provider.fetch_ohlcv("bitcoin", str(start), str(end), frequency="daily")
df = provider.fetch_ohlcv("ethereum", str(start), str(end), frequency="daily")

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

The OHLC endpoint supplies daily source candles only for the most recent 30 days. Older
requests fail with `DataValidationError` because CoinGecko returns four-day candles,
which cannot be converted into correct daily OHLC values.

---

## Rate Limits

- Free: 50 calls/minute
- Demo API key: 50 calls/minute (recommended)
- Pro: $129+/mo for higher limits

---

## See Also

- [CoinGecko API](https://www.coingecko.com/en/api/pricing)
- [Provider README](README.md)
