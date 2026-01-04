# CryptoCompare Provider

**Provider**: `CryptoCompareProvider`
**Website**: [cryptocompare.com](https://www.cryptocompare.com)
**API Key**: Required
**Free Tier**: 250,000 calls/month

---

## Overview

CryptoCompare provides comprehensive cryptocurrency historical data with good coverage and reasonable free tier.

**Best For**: Crypto historical data, alternative to Binance

---

## Quick Start

```python
import os
os.environ["CRYPTOCOMPARE_API_KEY"] = "your_key_here"

from ml4t.data.providers import CryptoCompareProvider

provider = CryptoCompareProvider()
df = provider.fetch_ohlcv("BTC", "2024-01-01", "2024-12-01", frequency="daily")
provider.close()
```

---

## Symbol Format

Use base currency symbols:
- `BTC`, `ETH`, `SOL`, `ADA`, etc.

Quote currency defaults to USD.

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `1m` | ✅ |
| `1h` | ✅ |
| `daily` | ✅ |

---

## API Key Setup

```bash
# .env file
CRYPTOCOMPARE_API_KEY=your_api_key_here
```

Get your API key at [cryptocompare.com/cryptopian/api-keys](https://www.cryptocompare.com/cryptopian/api-keys).

---

## Rate Limits

- Free: 250,000 calls/month
- Paid: $19.99+/mo for higher limits

---

## See Also

- [CryptoCompare Pricing](https://min-api.cryptocompare.com/pricing)
- [Provider README](README.md)
