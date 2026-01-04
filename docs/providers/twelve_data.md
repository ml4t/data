# TwelveData Provider

**Provider**: `TwelveDataProvider`
**Website**: [twelvedata.com](https://twelvedata.com)
**API Key**: Required
**Free Tier**: 800 API calls/day

---

## Overview

TwelveData provides multi-asset coverage including stocks, forex, and crypto with a generous free tier.

**Best For**: Multi-asset coverage, alternative data source

---

## Quick Start

```python
import os
os.environ["TWELVE_DATA_API_KEY"] = "your_key_here"

from ml4t.data.providers import TwelveDataProvider

provider = TwelveDataProvider()
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01", frequency="daily")
provider.close()
```

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `daily` | ✅ |
| `weekly` | ✅ |
| `1h` | ✅ |
| `1m` | ✅ |

---

## Coverage

- US and international stocks
- Forex pairs
- Cryptocurrencies
- ETFs

---

## API Key Setup

```bash
# .env file
TWELVE_DATA_API_KEY=your_api_key_here
```

Get your API key at [twelvedata.com/account](https://twelvedata.com/account).

---

## Rate Limits

- Free: 800 API calls/day, 8 calls/minute
- Paid tiers available

---

## See Also

- [TwelveData Pricing](https://twelvedata.com/pricing)
- [Provider README](README.md)
