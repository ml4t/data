# Tiingo Provider

**Provider**: `TiingoProvider`
**Website**: [tiingo.com](https://tiingo.com)
**API Key**: Required
**Free Tier**: 1,000 requests/day

---

## Overview

Tiingo provides US equity data with a generous free tier, good for alternative data source or Yahoo Finance backup.

**Best For**: US equities alternative, redundancy

---

## Quick Start

```python
import os
os.environ["TIINGO_API_KEY"] = "your_key_here"

from ml4t.data.providers import TiingoProvider

provider = TiingoProvider()
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01", frequency="daily")
provider.close()
```

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `daily` | ✅ |
| `1h` | ✅ |
| `1m` | ✅ (limited) |

---

## API Key Setup

```bash
# .env file
TIINGO_API_KEY=your_api_key_here
```

Get your API key at [tiingo.com/account/api/token](https://api.tiingo.com/account/api/token).

---

## Rate Limits

- Free: 1,000 requests/day, 500 unique symbols/month
- Paid tiers available for higher limits

---

## See Also

- [Tiingo Pricing](https://tiingo.com/about/pricing)
- [Provider README](README.md)
