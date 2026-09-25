# Tiingo Provider

**Provider**: `TiingoProvider`
**Website**: [tiingo.com](https://tiingo.com)
**API Key**: Required
**Free Tier**: 1,000 requests/day

---

## Overview

Tiingo provides authenticated US equity data. Account plans determine quotas and historical depth.

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
# Environment variable
export TIINGO_API_KEY=your_api_key_here
```

Get your API key at [tiingo.com/account/api/token](https://api.tiingo.com/account/api/token).

---

## Rate Limits

- Free: 1,000 requests/day, 500 unique symbols/month
- Paid tiers available for higher limits

---

## See Also

- [Equity](equities.md) and [ETF](etfs.md) source references
- [Tiingo Pricing](https://tiingo.com/about/pricing)
- [Provider reference](index.md)
