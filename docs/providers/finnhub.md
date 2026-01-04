# Finnhub Provider

**Provider**: `FinnhubProvider`
**Website**: [finnhub.io](https://finnhub.io)
**API Key**: Required
**Free Tier**: 60 requests/minute

---

## Overview

Finnhub provides multi-asset market data with strong fundamentals and company metrics coverage.

**Best For**: Company metrics, analyst estimates, real-time quotes

**Pricing**:
| Tier | Price | Features |
|------|-------|----------|
| Free | $0/mo | 60 req/min, basic data |
| All-in-one | $49/mo | Premium features |
| Professional | Custom | Full access |

---

## Quick Start

```python
import os
os.environ["FINNHUB_API_KEY"] = "your_key_here"

from ml4t.data.providers import FinnhubProvider

provider = FinnhubProvider()
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01", frequency="daily")
provider.close()
```

---

## Supported Frequencies

| Frequency | Available |
|-----------|-----------|
| `daily` | ✅ |
| `1h` | ✅ |
| `1m` | ✅ |

---

## API Key Setup

```bash
# .env file
FINNHUB_API_KEY=your_api_key_here
```

Get your API key at [finnhub.io/register](https://finnhub.io/register).

---

## Not Yet Implemented

| Feature | Priority |
|---------|----------|
| Company metrics | HIGH |
| Analyst estimates | MEDIUM |
| Earnings calendar | MEDIUM |
| ESG scores | LOW |
| News sentiment | LOW |

---

## See Also

- [Finnhub Pricing](https://finnhub.io/pricing)
- [Provider README](README.md)
