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
| Free | $0/month | 60 requests/minute, US quotes and selected company data |
| All-in-One | See current pricing | Historical OHLC and additional datasets |

---

## Quick Start

```python
import os
os.environ["FINNHUB_API_KEY"] = "your_key_here"

from ml4t.data.providers import FinnhubProvider

with FinnhubProvider() as provider:
    quote = provider.fetch_quote("AAPL")
```

`fetch_quote()` uses Finnhub's free US quote endpoint. Historical candles use a premium endpoint:

```python
with FinnhubProvider() as provider:
    history = provider.fetch_ohlcv(
        "AAPL",
        "2024-01-01",
        "2024-12-01",
        frequency="daily",
    )
```

Do not assume that a free key can call `fetch_ohlcv()`. Finnhub's current pricing table lists US
OHLC history under its paid plan.

---

## API Key Setup

```bash
# Environment variable
export FINNHUB_API_KEY=your_api_key_here
```

Get your API key at [finnhub.io/register](https://finnhub.io/register).

---

## Implemented Capabilities

| Method | Account requirement |
|--------|---------------------|
| `fetch_quote()` | Free key |
| `fetch_company_metrics()` | Depends on requested metric and account coverage |
| `fetch_financials()` | Depends on requested company and account coverage |
| `fetch_ohlcv()` | Paid OHLC access |

---

## See Also

- [Finnhub Pricing](https://finnhub.io/pricing)
- [Provider reference](index.md)
