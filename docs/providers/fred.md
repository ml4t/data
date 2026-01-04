# FRED Provider

**Provider**: `FREDProvider`
**Website**: [fred.stlouisfed.org](https://fred.stlouisfed.org)
**API Key**: Required (free)
**Free Tier**: 120 requests/minute

---

## Overview

FRED (Federal Reserve Economic Data) provides 800,000+ economic time series from the St. Louis Fed.

**Best For**: Macroeconomic indicators, interest rates, economic research

---

## Quick Start

```python
import os
os.environ["FRED_API_KEY"] = "your_key_here"

from ml4t.data.providers import FREDProvider

provider = FREDProvider()

# Interest rates
df = provider.fetch_ohlcv("DGS10", "2020-01-01", "2024-12-01")  # 10-Year Treasury

# Economic indicators
df = provider.fetch_ohlcv("UNRATE", "2020-01-01", "2024-12-01")  # Unemployment
df = provider.fetch_ohlcv("CPIAUCSL", "2020-01-01", "2024-12-01")  # CPI

provider.close()
```

---

## Popular Series

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `DFF` | Fed Funds Rate | Daily |
| `DGS10` | 10-Year Treasury | Daily |
| `DGS2` | 2-Year Treasury | Daily |
| `T10Y2Y` | 10Y-2Y Spread | Daily |
| `UNRATE` | Unemployment Rate | Monthly |
| `CPIAUCSL` | CPI All Items | Monthly |
| `INDPRO` | Industrial Production | Monthly |
| `VIXCLS` | VIX Index | Daily |

See [FRED Categories](https://fred.stlouisfed.org/categories) for all 800,000+ series.

---

## API Key Setup

```bash
# .env file
FRED_API_KEY=your_api_key_here
```

Get your free API key at [fred.stlouisfed.org/docs/api/fred](https://fred.stlouisfed.org/docs/api/fred/).

---

## Rate Limits

- 120 requests/minute
- No daily limit

---

## See Also

- [FRED API Docs](https://fred.stlouisfed.org/docs/api/fred/)
- [Example Config](../../configs/examples/macro_factors.yaml)
- [Provider README](README.md)
