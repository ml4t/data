# FRED Provider

**Provider**: `FREDProvider`
**Website**: [fred.stlouisfed.org](https://fred.stlouisfed.org)
**API Key**: Required for the API; optional for public scalar CSV access
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

with FREDProvider() as provider:
    unemployment = provider.fetch_series("UNRATE", "2020-01-01", "2024-12-01")
    vintage = provider.fetch_series(
        "UNRATE", "2020-01-01", "2024-12-01", vintage_date="2025-01-15"
    )
```

---

## Scalar observations and public access

`fetch_series` returns source date labels, `series_id`, `value`, units, frequency, source route
and realtime/vintage declarations. Reported missing values remain null. The default requests the
native cadence; it does not ask FRED to upsample a monthly series to daily observations.
Dates are inclusive. An annual percent yield is not a daily return.

```python
with FREDProvider(anonymous=True) as provider:
    yields = provider.fetch_series("DGS3MO", "2024-01-02", "2024-01-04")
```

Public CSV access has no API key requirement. It does not support vintage, aggregation or API
metadata requests. The six documented Treasury maturities have known Percent units and daily
cadence; generic CSV series retain unknown units/frequency rather than infer them from values.
`units_source` distinguishes series definitions from API metadata. Both routes leave
`available_at` unknown: an observation date or realtime date does not establish an intraday
publication timestamp. A configured API authentication failure is not retried anonymously.

Existing `fetch_ohlcv` remains available in API mode. Its replicated prices and placeholder volume
are economic-series packaging, not tradable OHLCV or actual volume; its null-price dropping differs
from the scalar route. See [acquisition migration](../getting-started/migration.md#empyrical-acquisition-utilities)
for the opt-in pandas compatibility helpers.

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
# Environment variable
export FRED_API_KEY=your_api_key_here
```

Get your free API key at [fred.stlouisfed.org/docs/api/fred](https://fred.stlouisfed.org/docs/api/fred/).

---

## Rate Limits

- 120 requests/minute
- No daily limit

---

## See Also

- [Macroeconomic](macro.md) and [fixed-income](fixed_income.md) source references
- [FRED API Docs](https://fred.stlouisfed.org/docs/api/fred/)
- [Example Config](https://github.com/ml4t/data/blob/main/examples/configs/fred_economic.yaml)
- [Provider reference](index.md)
