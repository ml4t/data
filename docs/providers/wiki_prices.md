# Wiki Prices Provider

**Provider**: `WikiPricesProvider`
**Data Source**: Local Parquet file
**API Key**: Not required
**Free Tier**: Free (local file)

---

## Overview

Wiki Prices provides historical US equity data from 1962-2018, including delisted companies for survivorship-bias-free research.

**Best For**: Long-term backtesting, survivorship-bias-free research

---

## Quick Start

```python
from ml4t.data.providers import WikiPricesProvider

# Point to local Parquet file
provider = WikiPricesProvider(parquet_path="~/data/wiki_prices.parquet")

# Fetch historical data
df = provider.fetch_ohlcv("AAPL", "1990-01-01", "2018-03-27")

provider.close()
```

---

## Dataset Details

| Attribute | Value |
|-----------|-------|
| Coverage | 3,199 US stocks |
| Rows | 15.4 million |
| Period | 1962-2018 |
| Size | ~631 MB (Parquet) |
| Includes Delisted | Yes |

---

## Fallback Pattern

Combine with Yahoo Finance for complete history:

```python
from ml4t.data.providers import WikiPricesProvider, YahooFinanceProvider
import polars as pl

# Pre-2018: Wiki Prices
wiki = WikiPricesProvider(parquet_path="~/data/wiki_prices.parquet")
historical = wiki.fetch_ohlcv("AAPL", "1990-01-01", "2018-03-27")

# Post-2018: Yahoo Finance
yahoo = YahooFinanceProvider()
recent = yahoo.fetch_ohlcv("AAPL", "2018-03-28", "2024-12-01")

# Combine
combined = pl.concat([historical, recent]).sort("timestamp")
print(f"Total: {len(combined)} rows from 1990-2024")
```

---

## Data Source

The Wiki Prices dataset was originally from Quandl's WIKI database before Quandl discontinued the free tier.

---

## See Also

- [Provider README](README.md)
- [Example Config](../../configs/examples/sp500.yaml)
