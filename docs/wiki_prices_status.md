# Wiki Prices Dataset Status Report

**Report Date**: 2025-11-24
**Dataset**: Quandl WIKI Prices (Community-curated US equities EOD prices)
**Provider**: NASDAQ Data Link (formerly Quandl)

## Executive Summary

The **Wiki Prices dataset is DEPRECATED and no longer maintained** as of April 11, 2018. While the historical data remains accessible through multiple channels (NASDAQ Data Link API, Kaggle mirror, local archives), no new data is being added. The dataset provides high-quality historical coverage from 1962-2018 for 3,199 US stocks, making it valuable for historical research but unsuitable for live trading or current market analysis.

**Recommendation**: Use Wiki Prices for historical backtesting and research (pre-2018), but implement fallback to Yahoo Finance, EODHD, or other active providers for recent data and ongoing workflows.

---

## Current Availability

### 1. NASDAQ Data Link (Official Source)

**Status**: 🟡 Available (historical only)
**URL**: https://data.nasdaq.com/data/WIKI-wiki-eod-stock-prices
**Access Method**: Free account required
**API Endpoint**: `https://data.nasdaq.com/api/v3/datatables/WIKI/PRICES`

**Limitations**:
- Data frozen at March 27, 2018
- No updates since April 2018
- Free tier includes API access
- NASDAQ recommends against using for investment decisions

### 2. Kaggle Mirror

**Status**: 🟢 Available (complete archive)
**URL**: https://www.kaggle.com/datasets/marketneutral/quandl-wiki-prices-us-equites
**File Size**: 463.2 MB (compressed)
**Last Updated**: February 2, 2022 (mirror created)
**Access Method**: Kaggle account required

**Advantages**:
- Single Parquet download
- Complete historical dataset
- No API rate limits
- Snapshot preserved

### 3. Local ML4T Copy

**Status**: 🟢 Available (production-ready)
**Location**: `/home/stefan/ml4t/software/projects/daily_us_equities/wiki_prices.parquet`
**File Size**: 631.7 MB
**Format**: Polars-compatible Parquet

**Dataset Details**:
```
Total rows:      15,389,314
Date range:      1962-01-02 to 2018-03-27 (56 years)
Unique symbols:  3,199 US companies
Columns:         14 (OHLCV + dividends + splits + adjusted prices)
```

**Schema**:
| Column        | Type      | Description                          |
|---------------|-----------|--------------------------------------|
| ticker        | String    | Stock symbol                         |
| date          | Datetime  | Trading date                         |
| open          | Float64   | Opening price (unadjusted)           |
| high          | Float64   | High price (unadjusted)              |
| low           | Float64   | Low price (unadjusted)               |
| close         | Float64   | Closing price (unadjusted)           |
| volume        | Float64   | Trading volume (unadjusted)          |
| ex-dividend   | Float64   | Dividend amount                      |
| split_ratio   | Float64   | Stock split ratio                    |
| adj_open      | Float64   | Adjusted opening price               |
| adj_high      | Float64   | Adjusted high price                  |
| adj_low       | Float64   | Adjusted low price                   |
| adj_close     | Float64   | Adjusted closing price               |
| adj_volume    | Float64   | Adjusted volume                      |

---

## Why Wiki Prices Ended

### Official Explanation

From NASDAQ Data Link Help Center ([source](https://help.data.nasdaq.com/article/506-why-does-wiki-prices-only-go-up-to-march-2018)):

> "One of the main sources of that data is no longer available, and our community volunteers have been unable to find a suitable alternative source."

### Root Causes

1. **Data Source Failure**: Primary data provider became inaccessible in early 2018
2. **Community Maintenance Model**: Volunteer-based curation couldn't scale to replace lost source
3. **Data Quality Degradation**: Missing data and inaccuracies increased without primary source
4. **Quandl Acquisition**: NASDAQ acquired Quandl in December 2018, shifted focus to premium products

### Timeline

- **Pre-2018**: Community-curated free dataset with broad cross-sectional coverage
- **April 11, 2018**: NASDAQ/Quandl announced end of updates
- **March 27, 2018**: Last available data point
- **December 2018**: NASDAQ acquired Quandl, Inc.
- **2021**: Quandl rebranded as NASDAQ Data Link
- **2022-Present**: Dataset remains available but deprecated

---

## Future Viability

### For Historical Research (Pre-2018)

**Viability**: ✅ **EXCELLENT**

**Use Cases**:
- Backtesting strategies from 1962-2018
- Long-term market structure research
- Academic studies requiring survivorship-bias-free data
- Benchmarking against historical performance
- Teaching quantitative finance concepts

**Advantages**:
- 56 years of high-quality data
- 3,199 stocks (comprehensive cross-section)
- Includes delisted companies (reduces survivorship bias)
- Adjusted prices (splits/dividends) included
- Free and unrestricted access

**Limitations**:
- No data after March 2018 (7 years outdated)
- Cannot use for live trading validation
- Some tickers may have gaps or quality issues near end-of-life

### For Current/Ongoing Work

**Viability**: ❌ **NOT SUITABLE**

**Reasons**:
- 7-year data gap (2018-2025)
- Cannot analyze COVID-19 market dynamics
- Missing recent IPOs (e.g., TSLA's rise, tech boom)
- Regulatory changes not reflected (SPAC boom, meme stocks)
- No way to validate against current market

**Required Alternatives**:
1. **Primary**: Yahoo Finance (free, unlimited, 2000-present)
2. **Educational**: EODHD (500 calls/day free, €19.99/mo paid)
3. **Professional**: DataBento, Polygon, Finnhub (paid tiers)

---

## Recommended Integration Strategy

### Multi-Tier Approach (ML4T Book)

**Tier 1: Historical Foundation (Pre-2018)**
- **Provider**: Wiki Prices local copy
- **Use Cases**: Chapters 1-5 (foundations, early exercises)
- **Advantages**: Fast, offline, free, comprehensive
- **Date Range**: 1962-01-02 to 2018-03-27

**Tier 2: Recent History (2018-2024)**
- **Provider**: Yahoo Finance (primary) + EODHD (fallback)
- **Use Cases**: Chapters 6+ (strategies, ML models, validation)
- **Advantages**: Free or low-cost, current data
- **Date Range**: 2018-01-01 to present

**Tier 3: Production Quality (Optional)**
- **Provider**: DataBento or Polygon
- **Use Cases**: Advanced readers, production deployment
- **Advantages**: Professional grade, institutional quality
- **Cost**: $9-99+/month

### Fallback Logic

```python
def fetch_historical_data(symbol: str, start: str, end: str):
    """
    Fetch with intelligent fallback based on date range.

    Strategy:
    1. If end_date <= 2018-03-27: Use Wiki Prices (local)
    2. If start_date >= 2018-03-28: Use Yahoo/EODHD
    3. If range spans 2018: Combine Wiki (pre) + Yahoo (post)
    """
    from datetime import datetime

    wiki_cutoff = datetime(2018, 3, 27)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    if end_dt <= wiki_cutoff:
        # Pure historical - use Wiki Prices
        return load_from_wiki_prices(symbol, start, end)

    elif start_dt > wiki_cutoff:
        # Pure recent - use Yahoo/EODHD
        try:
            return yahoo_provider.fetch_ohlcv(symbol, start, end)
        except Exception:
            return eodhd_provider.fetch_ohlcv(symbol, start, end)

    else:
        # Spans cutoff - combine sources
        wiki_data = load_from_wiki_prices(symbol, start, "2018-03-27")
        yahoo_data = yahoo_provider.fetch_ohlcv(symbol, "2018-03-28", end)
        return pl.concat([wiki_data, yahoo_data]).sort("date")
```

### Implementation Roadmap

**Phase 1: Basic Fallback** (Current Task)
- Document Wiki Prices status (✅ This document)
- Create ingestion utility for local Parquet
- Add fallback logic to book config

**Phase 2: Notebook Updates** (task-5.4)
- Update Notebook 1 (free_data_quickstart.py) with fallback demo
- Show date-based provider selection
- Demonstrate seamless historical + recent data loading

**Phase 3: Production Patterns** (Optional)
- Cache hybrid datasets (Wiki pre-2018 + Yahoo post-2018)
- Automate daily updates for recent data
- Health checks for data gaps at 2018 boundary

---

## NASDAQ Recommended Alternatives

NASDAQ Data Link recommends two **premium** replacements:

### 1. End of Day US Stock Prices (EOD)

**Coverage**: Active stocks from major US exchanges
**Date Range**: 1996-present
**Cost**: Premium subscription required
**Advantages**: Current data, maintained by NASDAQ

### 2. Sharadar Equity Prices (SEP)

**Coverage**: Active AND delisted stocks
**Date Range**: 1998-present
**Cost**: Premium subscription ($59-199/month)
**Advantages**: Survivorship-bias free, comprehensive fundamentals

**ML4T Assessment**: Both are excellent but **unnecessarily expensive** for educational use. Yahoo Finance + EODHD provide 95% of the value at 5% of the cost.

---

## Data Quality Assessment

### Strengths

✅ **Institutional-Grade Curation**: Community-maintained with professional standards
✅ **Survivorship-Bias Free**: Includes delisted companies
✅ **Adjusted Prices**: Proper handling of splits and dividends
✅ **Long History**: 56 years (1962-2018) valuable for long-term studies
✅ **Comprehensive Coverage**: 3,199 stocks across major exchanges
✅ **Open Data**: Public domain, no usage restrictions

### Weaknesses

⚠️ **Frozen Dataset**: No updates since March 2018
⚠️ **End-of-Life Quality**: Data quality degraded in final months
⚠️ **Volunteer Gaps**: Some stocks may have missing days or errors
⚠️ **No Intraday**: Daily OHLCV only (no minute/tick data)
⚠️ **US-Only**: No international equities

### Comparison to Modern Alternatives

| Feature              | Wiki Prices | Yahoo Finance | EODHD      | DataBento    |
|----------------------|-------------|---------------|------------|--------------|
| **Date Range**       | 1962-2018   | 2000-present  | 1980-pres  | 2000-present |
| **Update Frequency** | NONE        | Daily         | Daily      | Real-time    |
| **Cost**             | Free        | Free          | €20/mo     | $9+/mo       |
| **Survivorship**     | No bias     | Some bias     | Some bias  | No bias      |
| **Quality**          | High        | Good          | High       | Professional |
| **Coverage**         | 3,199       | 7,000+        | 150,000+   | 50,000+      |
| **Exchanges**        | US only     | US only       | Global     | Global       |

---

## Technical Integration

### Loading Wiki Prices (Local Parquet)

```python
import polars as pl
from pathlib import Path

# Load from local archive
wiki_file = Path("/home/stefan/ml4t/software/projects/daily_us_equities/wiki_prices.parquet")
df = pl.read_parquet(wiki_file)

# Filter by symbol and date
aapl = df.filter(
    (pl.col("ticker") == "AAPL") &
    (pl.col("date") >= "2010-01-01") &
    (pl.col("date") <= "2018-03-27")
).sort("date")

# Use adjusted close for returns
returns = aapl.select([
    "date",
    "adj_close",
    pl.col("adj_close").pct_change().alias("return")
])
```

### Creating Provider Wrapper

```python
from ml4t.data.providers.base import BaseProvider
import polars as pl
from pathlib import Path

class WikiPricesProvider(BaseProvider):
    """
    Provider for local Wiki Prices Parquet file.

    Date Range: 1962-01-02 to 2018-03-27
    Symbols: 3,199 US companies
    """

    def __init__(self, parquet_path: str = None):
        super().__init__(rate_limit=None)  # Local file, no rate limit
        self.parquet_path = parquet_path or Path(
            "/home/stefan/ml4t/software/projects/daily_us_equities/wiki_prices.parquet"
        )
        # Load once and cache in memory (632MB is acceptable)
        self._data = pl.read_parquet(self.parquet_path)

    @property
    def name(self) -> str:
        return "wiki_prices"

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """
        Extract data for symbol from cached Parquet.

        Note: Frequency parameter ignored - Wiki Prices is daily only.
        """
        if frequency != "daily":
            raise ValueError(f"Wiki Prices only supports daily frequency, got {frequency}")

        # Filter cached data
        filtered = self._data.filter(
            (pl.col("ticker") == symbol) &
            (pl.col("date") >= start) &
            (pl.col("date") <= end)
        ).sort("date")

        if filtered.is_empty():
            raise ValueError(f"No data found for {symbol} in Wiki Prices (1962-2018)")

        # Transform to standard OHLCV schema
        return filtered.select([
            pl.col("date").alias("timestamp"),
            pl.col("adj_open").alias("open"),
            pl.col("adj_high").alias("high"),
            pl.col("adj_low").alias("low"),
            pl.col("adj_close").alias("close"),
            pl.col("adj_volume").alias("volume"),
        ])
```

### CLI Integration

```bash
# Add to ml4t-data CLI
ml4t-data fetch AAPL --provider wiki_prices --start 2010-01-01 --end 2018-03-27

# Automatic fallback in config
ml4t-data fetch AAPL --start 2010-01-01 --end 2024-11-24 --fallback-strategy auto
# Uses: Wiki Prices (2010-2018) + Yahoo (2018-2024)
```

---

## Conclusions

### Key Findings

1. **Wiki Prices is alive but frozen** - Accessible through multiple channels but no updates since March 2018
2. **Local copy is production-ready** - 631MB Parquet with 15.4M rows covering 3,199 stocks (1962-2018)
3. **Historical research value remains high** - Excellent for backtesting and academic work pre-2018
4. **Not suitable for current work** - 7-year gap requires active provider for post-2018 data
5. **Fallback strategy is essential** - Combine Wiki (historical) + Yahoo/EODHD (recent) for complete coverage

### Recommended Next Steps

**Immediate** (Phase 5):
- ✅ Status documented (this document)
- ⏭️ Create WikiPricesProvider (task-5.2)
- ⏭️ Implement fallback logic in book config (task-5.3)
- ⏭️ Update Notebook 1 with fallback demo (task-5.4)

**Future** (Optional):
- Create hybrid cache (Wiki pre-2018 + Yahoo post-2018)
- Add boundary validation at 2018-03-27
- Document survivorship bias implications
- Benchmark Wiki vs Yahoo data quality overlap (2010-2018)

---

## References

- [NASDAQ Data Link Wiki Prices Documentation](https://data.nasdaq.com/data/WIKI-wiki-eod-stock-prices/documentation)
- [Why Wiki Prices Ended in March 2018 - NASDAQ Help Center](https://help.data.nasdaq.com/article/506-why-does-wiki-prices-only-go-up-to-march-2018)
- [Kaggle: Quandl WIKI Prices US Equities](https://www.kaggle.com/datasets/marketneutral/quandl-wiki-prices-us-equites)
- [NASDAQ Data Link Blog](https://blog.quantinsti.com/nasdaq-data-link/)
- [QuantConnect: Quandl is now NasdaqDataLink](https://www.quantconnect.com/forum/discussion/12963/quandl-is-now-nasdaqdatalink/)

---

**Document Status**: ✅ Complete
**Next Task**: task-5.2 - Implement Wiki Prices fallback provider
**Phase**: 5 (Wiki Fallback) - 1/4 tasks complete
