# Multi-Asset Support - Complete User Guide

**Version**: 1.0.0
**Last Updated**: 2025-11-15
**Status**: Production-Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Usage Patterns](#usage-patterns)
6. [Performance](#performance)
7. [Migration Guide](#migration-guide)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is Multi-Asset Support?

Multi-asset support enables efficient loading, processing, and analysis of data for multiple securities simultaneously. Instead of fetching data for one symbol at a time, you can:

- **Load 100+ symbols in seconds** with parallel fetching
- **Analyze cross-sectional patterns** across entire universes
- **Leverage cache-first loading** for 10-100x speedup
- **Use pre-defined universes** (S&P 500, NASDAQ-100, etc.)

### When to Use Multi-Asset Features

**Use multi-asset support when you need to**:
- Backtest portfolio strategies across many symbols
- Perform sector rotation or pair trading analysis
- Calculate cross-sectional metrics (rankings, quintiles)
- Build market-wide indicators or sentiment measures
- Compare performance across asset groups

**Stick with single-asset if**:
- You're analyzing just one security
- You're building single-instrument strategies
- You're doing deep dive analysis on individual stocks

### Key Benefits

1. **Performance**: 10-100x faster than sequential loading
2. **Convenience**: Pre-defined universes (S&P 500, crypto top-100, etc.)
3. **Consistency**: Standardized stacked format across all operations
4. **Scalability**: Handles 500+ symbols efficiently
5. **Caching**: Load from storage in <1 second instead of minutes

---

## Core Concepts

### Stacked (Long) Format

Multi-asset data uses a **stacked format** with a `symbol` column to identify each security:

```python
┌─────────────────────┬────────┬───────┬───────┬───────┬───────┬──────────┐
│ timestamp           │ symbol │ open  │ high  │ low   │ close │ volume   │
├─────────────────────┼────────┼───────┼───────┼───────┼───────┼──────────┤
│ 2024-01-01 09:30:00 │ AAPL   │ 180.0 │ 182.0 │ 179.0 │ 181.0 │ 1200000  │
│ 2024-01-01 09:30:00 │ MSFT   │ 370.0 │ 372.0 │ 368.0 │ 371.0 │ 800000   │
│ 2024-01-01 09:30:00 │ GOOG   │ 140.0 │ 142.0 │ 139.0 │ 141.0 │ 950000   │
│ 2024-01-02 09:30:00 │ AAPL   │ 181.0 │ 183.0 │ 180.0 │ 182.0 │ 1100000  │
│ 2024-01-02 09:30:00 │ MSFT   │ 371.0 │ 373.0 │ 370.0 │ 372.0 │ 750000   │
│ 2024-01-02 09:30:00 │ GOOG   │ 141.0 │ 143.0 │ 140.0 │ 142.0 │ 900000   │
└─────────────────────┴────────┴───────┴───────┴───────┴───────┴──────────┘
```

**Why stacked format?**
- **Polars-native**: No MultiIndex needed (unlike pandas)
- **Efficient grouping**: Natural `group_by('symbol')` operations
- **Proven architecture**: Used in qengine backtest library
- **Scalable**: Works with 500+ symbols without performance degradation

### MultiAssetSchema

The `MultiAssetSchema` class validates and standardizes multi-asset DataFrames:

```python
from ml4t.data.core.schemas import MultiAssetSchema

# Required columns
# - timestamp (datetime with timezone)
# - symbol (string)
# - open, high, low, close, volume (float)

# Validate a DataFrame
is_valid = MultiAssetSchema.validate(df, strict=True)

# Standardize column order and sorting
df = MultiAssetSchema.standardize_order(df)

# Create empty DataFrame with proper schema
df = MultiAssetSchema.create_empty('equities')
```

### Format Conversion

While stacked format is canonical, you can convert to/from wide format when needed:

**Stacked → Wide** (for correlation analysis, pandas compatibility):
```python
from ml4t.data.utils.format import pivot_to_wide

df_wide = pivot_to_wide(df_stacked, value_cols=['close', 'volume'])

# Result:
┌────────────┬────────────┬────────────┬────────────┬─────────────┬─────────────┐
│ timestamp  │ close_AAPL │ close_MSFT │ close_GOOG │ volume_AAPL │ volume_MSFT │
├────────────┼────────────┼────────────┼────────────┼─────────────┼─────────────┤
│ 2024-01-01 │ 181.0      │ 371.0      │ 141.0      │ 1200000     │ 800000      │
│ 2024-01-02 │ 182.0      │ 372.0      │ 142.0      │ 1100000     │ 750000      │
└────────────┴────────────┴────────────┴────────────┴─────────────┴─────────────┘
```

**Wide → Stacked** (reverse conversion):
```python
from ml4t.data.utils.format import pivot_to_stacked

df_stacked = pivot_to_stacked(df_wide)  # Back to stacked format
```

**⚠️ Performance Warning**: Wide format doesn't scale beyond ~100 symbols. Use stacked format whenever possible.

---

## Quick Start

### 1. Basic Batch Loading

Load multiple symbols in one call:

```python
from ml4t.data import DataManager

manager = DataManager()

# Load 5 tech stocks
df = manager.batch_load(
    symbols=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
    start='2024-01-01',
    end='2024-12-31',
    provider='yahoo'
)

print(f"Loaded {len(df)} rows for {df['symbol'].n_unique()} symbols")
# Output: Loaded 1,260 rows for 5 symbols
```

### 2. Universe Loading

Use pre-defined symbol lists:

```python
from ml4t.data.universe import Universe

# S&P 500
df = manager.batch_load_universe(
    universe='SP500',
    start='2024-01-01',
    end='2024-12-31'
)

# Or get the symbol list directly
symbols = Universe.SP500
print(f"S&P 500 contains {len(symbols)} symbols")
```

**Available universes**:
- `Universe.SP500` - S&P 500 index (500+ symbols)
- `Universe.NASDAQ100` - NASDAQ-100 index (100 symbols)
- `Universe.CRYPTO_TOP_100` - Top 100 cryptocurrencies by market cap
- `Universe.FOREX_MAJORS` - Major forex pairs (EUR/USD, GBP/USD, etc.)

### 3. Cache-First Loading

For maximum performance, load from storage instead of network:

```python
from ml4t.data.storage import HiveStorage, StorageConfig

# Setup storage
storage = HiveStorage(StorageConfig(base_path="./data"))
manager = DataManager(storage=storage)

# First time: Load from network and cache
df = manager.batch_load(
    symbols=['AAPL', 'MSFT', 'GOOG'],
    start='2024-01-01',
    end='2024-12-31',
    provider='yahoo'
)
# Takes ~5-10 seconds

# Subsequent times: Load from cache (very fast!)
df = manager.batch_load_from_storage(
    symbols=['AAPL', 'MSFT', 'GOOG'],
    start='2024-01-01',
    end='2024-12-31'
)
# Takes ~0.01 seconds (100x faster!)
```

### 4. Cross-Sectional Analysis

Analyze patterns across symbols:

```python
import polars as pl

# Calculate returns by symbol
df = df.with_columns(
    pl.col('close').pct_change().over('symbol').alias('returns')
)

# Daily cross-sectional statistics
daily_stats = df.group_by('timestamp').agg([
    pl.col('returns').mean().alias('mean_return'),
    pl.col('returns').std().alias('std_return'),
    pl.col('volume').sum().alias('total_volume'),
    pl.col('symbol').count().alias('n_symbols')
])

print(daily_stats.head())
```

---

## API Reference

### batch_load()

Load multiple symbols in parallel from network providers.

```python
df = manager.batch_load(
    symbols: list[str],
    start: str,
    end: str,
    frequency: str = "daily",
    provider: str | None = None,
    max_workers: int = 4,
    fail_on_partial: bool = False,
    **kwargs
) -> pl.DataFrame
```

**Parameters**:
- `symbols`: List of symbol identifiers (e.g., `['AAPL', 'MSFT', 'GOOG']`)
- `start`: Start date (ISO format: `'2024-01-01'`)
- `end`: End date (ISO format: `'2024-12-31'`)
- `frequency`: Bar frequency (`'daily'`, `'1h'`, `'5min'`, etc.)
- `provider`: Data provider (`'yahoo'`, `'databento'`, `'binance'`, etc.)
- `max_workers`: Number of parallel workers (default: 4)
- `fail_on_partial`: Raise error if any symbol fails (default: False)

**Returns**: Polars DataFrame in stacked format with `symbol` column

**Example**:
```python
df = manager.batch_load(
    symbols=['AAPL', 'MSFT', 'GOOG'],
    start='2024-01-01',
    end='2024-01-31',
    provider='yahoo',
    max_workers=8  # Faster parallel fetching
)
```

---

### batch_load_universe()

Load a pre-defined universe of symbols.

```python
df = manager.batch_load_universe(
    universe: str | list[str],
    start: str,
    end: str,
    frequency: str = "daily",
    provider: str | None = None,
    max_workers: int = 4,
    **kwargs
) -> pl.DataFrame
```

**Parameters**:
- `universe`: Universe name (`'SP500'`, `'NASDAQ100'`, etc.) or custom list
- Other parameters same as `batch_load()`

**Returns**: Polars DataFrame in stacked format

**Example**:
```python
# Pre-defined universe
df = manager.batch_load_universe('SP500', '2024-01-01', '2024-12-31')

# Custom universe
df = manager.batch_load_universe(
    universe=['SPY', 'QQQ', 'IWM'],  # ETF trio
    start='2024-01-01',
    end='2024-12-31'
)
```

---

### batch_load_from_storage()

Load multiple symbols from local storage (cache-first).

```python
df = manager.batch_load_from_storage(
    symbols: list[str],
    start: str,
    end: str,
    frequency: str = "daily",
    asset_class: str = "equities",
    fetch_missing: bool = False,
    provider: str | None = None,
    max_workers: int = 4,
    **kwargs
) -> pl.DataFrame
```

**Parameters**:
- `symbols`: List of symbol identifiers
- `start`, `end`: Date range
- `frequency`: Bar frequency
- `asset_class`: Asset class for storage lookup (`'equities'`, `'crypto'`, etc.)
- `fetch_missing`: If True, fetch missing symbols from provider (default: False)
- `provider`: Provider to use if fetching missing symbols
- `max_workers`: Number of parallel workers for storage reads

**Returns**: Polars DataFrame in stacked format

**Performance**: **10-100x faster** than network fetching when data is cached

**Example**:
```python
# Fast load from cache (errors if any symbol missing)
df = manager.batch_load_from_storage(
    symbols=['AAPL', 'MSFT', 'GOOG'],
    start='2024-01-01',
    end='2024-12-31',
    fetch_missing=False
)

# Load from cache with fallback to network
df = manager.batch_load_from_storage(
    symbols=['AAPL', 'MSFT', 'NEWSTOCK'],  # NEWSTOCK not in cache
    start='2024-01-01',
    end='2024-12-31',
    fetch_missing=True,
    provider='yahoo'
)
```

---

### pivot_to_wide()

Convert stacked format to wide (pivoted) format.

```python
from ml4t.data.utils.format import pivot_to_wide

df_wide = pivot_to_wide(
    df: pl.DataFrame,
    value_cols: list[str] | None = None
) -> pl.DataFrame
```

**Parameters**:
- `df`: Stacked DataFrame with `timestamp` and `symbol` columns
- `value_cols`: Columns to pivot (default: all OHLCV columns)

**Returns**: Wide DataFrame with columns like `close_AAPL`, `close_MSFT`, etc.

**Example**:
```python
# Pivot only close prices
df_wide = pivot_to_wide(df, value_cols=['close'])

# Use for pandas correlation analysis
df_pandas = df_wide.to_pandas().set_index('timestamp')
correlation_matrix = df_pandas.corr()
```

---

### pivot_to_stacked()

Convert wide format back to stacked format.

```python
from ml4t.data.utils.format import pivot_to_stacked

df_stacked = pivot_to_stacked(
    df: pl.DataFrame
) -> pl.DataFrame
```

**Parameters**:
- `df`: Wide DataFrame with symbol-suffixed columns

**Returns**: Stacked DataFrame with `symbol` column

**Example**:
```python
# Round-trip conversion
df_wide = pivot_to_wide(df_stacked)
df_back = pivot_to_stacked(df_wide)

# Data integrity preserved
assert df_stacked.sort(['timestamp', 'symbol']).equals(
    df_back.sort(['timestamp', 'symbol'])
)
```

---

## Usage Patterns

### Pattern 1: Portfolio Backtesting

Load all portfolio constituents and run backtest:

```python
# Define portfolio
portfolio_symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']

# Load data
df = manager.batch_load_from_storage(
    symbols=portfolio_symbols,
    start='2020-01-01',
    end='2024-12-31',
    fetch_missing=True,
    provider='yahoo'
)

# Calculate returns
df = df.with_columns(
    pl.col('close').pct_change().over('symbol').alias('returns')
)

# Equal-weight portfolio return
portfolio_returns = df.group_by('timestamp').agg(
    pl.col('returns').mean().alias('portfolio_return')
)
```

### Pattern 2: Sector Rotation

Analyze sector performance:

```python
# Define sectors
tech = ['AAPL', 'MSFT', 'GOOG']
finance = ['JPM', 'BAC', 'GS']
healthcare = ['UNH', 'JNJ', 'PFE']

all_symbols = tech + finance + healthcare

# Load data
df = manager.batch_load_universe(all_symbols, '2024-01-01', '2024-12-31')

# Add sector labels
df = df.with_columns(
    pl.when(pl.col('symbol').is_in(tech))
    .then(pl.lit('Tech'))
    .when(pl.col('symbol').is_in(finance))
    .then(pl.lit('Finance'))
    .otherwise(pl.lit('Healthcare'))
    .alias('sector')
)

# Sector performance
sector_perf = df.group_by(['timestamp', 'sector']).agg([
    pl.col('close').mean().alias('avg_close'),
    pl.col('volume').sum().alias('total_volume')
])
```

### Pattern 3: Universe Screening

Screen entire universe for trading signals:

```python
# Load S&P 500
df = manager.batch_load_from_storage(
    symbols=Universe.SP500,
    start='2024-01-01',
    end='2024-12-31'
)

# Calculate 20-day momentum
df = df.with_columns(
    (pl.col('close') / pl.col('close').shift(20) - 1)
    .over('symbol')
    .alias('momentum_20d')
)

# Screen: momentum > 10%
winners = df.filter(pl.col('momentum_20d') > 0.10)

print(f"Found {winners['symbol'].n_unique()} symbols with >10% momentum")
```

### Pattern 4: Cross-Sectional Ranking

Rank symbols by various metrics:

```python
import polars as pl

# Load data
df = manager.batch_load_from_storage(
    symbols=Universe.NASDAQ100,
    start='2024-01-01',
    end='2024-12-31'
)

# Calculate returns and volatility
df = df.with_columns([
    pl.col('close').pct_change().over('symbol').alias('returns'),
])

# Daily cross-sectional ranks
df = df.with_columns([
    pl.col('returns').rank('ordinal').over('timestamp').alias('return_rank'),
    pl.col('volume').rank('ordinal').over('timestamp').alias('volume_rank'),
])

# Top decile by returns
top_decile = df.filter(pl.col('return_rank') > pl.col('return_rank').max() * 0.9)
```

### Pattern 5: Pair Trading

Find and analyze correlated pairs:

```python
# Load potential pairs
df = manager.batch_load(
    symbols=['XLE', 'XLF', 'XLK', 'XLV', 'XLI'],  # Sector ETFs
    start='2024-01-01',
    end='2024-12-31'
)

# Convert to wide for correlation
df_wide = pivot_to_wide(df, value_cols=['close'])

# Calculate correlation matrix
df_pandas = df_wide.to_pandas().set_index('timestamp')
corr_matrix = df_pandas.corr()

# Find highly correlated pairs (>0.8)
high_corr_pairs = corr_matrix[(corr_matrix > 0.8) & (corr_matrix < 1.0)]
print(high_corr_pairs)
```

---

## Performance

### Benchmark Results

**Storage Load Performance** (100 symbols, 252 trading days each):

| Operation | Time | Rows/Second | Speedup vs Network |
|-----------|------|-------------|--------------------|
| batch_load_from_storage() | 0.165s | 152,000 | **76-303x faster** |
| batch_load() | 12-50s | 500-2,000 | Baseline |

**Format Conversion Performance** (50 symbols, 252 days):

| Operation | Input Size | Time | Throughput |
|-----------|------------|------|------------|
| Stacked → Wide | 12,600 rows | 0.009s | 1.4M rows/sec |
| Wide → Stacked | 252×50 matrix | 0.010s | 1.26M cells/sec |

### Performance Tips

#### 1. Use Storage for Repeated Analysis

```python
# DON'T: Fetch every time (slow)
for iteration in range(100):
    df = manager.batch_load(symbols, start, end)  # 5-10 seconds each!

# DO: Load once, cache, then reuse
df = manager.batch_load(symbols, start, end)
for iteration in range(100):
    df_cached = manager.batch_load_from_storage(symbols, start, end)  # 0.1 seconds!
```

#### 2. Increase Workers for Large Universes

```python
# Default (conservative)
df = manager.batch_load(Universe.SP500, start, end, max_workers=4)

# Faster (if provider allows)
df = manager.batch_load(Universe.SP500, start, end, max_workers=16)
```

#### 3. Use Narrow Date Ranges

```python
# DON'T: Load full history if you only need recent data
df = manager.batch_load_from_storage(symbols, '2000-01-01', '2024-12-31')

# DO: Load only what you need
df = manager.batch_load_from_storage(symbols, '2024-01-01', '2024-12-31')
```

#### 4. Prefer Stacked Format

```python
# DON'T: Convert to wide unless necessary
df_wide = pivot_to_wide(df)  # Extra overhead

# DO: Work in stacked format
df_stacked.group_by('symbol').agg(...)  # Native Polars operations
```

---

## Migration Guide

### From Single-Symbol to Multi-Symbol

**Before (single-symbol)**:
```python
# Load one at a time
aapl = manager.load('AAPL', '2024-01-01', '2024-12-31')
msft = manager.load('MSFT', '2024-01-01', '2024-12-31')
goog = manager.load('GOOG', '2024-01-01', '2024-12-31')

# Combine manually
import polars as pl
aapl = aapl.with_columns(pl.lit('AAPL').alias('symbol'))
msft = msft.with_columns(pl.lit('MSFT').alias('symbol'))
goog = goog.with_columns(pl.lit('GOOG').alias('symbol'))
df = pl.concat([aapl, msft, goog])
```

**After (multi-symbol)**:
```python
# Load all at once
df = manager.batch_load(
    symbols=['AAPL', 'MSFT', 'GOOG'],
    start='2024-01-01',
    end='2024-12-31'
)
# Done! Already has symbol column and proper format
```

### From Pandas to Polars Stacked Format

**Before (pandas with MultiIndex)**:
```python
import pandas as pd

# Pandas MultiIndex DataFrame
df_pandas = pd.DataFrame({
    ('AAPL', 'close'): [180, 181, 182],
    ('MSFT', 'close'): [370, 371, 372],
}, index=pd.date_range('2024-01-01', periods=3))
```

**After (Polars stacked)**:
```python
import polars as pl

df = manager.batch_load(['AAPL', 'MSFT'], '2024-01-01', '2024-01-03')

# Stacked format - much cleaner!
# - No MultiIndex complexity
# - Easier filtering and grouping
# - 10-100x faster operations
```

### From Wide to Stacked Format

**If you have wide data**:
```python
# Wide format DataFrame (e.g., from CSV)
df_wide = pl.read_csv('wide_data.csv')
# Columns: timestamp, AAPL_close, MSFT_close, GOOG_close

# Convert to stacked
from ml4t.data.utils.format import pivot_to_stacked
df_stacked = pivot_to_stacked(df_wide)

# Now compatible with all multi-asset features
df_stacked.group_by('symbol').agg(...)
```

---

## Best Practices

### 1. Always Validate Schema

```python
from ml4t.data.core.schemas import MultiAssetSchema

# After loading external data
df = pl.read_parquet('external_data.parquet')

# Validate before processing
if not MultiAssetSchema.validate(df, strict=False):
    print("Warning: Schema validation failed")
    df = MultiAssetSchema.standardize_order(df)  # Fix it
```

### 2. Use Graceful Failure Mode

```python
# Allow partial success (recommended for research)
df = manager.batch_load(
    symbols=['AAPL', 'INVALID', 'MSFT'],
    fail_on_partial=False  # Returns AAPL + MSFT
)

# Strict mode (for production)
df = manager.batch_load(
    symbols=['AAPL', 'MSFT', 'GOOG'],
    fail_on_partial=True  # Raises error if any fail
)
```

### 3. Pre-populate Storage

```python
# One-time setup: Populate storage
for symbol in Universe.SP500:
    manager.load(symbol, '2020-01-01', '2024-12-31', provider='yahoo')

# Then use fast storage loading
df = manager.batch_load_from_storage(
    symbols=Universe.SP500,
    start='2024-01-01',
    end='2024-12-31'
)
# 100x faster!
```

### 4. Leverage Polars Operations

```python
# DON'T: Convert to pandas unnecessarily
df_pandas = df.to_pandas()
df_pandas.groupby('symbol')['close'].mean()  # Slow!

# DO: Use Polars native operations
df.group_by('symbol').agg(pl.col('close').mean())  # Fast!
```

### 5. Monitor Memory Usage

```python
# For very large universes (500+ symbols), process in batches
symbols = Universe.SP500
batch_size = 50

for i in range(0, len(symbols), batch_size):
    batch_symbols = symbols[i:i+batch_size]
    df_batch = manager.batch_load_from_storage(
        symbols=batch_symbols,
        start='2024-01-01',
        end='2024-12-31'
    )
    # Process batch
    process_batch(df_batch)
```

---

## Troubleshooting

### Issue: "ValueError: Symbol column not found"

**Cause**: DataFrame missing required `symbol` column

**Solution**:
```python
# Add symbol column manually
df = df.with_columns(pl.lit('AAPL').alias('symbol'))

# Or use schema helper
from ml4t.data.core.schemas import MultiAssetSchema
df = MultiAssetSchema.add_symbol_column(df, symbol='AAPL')
```

### Issue: "Storage not configured"

**Cause**: Calling `batch_load_from_storage()` without storage setup

**Solution**:
```python
from ml4t.data.storage import HiveStorage, StorageConfig

# Configure storage
storage = HiveStorage(StorageConfig(base_path="./data"))
manager = DataManager(storage=storage)

# Now storage methods work
df = manager.batch_load_from_storage(...)
```

### Issue: "No data for date range"

**Cause**: Requested date range not in storage

**Solution**:
```python
# Option 1: Use fetch_missing=True
df = manager.batch_load_from_storage(
    symbols=['AAPL'],
    start='2024-01-01',
    end='2024-12-31',
    fetch_missing=True,  # Auto-fetch if missing
    provider='yahoo'
)

# Option 2: Pre-populate storage
manager.load('AAPL', '2024-01-01', '2024-12-31', provider='yahoo')
df = manager.batch_load_from_storage(['AAPL'], '2024-01-01', '2024-12-31')
```

### Issue: "Duplicate timestamp-symbol pairs"

**Cause**: Data contains duplicates (data quality issue)

**Solution**:
```python
# Remove duplicates
df = df.unique(subset=['timestamp', 'symbol'], keep='last')

# Or detect and investigate
duplicates = df.filter(
    pl.struct(['timestamp', 'symbol']).is_duplicated()
)
print(f"Found {len(duplicates)} duplicates")
```

### Issue: Wide format "too many columns" error

**Cause**: Trying to pivot 100+ symbols creates thousands of columns

**Solution**:
```python
# DON'T: Pivot large universes
df_wide = pivot_to_wide(df)  # Fails with 100+ symbols

# DO: Work in stacked format
df_stacked.group_by('symbol').agg(...)  # Scales to 1000+ symbols
```

### Issue: Slow batch loading

**Cause**: Network rate limits or too few workers

**Solution**:
```python
# Increase workers (if provider allows)
df = manager.batch_load(
    symbols=Universe.SP500,
    max_workers=16  # Up from default 4
)

# Or use storage for subsequent loads
df = manager.batch_load_from_storage(...)  # 100x faster
```

---

## Advanced Topics

### Custom Universes

Create your own symbol lists:

```python
from ml4t.data.universe import Universe

# Define custom universe
tech_giants = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA']
Universe.register_universe('TECH_GIANTS', tech_giants)

# Use it
df = manager.batch_load_universe('TECH_GIANTS', '2024-01-01', '2024-12-31')

# Or create dynamically
faang = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOG']
df = manager.batch_load_universe(faang, '2024-01-01', '2024-12-31')
```

### Date Range Optimization

Use lazy evaluation for efficient date filtering:

```python
# Storage reads apply filters during scan (very fast)
df = manager.batch_load_from_storage(
    symbols=['AAPL', 'MSFT'],
    start='2024-11-01',  # Only loads November partitions
    end='2024-11-30'
)

# Predicate pushdown = only read relevant Parquet files
```

### Parallel Storage Workers

Tune parallelism for your hardware:

```python
# Default: 4 workers
df = manager.batch_load_from_storage(symbols, start, end)

# More workers for SSD/NVMe (faster I/O)
df = manager.batch_load_from_storage(
    symbols, start, end,
    max_workers=8  # Faster on modern hardware
)

# Fewer workers for HDD (avoid thrashing)
df = manager.batch_load_from_storage(
    symbols, start, end,
    max_workers=2  # Better for rotational disks
)
```

---

## Examples

### Complete Workflow Example

```python
from ml4t.data import DataManager
from ml4t.data.storage import HiveStorage, StorageConfig
from ml4t.data.universe import Universe
from ml4t.data.utils.format import pivot_to_wide
import polars as pl

# Setup
storage = HiveStorage(StorageConfig(base_path="./data"))
manager = DataManager(storage=storage)

# 1. Initial load (populate storage)
print("Loading S&P 500 data...")
df = manager.batch_load_universe(
    universe='SP500',
    start='2024-01-01',
    end='2024-12-31',
    provider='yahoo',
    max_workers=8
)
print(f"Loaded {len(df):,} rows for {df['symbol'].n_unique()} symbols")

# 2. Subsequent analysis (fast cache loading)
print("\nReloading from cache...")
df = manager.batch_load_from_storage(
    symbols=Universe.SP500,
    start='2024-01-01',
    end='2024-12-31'
)
print(f"Loaded in <1 second from cache")

# 3. Calculate returns
df = df.with_columns(
    pl.col('close').pct_change().over('symbol').alias('returns')
)

# 4. Cross-sectional analysis
daily_stats = df.group_by('timestamp').agg([
    pl.col('returns').mean().alias('mean_return'),
    pl.col('returns').std().alias('std_return'),
    pl.col('returns').quantile(0.9).alias('p90_return'),
    pl.col('volume').sum().alias('total_volume'),
])

print("\nDaily statistics:")
print(daily_stats.head())

# 5. Correlation analysis (use wide format)
df_wide = pivot_to_wide(df, value_cols=['returns'])
df_pandas = df_wide.to_pandas().set_index('timestamp')
correlation = df_pandas.corr()

print(f"\nCorrelation matrix: {correlation.shape}")

# 6. Screen for winners
winners = df.group_by('symbol').agg([
    (pl.col('close').last() / pl.col('close').first() - 1).alias('total_return')
]).filter(pl.col('total_return') > 0.2)

print(f"\nSymbols with >20% returns: {len(winners)}")
print(winners.sort('total_return', descending=True).head())
```

---

## Summary

**Multi-asset support in ml4t-data provides**:

✅ **Efficient loading**: 10-100x faster with storage caching
✅ **Convenient universes**: Pre-defined symbol lists (S&P 500, etc.)
✅ **Scalable format**: Stacked format handles 500+ symbols
✅ **Flexible conversion**: Wide format when needed
✅ **Production-ready**: Comprehensive validation and error handling

**Get started**:
1. Use `batch_load()` for initial data fetch
2. Use `batch_load_from_storage()` for fast subsequent access
3. Leverage pre-defined universes for common use cases
4. Stick with stacked format for best performance

**For more help**:
- Examples: `/examples/multi_asset_quickstart.py`
- Tests: `/tests/test_multi_asset.py`
- Performance: `/PERFORMANCE_BENCHMARKS.md`
