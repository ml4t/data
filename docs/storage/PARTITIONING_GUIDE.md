# Hive Partitioning Guide

ml4t-data uses Hive-style partitioned Parquet storage for efficient incremental updates and fast queries.

---

## Why Hive Partitioning?

| Benefit | Description |
|---------|-------------|
| **Partition pruning** | Query only relevant time periods |
| **Incremental writes** | Append new data without rewriting history |
| **Parallel processing** | Process partitions independently |
| **DuckDB/Polars integration** | Native support for partition-aware queries |

---

## Directory Structure

```
base_path/
└── {provider}/
    └── {frequency}/
        └── year={YYYY}/
            └── month={MM}/
                └── {symbol}/
                    └── data.parquet
```

### Example

```
~/ml4t-data/
├── yahoo/
│   └── daily/
│       ├── year=2024/
│       │   ├── month=01/
│       │   │   ├── AAPL/data.parquet
│       │   │   ├── MSFT/data.parquet
│       │   │   └── GOOGL/data.parquet
│       │   ├── month=02/
│       │   │   └── ...
│       │   └── month=12/
│       └── year=2023/
│           └── ...
├── eodhd/
│   └── daily/
│       └── ...
└── databento/
    ├── ohlcv-1m/
    │   └── year=2024/
    │       └── month=01/
    │           └── day=15/   # Day partitions for minute data
    │               └── ESH5/data.parquet
    └── ohlcv-1d/
        └── ...
```

---

## Auto-Heuristics

ml4t-data automatically selects partition granularity based on frequency:

| Frequency | Partition Keys | Rationale |
|-----------|----------------|-----------|
| `daily` | year/month | ~20-22 rows per month per symbol |
| `1h` | year/month | ~160-180 rows per month per symbol |
| `15m` | year/month/day | ~26 rows per day per symbol |
| `5m` | year/month/day | ~78 rows per day per symbol |
| `1m` | year/month/day | ~390 rows per day per symbol |

### Target File Size

**Optimal: 50-200MB per Parquet file**

- Too small (<10MB): Overhead from many files, metadata costs
- Too large (>500MB): Slow to read/write, poor parallelism

### Symbol Count Considerations

| Symbol Count | Recommendation |
|--------------|----------------|
| 1-50 | Year/month partitions sufficient |
| 50-500 | Year/month + symbol subdirectory |
| 500+ | Consider additional partitioning or separate configs |

---

## Configuration Override

While auto-heuristics work for most cases, override via YAML:

```yaml
storage:
  path: ~/ml4t-data
  format: hive

datasets:
  # Auto-heuristic (recommended for most cases)
  sp500_daily:
    provider: yahoo
    frequency: daily
    symbols_file: sp500.txt
    # partition_by: auto  # implicit

  # Manual override for minute data
  futures_1m:
    provider: databento
    frequency: 1m
    partition_by:
      - year
      - month
      - day
    symbols:
      - ES
      - NQ
      - CL
```

### Partition Key Options

| Key | Values | Use Case |
|-----|--------|----------|
| `year` | 2020, 2021, ... | Always include |
| `month` | 01-12 | Most common |
| `day` | 01-31 | Intraday data |
| `symbol` | AAPL, MSFT, ... | Large universes |

---

## Querying Partitioned Data

### With Polars (Recommended)

```python
import polars as pl

# Lazy scan with partition pruning
df = pl.scan_parquet(
    "~/ml4t-data/yahoo/daily/**/*/data.parquet",
    hive_partitioning=True
).filter(
    (pl.col("year") == 2024) &
    (pl.col("month") >= 10) &
    (pl.col("symbol") == "AAPL")
).collect()

# Only reads 2024 Q4 partitions for AAPL
```

### With DuckDB

```sql
SELECT
    symbol,
    date_trunc('month', timestamp) as month,
    AVG(close) as avg_close
FROM read_parquet(
    '~/ml4t-data/yahoo/daily/**/*/data.parquet',
    hive_partitioning=true
)
WHERE year = 2024 AND month >= 10
GROUP BY 1, 2
ORDER BY 1, 2
```

### With Storage API

```python
from ml4t.data.storage.hive import HiveStorage, StorageConfig

storage = HiveStorage(config=StorageConfig(base_path="~/ml4t-data"))

# Read specific symbol
df = storage.read(provider="yahoo", frequency="daily", symbol="AAPL")

# Read all symbols for a provider
df = storage.read_all(provider="yahoo", frequency="daily")
```

---

## Space Efficiency

### Compression

Parquet provides excellent compression for financial data:

| Data Type | Typical Compression |
|-----------|---------------------|
| Timestamps | 10-20x (dictionary encoding) |
| Symbols | 50-100x (dictionary encoding) |
| Prices | 3-5x (zstd compression) |
| Volume | 5-10x |

**Overall: 5-10x compression vs CSV**

### Example Storage Requirements

| Dataset | Rows | CSV Size | Parquet Size |
|---------|------|----------|--------------|
| 1 symbol, 5 years daily | ~1,250 | 100 KB | 15 KB |
| 500 symbols, 5 years daily | ~625K | 50 MB | 8 MB |
| 500 symbols, 5 years minute | ~390M | 30 GB | 5 GB |

---

## Migration from Flat Files

If you have existing flat Parquet files:

```python
from ml4t.data.storage.hive import HiveStorage, StorageConfig
import polars as pl

# Read existing flat file
df = pl.read_parquet("existing_data.parquet")

# Setup Hive storage
storage = HiveStorage(config=StorageConfig(base_path="~/ml4t-data"))

# Migrate each symbol
for symbol in df['symbol'].unique():
    symbol_df = df.filter(pl.col("symbol") == symbol)
    storage.write(symbol_df, provider="yahoo", frequency="daily", symbol=symbol)
    print(f"Migrated {symbol}: {len(symbol_df)} rows")
```

---

## Best Practices

### 1. Consistent Partition Keys

Always use the standard hierarchy:
```
provider/frequency/year/month/[day]/symbol/
```

### 2. Don't Over-Partition

- Daily data: year/month is sufficient
- Only add day partitions for intraday data
- Symbol subdirectories prevent large directories

### 3. Use Predicate Pushdown

```python
# Good: Filter pushed to storage layer
pl.scan_parquet(...).filter(pl.col("year") == 2024)

# Bad: Full scan then filter
pl.read_parquet(...).filter(pl.col("year") == 2024)
```

### 4. Monitor Partition Sizes

```python
from pathlib import Path

def check_partitions(base_path: str):
    """Check partition file sizes."""
    for pq in Path(base_path).rglob("*.parquet"):
        size_mb = pq.stat().st_size / 1024 / 1024
        if size_mb > 200:
            print(f"Warning: Large partition {pq}: {size_mb:.1f} MB")
        elif size_mb < 0.01:
            print(f"Warning: Tiny partition {pq}: {size_mb*1024:.1f} KB")
```

---

## See Also

- [Incremental Architecture](INCREMENTAL_ARCHITECTURE.md) - Update strategies
- [Workflow Notebooks](../../notebooks/storage/01_hive_partitioning.py) - Interactive examples
- [Provider README](../providers/README.md) - Provider-specific storage notes
