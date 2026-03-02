# Storage

ml4t-data stores time-series data as Parquet files with two backend strategies:
**Hive** (partitioned by time) and **Flat** (single file per key). Both backends
share atomic writes, file locking, metadata tracking, and Polars lazy evaluation.

## Choosing a Backend

| | Hive | Flat |
|---|------|------|
| **Best for** | Large datasets, time-range queries | Small datasets, simple access |
| **Layout** | Directory tree with `year=.../month=.../data.parquet` | Single `.parquet` file per key |
| **Query speed** | 7x faster for date-filtered reads (partition pruning) | Reads entire file every time |
| **Write cost** | Higher (one file per partition) | Lower (single file) |
| **Default** | Yes | No |

```python
from ml4t.data.storage import create_storage

# Hive storage (default)
storage = create_storage("./data", strategy="hive")

# Flat storage
storage = create_storage("./data", strategy="flat")
```

## Partition Granularity

Hive storage partitions data by time. Choose granularity based on your data
frequency to keep partition sizes in the 200-5,000 row range:

| Granularity | Partition columns | Best for | Rows/partition (stocks) |
|-------------|-------------------|----------|------------------------|
| `year` | `year/` | Daily data | ~252 |
| `month` | `year=/month=/` | Hourly data | ~720 |
| `day` | `year=/month=/day=/` | Minute data | ~1,440 |
| `hour` | `year=/month=/day=/hour=/` | Second/tick data | ~3,600 |

```python
from ml4t.data.storage import HiveStorage, StorageConfig

# Daily equity data -- partition by year
config = StorageConfig(
    base_path="./data",
    partition_granularity="year",
)
storage = HiveStorage(config)

# Minute crypto data -- partition by day
config = StorageConfig(
    base_path="./data",
    partition_granularity="day",
)
storage = HiveStorage(config)
```

On-disk layout for month-level partitioning:

```
data/
  AAPL/
    year=2024/
      month=1/
        data.parquet
      month=2/
        data.parquet
      ...
    year=2025/
      month=1/
        data.parquet
  .metadata/
    AAPL.json
```

## Reading and Writing

Both backends use the same `StorageBackend` interface.

### Writing Data

```python
import polars as pl

df = pl.DataFrame({
    "timestamp": [...],
    "open": [...],
    "high": [...],
    "low": [...],
    "close": [...],
    "volume": [...],
})

# Write with a storage key
storage.write(df, "AAPL")
```

The write operation:

1. Adds partition columns derived from `timestamp` (Hive only)
2. Groups data by partition values
3. Writes each partition atomically (temp file + rename)
4. Updates the JSON metadata manifest

### Reading Data

All reads return a Polars `LazyFrame` for deferred execution:

```python
# Read all data for a key
lf = storage.read("AAPL")
df = lf.collect()

# Date-filtered read (Hive prunes partitions before scanning)
from datetime import datetime

lf = storage.read(
    "AAPL",
    start_date=datetime(2024, 6, 1),
    end_date=datetime(2024, 12, 31),
)

# Column projection (only read what you need)
lf = storage.read("AAPL", columns=["timestamp", "close", "volume"])
```

With Hive storage, date filters prune entire partition directories before
any Parquet file is opened, giving measured 7x speedup on typical queries.

### Other Operations

```python
# List all stored keys
keys = storage.list_keys()  # ["AAPL", "BTC-USD", ...]

# Check existence
if storage.exists("AAPL"):
    ...

# Delete all data for a key
storage.delete("AAPL")

# Read metadata
meta = storage.get_metadata("AAPL")
# {"last_updated": "...", "row_count": 5040, "schema": [...], ...}
```

## StorageConfig Options

| Field | Default | Description |
|-------|---------|-------------|
| `base_path` | required | Base directory for all data |
| `strategy` | `"hive"` | `"hive"` or `"flat"` |
| `compression` | `"zstd"` | Parquet compression: `zstd`, `lz4`, `snappy`, or `None` |
| `partition_granularity` | `"month"` | `year`, `month`, `day`, `hour` (Hive only) |
| `atomic_writes` | `True` | Write to temp file then rename |
| `enable_locking` | `True` | File locking for concurrent access |
| `metadata_tracking` | `True` | JSON manifest files in `.metadata/` |
| `generate_profile` | `True` | Column-level statistics on write |

## Incremental Updates

Hive storage supports chunk-based incremental updates for streaming workflows:

```python
from datetime import datetime

# Save a new data chunk
chunk_path = storage.save_chunk(
    data=new_df,
    symbol="AAPL",
    provider="yahoo",
    start_time=datetime(2024, 12, 1),
    end_time=datetime(2024, 12, 31),
)

# Merge chunk into the main combined file (deduplicates by timestamp)
new_rows = storage.update_combined_file(new_df, symbol="AAPL", provider="yahoo")

# Get latest timestamp for a symbol (to know where to resume)
latest = storage.get_latest_timestamp("AAPL", "yahoo")
```

## Data Profiling

The `ProfileMixin` adds dataset profiling to any data manager class.
Profiles contain column-level statistics (dtype, null count, min/max, mean, std)
stored as JSON alongside the data.

```python
from ml4t.data.storage import generate_profile, save_profile, load_profile

# Generate a profile from a DataFrame
profile = generate_profile(df, source="ETFDataManager")
print(profile.summary())
# Dataset Profile (generated: 2024-12-15T10:30:00)
#   Rows: 25,200
#   Columns: 7
#   Date range: 2020-01-02 to 2024-12-13
#   Column Details:
#     close: Float64 (25200 unique, 0.0% null) [mean=156.3, std=42.1]
#     volume: Int64 (24891 unique, 0.0% null) [mean=7.83e+07, std=4.21e+07]

# Save and load profiles
save_profile(profile, Path("data/AAPL_profile.json"))
profile = load_profile(Path("data/AAPL_profile.json"))

# Convert to DataFrame for analysis
profile_df = profile.to_dataframe()
```

To add profiling to a custom data manager, implement the `ProfileMixin`:

```python
from ml4t.data.storage import ProfileMixin

class MyDataManager(ProfileMixin):
    def _get_profile_data(self) -> pl.DataFrame:
        return self.load_all()

    def _get_profile_data_path(self) -> Path:
        return self.storage_path / "data.parquet"

    def _get_profile_source_name(self) -> str:
        return "MyDataManager"

# Now available:
profile = manager.generate_profile()  # generates and saves
profile = manager.load_profile()      # loads existing
```

## Concurrency and Safety

Both backends use two mechanisms for safe concurrent access:

- **Atomic writes**: Data is written to a temporary file first, then renamed
  to the target path. This prevents readers from seeing partial writes.
- **File locking**: Metadata updates use `filelock` to prevent corruption
  when multiple processes write simultaneously. Lock timeout is 10 seconds.

These are enabled by default and can be toggled via `StorageConfig`.
