# API Reference

Complete API documentation for the `ml4t-data` library, auto-generated from
source docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

---

## DataManager

The primary entry point for all data operations. `DataManager` is a facade that
delegates to focused manager classes for configuration, fetching, storage,
metadata, and batch operations.

```python
from ml4t.data import DataManager

# Fetch-only (no storage)
manager = DataManager()
df = manager.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")

# With storage for load/update workflows
from ml4t.data.storage import HiveStorage, StorageConfig

storage = HiveStorage(StorageConfig(base_path="./data"))
manager = DataManager(storage=storage, use_transactions=True)
key = manager.load("AAPL", "2024-01-01", "2024-12-31")
key = manager.update("AAPL")
```

::: ml4t.data.DataManager
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - fetch
        - fetch_batch
        - batch_load
        - batch_load_universe
        - batch_load_from_storage
        - load
        - import_data
        - update
        - list_symbols
        - get_metadata
        - assign_sessions
        - complete_sessions
        - update_all
        - list_providers
        - get_provider_info
        - clear_cache
        - config
        - output_format
        - storage

---

## Storage

### StorageConfig

Dataclass configuring the storage backend. Controls partitioning strategy,
compression, locking, and metadata tracking.

```python
from ml4t.data.storage import StorageConfig

# Hive-partitioned storage for minute data
config = StorageConfig(
    base_path="./market_data",
    strategy="hive",
    partition_granularity="day",
    compression="zstd",
)

# Flat storage for small datasets
config = StorageConfig(
    base_path="./data",
    strategy="flat",
    compression="snappy",
)
```

::: ml4t.data.storage.backend.StorageConfig
    options:
      show_source: false
      heading_level: 4

### StorageBackend

Abstract base class defining the storage interface. All backends (Hive, Flat)
implement this contract.

::: ml4t.data.storage.backend.StorageBackend
    options:
      show_source: false
      heading_level: 4
      members:
        - __init__
        - write
        - read
        - list_keys
        - exists
        - delete
        - get_metadata

### HiveStorage

Hive-partitioned storage with configurable time-based partitioning. Delivers
7x query performance improvement for time-range queries via partition pruning.

```python
from ml4t.data.storage import HiveStorage, StorageConfig

config = StorageConfig(
    base_path="./data",
    partition_granularity="month",  # year, month, day, or hour
)
storage = HiveStorage(config)

# Write data (partitions by timestamp automatically)
storage.write(df, "equities/daily/AAPL")

# Read with partition pruning
from datetime import datetime
lf = storage.read(
    "equities/daily/AAPL",
    start_date=datetime(2024, 6, 1),
    end_date=datetime(2024, 12, 31),
    columns=["timestamp", "close", "volume"],
)
df = lf.collect()
```

::: ml4t.data.storage.hive.HiveStorage
    options:
      show_source: false
      heading_level: 4
      members:
        - __init__
        - write
        - read
        - list_keys
        - exists
        - delete
        - get_latest_timestamp
        - save_chunk
        - update_combined_file
        - read_data
        - update_metadata

### FlatStorage

Simple single-file-per-key storage. Suitable for smaller datasets or when
partition pruning is not beneficial.

```python
from ml4t.data.storage import FlatStorage, StorageConfig

config = StorageConfig(base_path="./data", strategy="flat")
storage = FlatStorage(config)

storage.write(df, "reference/spy")
lf = storage.read("reference/spy")
```

::: ml4t.data.storage.flat.FlatStorage
    options:
      show_source: false
      heading_level: 4
      members:
        - __init__
        - write
        - read
        - list_keys
        - exists
        - delete

### create_storage

Factory function for creating storage backends from a strategy name.

```python
from ml4t.data.storage import create_storage

storage = create_storage("./data", strategy="hive", partition_granularity="day")
```

::: ml4t.data.storage.create_storage
    options:
      show_source: false
      heading_level: 4

---

## Providers

### BaseProvider

Abstract base class for all data providers. Composes rate-limiting,
circuit-breaker, validation, and HTTP session mixins into a single base.

Concrete providers implement either:

- `_fetch_and_transform_data()` for a single-step workflow, or
- `_fetch_raw_data()` + `_transform_data()` for a two-step workflow.

```python
from ml4t.data.providers.base import BaseProvider
import polars as pl

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    def _fetch_and_transform_data(self, symbol, start, end, frequency):
        # Fetch from API and return canonical OHLCV DataFrame
        ...
```

::: ml4t.data.providers.base.BaseProvider
    options:
      show_source: false
      heading_level: 4
      members:
        - __init__
        - name
        - fetch_ohlcv
        - fetch_ohlcv_async
        - capabilities
        - close

### ProviderCapabilities

Frozen dataclass describing what a provider supports (intraday, crypto, forex,
futures, authentication requirements, rate limits).

```python
from ml4t.data.providers.protocols import ProviderCapabilities

caps = ProviderCapabilities(
    supports_intraday=True,
    supports_crypto=True,
    requires_api_key=True,
    rate_limit=(120, 60.0),  # 120 calls per 60 seconds
)
```

::: ml4t.data.providers.protocols.ProviderCapabilities
    options:
      show_source: false
      heading_level: 4

### OHLCVProvider (Protocol)

Structural typing protocol for OHLCV providers. Any class implementing
`name`, `fetch_ohlcv()`, and `capabilities()` satisfies this protocol
without inheriting from `BaseProvider`.

::: ml4t.data.providers.protocols.OHLCVProvider
    options:
      show_source: false
      heading_level: 4

---

## Configuration

### Config

Pydantic model for top-level library configuration. Reads defaults from
environment variables (`QLDM_DATA_ROOT`, `QLDM_LOG_LEVEL`).

```python
from ml4t.data import Config

# Use defaults
config = Config()

# Override data root
config = Config(data_root="/mnt/fast/market_data", log_level="DEBUG")
```

::: ml4t.data.core.config.Config
    options:
      show_source: false
      heading_level: 4
      members:
        - __init__
        - data_root
        - log_level
        - storage
        - retry
        - cache
        - validation
        - base_dir

### RetryConfig

Configuration for automatic retry with exponential backoff.

::: ml4t.data.core.config.RetryConfig
    options:
      show_source: false
      heading_level: 4

### CacheConfig

Configuration for in-memory caching.

::: ml4t.data.core.config.CacheConfig
    options:
      show_source: false
      heading_level: 4

---

## Exceptions

All exceptions inherit from `ML4TDataError`, which carries an optional
`details` dictionary for structured error context.

```
ML4TDataError
├── ProviderError
│   ├── NetworkError
│   │   └── RateLimitError
│   ├── AuthenticationError
│   ├── DataValidationError
│   ├── SymbolNotFoundError
│   └── DataNotAvailableError
├── StorageError
│   └── LockError
├── ConfigurationError
└── CircuitBreakerOpenError
```

### ML4TDataError

::: ml4t.data.core.exceptions.ML4TDataError
    options:
      show_source: false
      heading_level: 4

### ProviderError

::: ml4t.data.core.exceptions.ProviderError
    options:
      show_source: false
      heading_level: 4

### NetworkError

::: ml4t.data.core.exceptions.NetworkError
    options:
      show_source: false
      heading_level: 4

### RateLimitError

::: ml4t.data.core.exceptions.RateLimitError
    options:
      show_source: false
      heading_level: 4

### AuthenticationError

::: ml4t.data.core.exceptions.AuthenticationError
    options:
      show_source: false
      heading_level: 4

### DataValidationError

::: ml4t.data.core.exceptions.DataValidationError
    options:
      show_source: false
      heading_level: 4

### SymbolNotFoundError

::: ml4t.data.core.exceptions.SymbolNotFoundError
    options:
      show_source: false
      heading_level: 4

### DataNotAvailableError

::: ml4t.data.core.exceptions.DataNotAvailableError
    options:
      show_source: false
      heading_level: 4

### StorageError

::: ml4t.data.core.exceptions.StorageError
    options:
      show_source: false
      heading_level: 4

### LockError

::: ml4t.data.core.exceptions.LockError
    options:
      show_source: false
      heading_level: 4

### ConfigurationError

::: ml4t.data.core.exceptions.ConfigurationError
    options:
      show_source: false
      heading_level: 4

### CircuitBreakerOpenError

::: ml4t.data.core.exceptions.CircuitBreakerOpenError
    options:
      show_source: false
      heading_level: 4
