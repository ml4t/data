# API Reference

API documentation for the `ml4t-data` library, generated from
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
manager = DataManager(storage=storage)
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

## Acquisition compatibility

The opt-in `ml4t.data.compat.empyrical` module adapts acquisition, CSV caches and paths only.
Analytical Empyrical replacements belong in Diagnostic. No legacy package is a Data runtime
dependency. See the [migration workflow](../getting-started/migration.md#empyrical-acquisition-utilities)
and [FRED](../providers/fred.md), [Fama-French](../providers/fama_french.md) and
[Yahoo](../providers/yahoo.md) provider contracts.

::: ml4t.data.compat.empyrical
    options:
      show_source: false
      heading_level: 3
      members:
        - get_fama_french
        - load_portfolio_risk_factors
        - get_treasury_yield
        - get_symbol_returns_from_yahoo
        - default_returns_func
        - get_returns_cached
        - cache_dir
        - data_path
        - ensure_directory

## Storage

### StorageConfig

Validated configuration for the storage backend. Controls partitioning strategy,
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

Hive-partitioned storage with configurable time-based partitioning and
time-range partition pruning.

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

Pydantic model for top-level library configuration. Reads `ML4T_DATA_PATH` and
`ML4T_DATA_LOG_LEVEL`. Former QLDM names are covered by the
[migration guide](../getting-started/migration.md).

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

## Futures Expiry Rolls

`TimeBasedRoll` selects the nearest available contract until its configured pre-expiry trigger.
`FirstNoticeDateRoll` delegates the same calendar selection while retaining its existing
first-notice offset convention and five-session cash-settled fallback.

| Mode | Offset units | Calendar behavior |
|------|--------------|-------------------|
| `use_business_days=True` (default) | Authoritative sessions strictly before expiration | Explicit `calendar`, then `feed_spec`, then NYSE |
| `use_business_days=False` | Elapsed civil days | No session counting; an explicitly disabled calendar is allowed |
| Session-count mode with `calendar=None` | Unsupported observation-based expiry offset | Raises clearly; does not substitute NYSE or 24/7 |

The legacy `use_business_days` name now counts the selected exchange's sessions rather than
weekdays. `calendar="24/7"` counts a real continuous schedule. Contract exchange identity does not
select a calendar. Pass the feed declaration explicitly when constructing a roll strategy from
stored metadata. Named calendars are validated before selection.

An offset of zero uses expiration itself. A missing scheduled trigger does not move the trigger
backward to an observed date: selection switches at the first eligible later observation, subject
to the existing contract availability and paired-price checks. Explicit elapsed-day mode also
stops snapping its target backward. Neither mode reinterprets missing data as a closure or changes
elapsed days into row counts.

::: ml4t.data.futures.roll.TimeBasedRoll
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - select_contracts

::: ml4t.data.futures.roll.FirstNoticeDateRoll
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - select_contracts

## Versioned individual-contract inputs

`prepare_futures_inputs` prepares individual bars, dated definitions, independently available
volume/OI facts, one current selection per root/venue and paired source roll marks. It does not
execute a continuous series or infer fills, margin or settlement. See the
[offline migration recipe](../getting-started/migration.md#prepare-individual-futures-inputs).

::: ml4t.data.futures.prepare_futures_inputs
    options:
      show_source: false
      heading_level: 4

::: ml4t.data.futures.FuturesInputs
    options:
      show_source: false
      heading_level: 4

## Price-table representations

Convert one price field between native long observations and a pandas clock-by-asset panel.
The functions preserve supplied values and missing cells; they do not fill prices, infer
availability or calculate returns. Other observation fields remain in the original table.
The panel's attrs retain the declared feed, source dtypes and sparse presence information.
Default metadata supports pandas Parquet round trips; caller-supplied provenance must also be
JSON-serializable to use pandas' metadata export.

See the [offline migration recipe](../getting-started/migration.md#move-a-long-price-table-to-a-pandas-analytical-panel)
for required input columns, mapping conflicts and the qualified consumer versions.

::: ml4t.data.utils.conversion.to_price_panel
    options:
      show_source: false
      heading_level: 3

::: ml4t.data.utils.conversion.from_price_panel
    options:
      show_source: false
      heading_level: 3

## Corporate-action input preparation

Validate versioned source facts or apply supplied per-observation factors without changing the
existing split/dividend kernels. Source facts do not mutate broker positions or cash. Unavailable
terms remain explicit; requesting missing applicable terms fails before output.

See the [offline action migration recipe](../getting-started/migration.md#import-corporate-action-facts-without-changing-their-financial-definition)
for ratio direction, required source fields and native versus vendor adjustment definitions.

::: ml4t.data.adjustments.prepare_action_inputs
    options:
      show_source: false
      heading_level: 3

::: ml4t.data.adjustments.apply_adjustment_factors
    options:
      show_source: false
      heading_level: 3

## Dated identity and known-at history inputs

Use captured identity intervals and source availability for preparation,not an asset database or
runtime history request. The [offline dated-history recipe](../getting-started/migration.md#prepare-dated-identities-and-known-at-history)
shows symbol normalization,pre-entry observations and actual source revisions.

::: ml4t.data.assets.InstrumentRegistry
    options:
      show_source: false
      heading_level: 3
      members:
        - from_frame
        - resolve
        - lifetimes
        - identities
        - memberships

::: ml4t.data.adjustments.prepare_adjusted_history
    options:
      show_source: false
      heading_level: 3

::: ml4t.data.adjustments.AdjustedHistory
    options:
      show_source: false
      heading_level: 3

## Portable input capture

Capture selected tables and caller-declared effective configuration, then verify all members before
returning reconstructed native frames. See the [storage contract](../user-guide/storage.md#portable-input-snapshots)
and [migration recipe](../getting-started/migration.md#reconstruct-inputs-after-source-pruning).
Optional bundle exchange uses standard Zipline readers/writers only in the conversion environment.

::: ml4t.data.storage.SnapshotTable
    options:
      show_source: false
      heading_level: 3

::: ml4t.data.storage.InputSnapshot
    options:
      show_source: false
      heading_level: 3

::: ml4t.data.storage.export_snapshot
    options:
      show_source: false
      heading_level: 3

::: ml4t.data.storage.read_snapshot
    options:
      show_source: false
      heading_level: 3

::: ml4t.data.compat.zipline.export_bundle
    options:
      show_source: false
      heading_level: 3

::: ml4t.data.compat.zipline.make_bundle_ingest
    options:
      show_source: false
      heading_level: 3
