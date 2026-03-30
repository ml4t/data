# Configuration

ml4t-data uses YAML configuration files validated by Pydantic models. The config
system supports environment variable interpolation, file includes, and
per-asset-class defaults.

Use this page when you want a repeatable, config-driven workflow instead of
ad hoc provider calls. It is the right place to start for scheduled downloads,
multi-dataset pipelines, and book-style dataset orchestration.

## Minimal Working Example

```yaml
storage:
  path: ~/ml4t-data

datasets:
  etf_core:
    provider: yahoo
    symbols: [SPY, QQQ, IWM, TLT, GLD]
    frequency: daily
```

```python
from pathlib import Path

from ml4t.data.config import load_config

config = load_config(Path("ml4t-data.yaml"))
print(config)
```

## File Discovery

The `ConfigLoader` searches these locations in order:

1. `./ml4t.data.yaml`
2. `./ml4t.data.yml`
3. `./.ml4t-data.yaml`
4. `./.ml4t-data.yml`
5. `./config/ml4t.data.yaml`
6. `~/.config/ml4t-data/config.yaml`

You can also pass an explicit path:

```python
from ml4t.data.config import load_config

config = load_config(Path("my-config.yaml"))
```

## Full Configuration Example

```yaml
version: "1.0"
base_dir: ./data
log_level: INFO
parallel_downloads: 4

# Storage backend
storage:
  strategy: hive            # "hive" (partitioned) or "flat" (single file)
  base_path: ~/ml4t-data
  compression: zstd         # zstd, lz4, snappy, or none
  partition_granularity: month  # year, month, day, or hour
  atomic_writes: true
  enable_locking: true
  metadata_tracking: true

# Data providers
providers:
  - name: yahoo
    type: yahoo
    enabled: true
    rate_limit:
      requests_per_second: 5.0
      retry_max_attempts: 3
    timeout: 30

  - name: databento
    type: databento
    api_key: ${DATABENTO_API_KEY}
    rate_limit:
      requests_per_second: 10.0
      circuit_breaker_threshold: 5
      circuit_breaker_timeout: 60

# Symbol universes
universes:
  - name: tech_stocks
    symbols: [AAPL, MSFT, GOOGL, AMZN, META]
    asset_class: equity

  - name: sp500
    file: symbols/sp500.txt   # one symbol per line
    provider: yahoo
    asset_class: equity

# Datasets
datasets:
  - name: us_equities
    universe: sp500
    provider: yahoo
    frequency: daily
    asset_class: equity
    update_mode: incremental
    validation_enabled: true
    anomaly_detection: false

  - name: crypto_spot
    symbols: [BTC, ETH, SOL]
    provider: binance_api
    frequency: hourly
    asset_class: crypto

# Workflows
workflows:
  - name: daily_update
    datasets: [us_equities, crypto_spot]
    schedule:
      type: daily
      time: "18:00"
      timezone: US/Eastern
    on_error: continue
```

## Environment Variables

API keys and secrets support `${VAR}` interpolation with optional defaults:

```yaml
providers:
  - name: polygon
    api_key: ${POLYGON_API_KEY}           # required
    api_secret: ${POLYGON_SECRET:default}  # with fallback
```

The `env` section defines variables that are set if not already present:

```yaml
env:
  ML4T_DATA_DIR: ~/ml4t-data
  DEFAULT_PROVIDER: yahoo
```

ml4t-data also reads `.env` files automatically via Pydantic Settings.

## Key Configuration Sections

### Storage

| Field | Default | Description |
|-------|---------|-------------|
| `strategy` | `hive` | `hive` for partitioned Parquet, `flat` for single files |
| `base_path` | `./data` | Base directory (supports `~` expansion) |
| `compression` | `zstd` | Parquet compression: `zstd`, `lz4`, `snappy`, `none` |
| `partition_granularity` | `month` | Hive partition level: `year`, `month`, `day`, `hour` |
| `atomic_writes` | `true` | Write to temp file then rename |
| `enable_locking` | `true` | File locking for concurrent access |
| `metadata_tracking` | `true` | JSON manifest files alongside data |

### Providers

Each provider entry configures connection parameters:

| Field | Default | Description |
|-------|---------|-------------|
| `name` | required | Provider identifier |
| `type` | required | `yahoo`, `binance_api`, `binance_bulk`, `cryptocompare`, `databento`, `oanda`, `polygon`, `mock` |
| `enabled` | `true` | Toggle provider on/off |
| `api_key` | `null` | API key (use `${ENV_VAR}` format) |
| `rate_limit` | see below | Rate limiting and circuit breaker config |
| `timeout` | `30` | Request timeout in seconds |
| `cache_enabled` | `true` | Response caching |
| `cache_ttl` | `3600` | Cache time-to-live in seconds |

Rate limiting sub-config:

| Field | Default | Description |
|-------|---------|-------------|
| `requests_per_second` | `10.0` | Maximum request rate |
| `burst_size` | `1` | Burst allowance |
| `retry_max_attempts` | `3` | Retry count |
| `retry_backoff_factor` | `2.0` | Exponential backoff multiplier |
| `circuit_breaker_threshold` | `5` | Failures before circuit opens |
| `circuit_breaker_timeout` | `60` | Seconds before circuit half-opens |

### Schedules

Workflows support five schedule types:

```yaml
# Cron expression
schedule:
  type: cron
  cron: "0 18 * * 1-5"

# Fixed interval (seconds)
schedule:
  type: interval
  interval: 3600

# Daily at specific time
schedule:
  type: daily
  time: "18:00"
  timezone: US/Eastern

# Weekly
schedule:
  type: weekly
  time: "09:00"
  weekday: 0  # Monday

# Relative to market hours
schedule:
  type: market_hours
  market_close_offset: 30  # 30 minutes before close
```

## File Includes

Split large configs across files with the `include` directive:

```yaml
# ml4t.data.yaml
include:
  - providers/yahoo.yaml
  - providers/databento.yaml
  - universes/equities.yaml

datasets:
  - name: us_equities
    universe: sp500
    provider: yahoo
```

Included files are merged recursively. The main file takes priority over includes.

## Validation

Use `ConfigValidator` to check for consistency errors before running:

```python
from ml4t.data.config import load_config, ConfigValidator

config = load_config()
validator = ConfigValidator(config)

if not validator.validate():
    for error in validator.errors:
        print(f"ERROR: {error}")
    for warning in validator.warnings:
        print(f"WARNING: {warning}")
```

The validator checks for:

- Duplicate provider, dataset, or universe names
- Datasets referencing non-existent providers or universes
- Workflows referencing non-existent datasets
- Invalid date ranges (start >= end)
- Missing cron expressions or interval values in schedules
- Orphaned providers or datasets not used by any workflow

## Programmatic Access

```python
from ml4t.data.config import DataConfig

# Load from YAML
config = DataConfig.from_yaml("ml4t.data.yaml")

# Look up components
provider = config.get_provider("yahoo")
universe = config.get_universe("sp500")
dataset = config.get_dataset("us_equities")

# Validate references
issues = config.validate_config()

# Save back to YAML
config.to_yaml("ml4t.data.yaml")
```

## See It In The Book

The book codebase uses the same pattern for canonical dataset automation:

- [Download orchestrator](https://github.com/ml4t/third-edition/blob/main/code/data/download_all.py)
- [ETF config](https://github.com/ml4t/third-edition/blob/main/code/data/etfs/config.yaml)
- [Crypto config](https://github.com/ml4t/third-edition/blob/main/code/data/crypto/config.yaml)
- [Macro config](https://github.com/ml4t/third-edition/blob/main/code/data/macro/config.yaml)

These files show how the book moves from one-off notebook exploration to
reusable dataset definitions that can be updated repeatedly.

## Next Steps

- [Incremental Updates](incremental-updates.md)
- [Storage](storage.md)
- [CLI Reference](cli-reference.md)
- [Book Guide](../book-guide/index.md)
