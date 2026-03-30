# Incremental Updates

`ml4t-data` supports two update styles:

- high-level updates through `DataManager.update()` for normal stored datasets
- lower-level update planning and metadata tracking through the utilities in
  `ml4t.data.update_manager`

The goal in both cases is the same: fetch only what changed, merge it into
existing storage, detect gaps, and keep enough metadata to monitor dataset
health over time.

## What the Current Update Stack Does

For a stored dataset such as `equities/daily/AAPL`, the update flow is:

1. Read the existing data from storage.
2. Use the latest stored timestamp plus a configurable lookback window to decide
   what range to re-fetch.
3. Fetch fresh data through the configured provider.
4. Merge and deduplicate by timestamp.
5. Optionally run gap detection and validation.
6. Persist the updated dataset and metadata.

The main entry points are:

| Surface | Use when | Notes |
|---|---|---|
| `DataManager.update()` | You already have a storage-backed manager and want the normal library workflow | Best default for production pipelines |
| `ml4t-data update` | You want to update one stored symbol from the CLI | Supports `incremental`, `append_only`, `full_refresh`, and `backfill` strategies |
| `ml4t-data update-all` | You manage recurring datasets from a YAML config | Good for book datasets and cron-style automation |
| `IncrementalUpdater` | You need direct access to range planning and update strategies | Lower-level API in `ml4t.data.update_manager` |
| `MetadataTracker` | You need status, history, or health summaries | Backing store for `status` and `health` style reporting |

## Recommended Python Workflow

```python
from pathlib import Path

from ml4t.data import DataManager
from ml4t.data.storage import HiveStorage
from ml4t.data.storage.backend import StorageConfig

storage = HiveStorage(StorageConfig(base_path=Path("./data")))
manager = DataManager(storage=storage, enable_validation=True)

# First run: load historical data into storage
manager.load(
    symbol="AAPL",
    start="2020-01-01",
    end="2024-12-31",
    provider="yahoo",
    asset_class="equities",
    frequency="daily",
)

# Later runs: only fetch the recent range plus a small lookback window
key = manager.update(
    symbol="AAPL",
    provider="yahoo",
    asset_class="equities",
    frequency="daily",
    lookback_days=7,
    fill_gaps=True,
)

print(key)  # equities/daily/AAPL
```

Use `create_storage(..., strategy="flat")` for smaller datasets, but prefer
Hive storage for time-series updates because it aligns with the library's
read, merge, and pruning workflow.

## CLI Workflows

### Update one dataset

```bash
ml4t-data update -s AAPL --provider yahoo --storage-path ./data
ml4t-data update -s AAPL --strategy full_refresh --storage-path ./data
ml4t-data update -s AAPL --strategy backfill --start 2020-01-01 --end 2020-12-31 \
    --storage-path ./data
```

### Inspect update status

```bash
ml4t-data status --storage-path ./data
ml4t-data status --detailed --storage-path ./data
ml4t-data health --storage-path ./data
```

### Run recurring dataset updates from YAML

```yaml
storage:
  path: ~/ml4t-data

datasets:
  etf_core:
    provider: yahoo
    symbols: [SPY, QQQ, IWM, TLT, GLD]
    frequency: daily

  macro:
    provider: fred
    symbols: [DGS3MO, CPIAUCSL, UNRATE]
    frequency: daily
```

```bash
ml4t-data update-all -c ml4t-data.yaml
ml4t-data update-all -c ml4t-data.yaml --dataset macro
ml4t-data update-all -c ml4t-data.yaml --dry-run
```

## Gap Detection and Validation

Incremental updates are useful only if the resulting data stays trustworthy.
`ml4t-data` uses two complementary mechanisms:

- `ml4t.data.utils.gaps.GapDetector` detects missing periods and summarizes gap
  counts and durations
- `OHLCVValidator` checks structural issues such as bad OHLC relationships,
  duplicates, negative values, and stale or extreme-return patterns

```python
from ml4t.data.utils.gaps import GapDetector
from ml4t.data.validation import OHLCVValidator

gaps = GapDetector().detect_gaps(df, frequency="daily", is_crypto=False)
validation = OHLCVValidator(max_return_threshold=0.5).validate(df)

print(len(gaps), validation.passed)
```

For crypto and other 24/7 markets, make sure the gap detector is configured with
the right market assumptions. Intraday equities and continuous crypto have very
different definitions of an "expected" gap.

## Metadata and Health Tracking

`MetadataTracker` records update history and dataset summaries under
`.metadata/` inside the storage root. This is what powers:

- `ml4t-data status`
- `ml4t-data health`
- per-dataset update histories and freshness checks

Each update record includes the provider, update type, date range, row counts,
duration, and any gap-filling information. That makes it practical to answer:

- when was this dataset last updated?
- was the last run incremental or a full refresh?
- how many rows were added or rewritten?
- is the dataset stale or healthy?

## Choosing an Update Strategy

| Strategy | Best for | Tradeoff |
|---|---|---|
| `incremental` | Normal daily or hourly refreshes | Re-fetches a short overlap window to stay robust |
| `append_only` | Immutable append workflows | Will not rewrite existing history |
| `full_refresh` | Provider corrections or schema changes | Most expensive option |
| `backfill` | Filling missing historical periods | Useful after outages or provider switches |

Default to `incremental`. Use `full_refresh` only when the upstream source or
your storage layout changed enough that merging old and new data is unsafe.

## See It in the Book

The book codebase demonstrates the same update concepts from notebook-scale
examples through reusable scripts:

- [Complete pipeline in Chapter 2](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/17_complete_pipeline.py)
- [Data management in Chapter 2](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/18_data_management.py)
- [Incremental updates in Chapter 2](https://github.com/ml4t/third-edition/blob/main/code/02_financial_data_universe/19_incremental_updates.py)
- [Canonical dataset downloader](https://github.com/ml4t/third-edition/blob/main/code/data/download_all.py)

The progression is intentional:

1. The chapter scripts show the mechanics directly with `DataManager`,
   `HiveStorage`, `GapDetector`, and `OHLCVValidator`.
2. The `code/data/` download scripts turn those ideas into repeatable dataset
   pipelines.
3. Your own production workflow can usually reuse the same library calls with a
   project-specific config and storage root.

## Related Guides

- [Storage](storage.md)
- [Data Quality](data-quality.md)
- [CLI Reference](cli-reference.md)
- [Book Guide](../book-guide/index.md)
