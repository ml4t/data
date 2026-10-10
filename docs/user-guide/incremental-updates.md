# Incremental Updates

`ml4t-data update` creates a dataset when it is absent and refreshes it when it already exists. The
stored key is `{asset_class}/{frequency}/{symbol}`.

## Create or Update One Dataset

The first call can define an explicit initial range:

```bash
ml4t-data update \
  --symbol AAPL \
  --provider yahoo \
  --asset-class equities \
  --frequency daily \
  --initial-start 2023-01-01 \
  --initial-end 2024-01-01 \
  --storage-path ./data
```

When the key does not exist, `update` performs an initial load. Without `--initial-start`, it asks
for `--initial-load-days` of history, subject to the provider's declared history limit.

Subsequent calls overlap the stored data by seven days by default:

```bash
ml4t-data update \
  --symbol AAPL \
  --asset-class equities \
  --frequency daily \
  --lookback-days 7 \
  --storage-path ./data
```

The overlap lets revised observations replace their stored values. The update merges on timestamp,
keeps the newest row for a duplicate timestamp, sorts the result, and publishes the replacement
through the storage backend's atomic generation mechanism.

If `--provider` is omitted for an existing dataset, the update reuses the provider recorded in its
metadata. Supplying the option overrides that value.

## Gap Handling

Gap detection and forward filling are enabled by default. Disable them when missing periods must
remain explicit:

```bash
ml4t-data update --symbol AAPL --no-fill-gaps --storage-path ./data
```

The detector uses the requested frequency and distinguishes crypto, which trades continuously, from
other asset classes. It treats common overnight and weekend intervals as expected for non-crypto
intraday data. This is a general interval check, not an exchange-calendar reconstruction. Use
[trading-session completion](sessions.md) when exact exchange schedules are required.

If a provider's history limit cannot reach the stored dataset, the updater records
`provider_history_limited` and skips gap filling across the unavailable interval.

## Python API

Configure a storage backend and pass it to `DataManager`:

```python
from ml4t.data import DataManager
from ml4t.data.storage import create_storage

storage = create_storage("./data", strategy="hive")
manager = DataManager(storage=storage)

key = manager.update(
    symbol="AAPL",
    provider="yahoo",
    frequency="daily",
    asset_class="equities",
    lookback_days=7,
    fill_gaps=True,
    initial_start="2023-01-01",
    initial_end="2024-01-01",
)
```

`DataManager.update()` returns the storage key. It raises when storage is not configured, stored data
is empty, metadata is invalid, or the provider or storage operation fails.

## Inspect Stored State

List keys and inspect one dataset:

```bash
ml4t-data list --storage-path ./data
ml4t-data info \
  --symbol AAPL \
  --asset-class equities \
  --frequency daily \
  --storage-path ./data
```

Show the overall metadata health classification:

```bash
ml4t-data status --detailed --stale-days 3 --storage-path ./data
```

`status` classifies a dataset from its recorded end date, or its last update time when no end date
is present. It reports an error when the timestamp or row count metadata is missing or invalid.

## Update Configured Datasets

Use a YAML configuration for repeatable multi-dataset updates:

```bash
ml4t-data update-all --config ml4t-data.yaml --dry-run
ml4t-data update-all --config ml4t-data.yaml
```

Pass `--dataset NAME` to select one configured dataset. The dry run reports the planned work without
fetching or writing data.

## Operational Notes

- Run an update only after the provider is expected to have finalized the latest requested period.
- Reuse one storage directory and the same asset-class and frequency values so updates address the
  intended key.
- Treat forward-filled rows as synthetic observations. For minute-session completion, the
  `is_imputed` column identifies inserted rows; the general updater does not add that marker.
- Use `ml4t-data validate` after an update when schema, OHLC relationships, duplicates, or anomaly
  checks are part of the ingestion contract.
