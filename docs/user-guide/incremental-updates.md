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

Daily detection counts missing sessions from the selected exchange schedule, not elapsed days.
The updater currently selects NYSE for non-crypto data and 24/7 for crypto. Intraday detection still
uses approximate US overnight/weekend rules. It is not an exchange-calendar reconstruction. Use
[trading-session completion](sessions.md) when exact intraday schedules are required.

Detection does not itself fill or modify observations. The updater's separately enabled forward
fill uses a regular elapsed-time grid, not the selected exchange schedule, and can insert closed
dates when a real gap is present. Disable filling when those synthetic rows are inappropriate.

### Inspect Daily Gaps Without Filling

```python
from ml4t.data.utils.gaps import GapDetector
from ml4t.specs import FeedSpec

feed = FeedSpec(calendar="NYSE", time_contract_version=2)
gaps = GapDetector().detect_gaps(df, frequency="daily", feed_spec=feed)
```

The canonical `calendar` keyword overrides the feed's declaration. Otherwise the declared calendar
applies, then NYSE. `calendar="24/7"` expects a daily observation on weekends and holidays;
`calendar=None` disables schedule-based gap classification, regardless of timestamp spacing.
`is_crypto=True` remains a compatibility hint for 24/7 only when neither an explicit calendar nor a
feed declaration selects a calendar. Version-1 feed metadata with `calendar=None` still means
unspecified; version-2 metadata preserves an explicit null as disabled after serialization.

Daily inputs are session labels (Polars Date, or the date component of Datetime values in their
existing timezone). Non-midnight labels remain supported. Exchange holidays and weekends do not
count as missing NYSE sessions; a shortened session still requires one daily observation. Gap bounds
remain the two observed timestamps, while `missing_periods` counts only scheduled dates strictly
between them. Inspect each asset separately; observed dates from other assets must not define the
schedule. This operation does not validate off-session observations or discover gaps outside the
observed range.

Explicit event-time/bar-close metadata is rejected rather than reinterpreted as session labels.
Named calendar/feed declarations for intraday detection are not supported yet; legacy intraday
calls retain their existing behavior. Unknown calendars and malformed daily timestamps fail without
modifying the input. `fill_gaps()` remains a separate elapsed-grid operation, not calendar-aware
imputation.

### Plan Daily Backfill Acquisition

`ml4t.data.update_manager.GapDetector` is a separate acquisition-range planner. Its constructor and
`detect_gaps()` / `detect_gaps_in_storage()` accept the canonical `calendar` and shared `feed_spec`
declaration. Explicit selection overrides the declaration, then NYSE applies. Storage reads use the
dataset's persisted declaration unless a feed is supplied. Explicit `None` disables schedule-based
planning, including for absent datasets; 24/7 still expects weekend observations.

Daily plans include missing sessions at either boundary and within the inclusive requested date
range. Closed dates never become missing sessions. Returned `start` and `end` are inclusive session
labels encoded at UTC midnight, not event instants. The legacy `size_days` field and `tolerance_days`
argument count expected sessions in daily mode, not elapsed civil days. A range can span closures
between missing sessions. Backfill selects only scheduled labels from the provider's actual rows;
it does not manufacture observations or insert closed-date bars.

The legacy `exclude_weekends=False` argument selects continuous days only when no canonical calendar
or feed declaration is present. `exclude_weekends=True` now follows NYSE holidays as well as weekends.
Omission follows NYSE, so callers requiring the old elapsed-day counts should pass `calendar="24/7"`.
Intraday/weekly range planning retains its legacy elapsed-time behavior; named calendar/feed
declarations for those frequencies are not supported by this planner yet. `calendar=None` disables
classification. Session-label dates retain their existing timezone's civil date. Explicit event-time
or bar-close declarations are rejected for daily planning rather than silently relabeled. Declare
the feed's `timezone` when a non-UTC label must survive storage's canonical UTC reads; the planner
restores that declared timezone before interpreting dates.

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
