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

Updates preserve absent observations and null market values. Calendar gap checks are enabled by
default and never create rows or substitute prices or volumes. To skip those checks:

```bash
ml4t-data update --symbol AAPL --no-check-gaps --storage-path ./data
```

Daily detection counts missing sessions from the selected exchange schedule, not elapsed days.
The Python updater resolves its explicit `calendar`, then supplied feed or stored metadata, then
NYSE. Asset class and descriptive exchange metadata do not select a schedule. Continuous feeds
must declare `calendar="24/7"`; `calendar=None` disables schedule-derived detection.
Intraday detection requires the complete version-2 grid declaration described below, supplied
through `feed_spec` or retained in stored metadata. Undeclared intraday datasets must migrate their
metadata or use `--no-check-gaps` / `check_gaps=False` for updates.
[Minute-grid expansion](sessions.md) is a separate operation that exposes absent positions as nulls.

Synthetic imputation is unsupported. An explicit `fill_gaps=True` or `--fill-gaps` request fails
before acquisition or a dataset write. The legacy `fill_gaps=False` / `--no-fill-gaps` controls remain
compatible aliases for disabling checks. Configurations should use `check_gaps` instead.
The updater records whether checking ran and the detected missing-observation count; disabled
checks are not reported as proof of zero gaps. Source-history limits do not suppress detection.

For an incremental tail, `IncrementalUpdater.determine_update_range()` uses the stored feed and
the same bounded calendar report to select the first missing scheduled observation after the
latest real row. It does not advance minute data by a day or request an exchange closure. Its
keyword-only `calendar` and `feed_spec` arguments override stored declarations. With an explicitly
disabled calendar, it overlaps the latest real row rather than guessing the next timestamp.
Use the bounded report below for older interior gaps, not the incremental tail planner.

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

Undeclared daily timestamps and explicit `session_label` inputs are labels (Polars Date, or the
date component of Datetime values). Aware labels restore a declared `timezone` before extracting
dates, including after storage's canonical UTC read. Naive labels remain civil identifiers rather
than localized event clocks. Non-midnight labels remain supported. Exchange holidays and weekends do not
count as missing NYSE sessions; a shortened session still requires one daily observation. Gap bounds
remain the two observed timestamps, while `missing_periods` counts only scheduled dates strictly
between them. Inspect each asset separately; observed dates from other assets must not define the
schedule. This operation does not validate off-session observations or discover gaps outside the
observed range.

Explicit `event_time`, `bar_close` or bar-side metadata instead uses native authoritative session
mapping, even with daily frequency or midnight timestamps. Right-labeled continuous midnight belongs
to the preceding session; event time belongs to the new session. Out-of-session instants fail clearly.
Gap classification does not add a new off-session rejection policy for legacy daily labels. The
requested daily frequency cannot silently override a declared intraday frequency.
Unknown calendars and malformed daily timestamps fail without modifying the input.
The retired `fill_gaps()` helper rejects nonempty gap lists rather than manufacturing observations.

### Inspect Intraday Gaps Against a Declared Grid

```python
feed = FeedSpec(
    time_contract_version=2,
    calendar="NYSE",
    data_frequency="minute",
    timestamp_semantics="event_time",
    bar_grid_anchor="session_open",
    bar_label="left",
    partial_bar_policy="include",
)
gaps = GapDetector().detect_gaps(df, frequency="minute", feed_spec=feed)
```

Intraday classification requires the complete version-2 declaration above. `event_time` alone
does not choose a bar side. `bar_close` requires `bar_label="right"`. Naive timestamps also require
`timezone`; ambiguous/nonexistent local clocks fail rather than selecting a DST interpretation.
Supported fixed intervals use positive minute/hour counts, such as `5minute`, `15min` or `hourly`.
Omitted `frequency` inherits the feed declaration, otherwise defaults to daily. An explicit
frequency must agree with the declaration. Event-count bars and session-label timestamps
cannot define an intraday grid.

The grid starts at each authoritative tradable interval's open and restarts after lunch breaks
and declared calendar interruptions. Zero-length breaks on shortened sessions remove no bars.
`partial_bar_policy="include"` retains a shorter final bar; `"exclude"` omits it. Right labels name
the bar's end, including the actual shortened close. Holidays, early closes, DST, overnight sessions
and registered custom calendars use their selected schedule, not the observed rows. Off-grid
observations fail clearly instead of changing the definition to fit the input.

Run classification separately per asset. Missing bars strictly between the first and last observed
timestamps are counted, including a completely missing session, even if every asset has that outage.
Returned intraday bounds are UTC instants; `missing_periods` counts expected bars, while `duration`
is elapsed time. Detection does not infer missing boundaries outside the observed range or fill rows.
Explicit `calendar=None` disables classification, while `"24/7"` still expects continuous bars.

This replaces the old fixed-US-clock approximation. Unqualified legacy intraday calls now require
a complete feed declaration or explicit `calendar=None`; the detector does not invent an anchor,
label side or partial-bar policy. Daily classification is unchanged. Use the bounded range report
below to inspect leading/trailing absences; weekly grids are not supported.

### Inspect Missing Observations on Disk

`ml4t.data.update_manager.GapDetector` compares a stored dataset with its expected calendar grid.
It reads timestamps, not prices, and neither contacts a provider nor changes stored rows or metadata.
The report works with timestamped measurements as well as OHLCV data. Its constructor and
`detect_gaps()` / `detect_gaps_in_storage()` accept the canonical `calendar` and shared `feed_spec`
declaration. Explicit selection overrides the declaration, then NYSE applies. Storage reads use the
dataset's persisted declaration and frequency unless supplied explicitly. Explicit `None` disables
schedule-based detection, including for absent datasets; 24/7 still expects weekend observations.

For an existing spot-crypto minute dataset, supply or retain its complete grid declaration:

```python
from datetime import UTC, datetime

from ml4t.data.storage import create_storage
from ml4t.data.update_manager import GapDetector
from ml4t.specs import FeedSpec

storage = create_storage("./data", strategy="hive")
feed = FeedSpec(
    time_contract_version=2,
    calendar="24/7",
    data_frequency="minute",
    timestamp_semantics="event_time",
    bar_grid_anchor="session_open",
    bar_label="left",
    partial_bar_policy="include",
)
report = GapDetector(feed_spec=feed).detect_gaps_in_storage(
    storage,
    "crypto/minute/coinbase-btc-usd-spot",
    datetime(2026, 10, 16, 0, 0, tzinfo=UTC),
    datetime(2026, 10, 16, 23, 59, tzinfo=UTC),
)
for gap in report:
    print(gap["start"], gap["end"], gap["size_bars"])
```

Intraday bounds are inclusive UTC instants. Each returned range starts at its first absent bar
timestamp, ends at its last, and reports `size_bars`, an exact count of scheduled bars. Bounds need
not land on a bar: only expected timestamps inside them count. The report includes missing boundary
bars, wholly absent sessions and absent datasets. Breaks, maintenance, holidays, DST, shortened
sessions and partial-bar policy follow the selected schedule. It shares the arithmetic expected grid
with in-memory detection rather than constructing a second calendar or every absent row.

Existing Parquet files do not need to be imported into a Data storage backend. Read the file with
Polars and supply the same declaration and explicit coverage bounds:

```python
import polars as pl

report = GapDetector(feed_spec=feed).detect_gaps(
    pl.read_parquet("./prices.parquet"),
    start_date=datetime(2026, 10, 16, 0, 0, tzinfo=UTC),
    end_date=datetime(2026, 10, 16, 23, 59, tzinfo=UTC),
)
```

Without explicit bounds, frame detection checks between the first and last stored timestamps.
Explicit bounds also reveal missing boundary bars and wholly empty intervals. Use `timestamp_col`
in the feed declaration when the file names its time column differently. A Boolean `is_imputed=True`
marker identifies a known placeholder, which does not satisfy an expected observation. Zero volume
alone does not identify a placeholder. Previously fabricated rows without a marker cannot be
distinguished from real observations by timestamps alone.

An intraday calendar alone is insufficient. Frequency, timestamp side, anchor and partial-bar policy
must be declared, as in the example; contradictory or missing declarations fail clearly. Multi-asset
datasets are compared separately by the declared `entity_col`, or `symbol` when present; their gap
ranges include an `entity` value. The stored entity column is inspected across the dataset so an
instrument absent throughout the requested interval is not overlooked. Instruments that have never
been stored cannot be inferred: inspect an explicit absent dataset key and supply its declaration.
For CME futures select the product's actual calendar, such as `CME_Equity`, rather than assuming
every CME product uses the crypto schedule. CME cryptocurrency futures use `CME Globex Crypto`;
spot crypto normally uses `24/7`. See [CME and spot session conventions](sessions.md#cme-crypto-and-spot-crypto).

A missing scheduled observation is not proof of a provider outage. Databento OHLCV and Coin Metrics
market candles can omit no-trade intervals. Keep these absences in the report; a source returning no
row does not justify creating a replacement candle. Targeted source downloads are a separate step,
not an effect of inspecting the dataset. Values inside present rows are checked by validation, not
by this timestamp-completeness report.

### Download Real Observations for a Selected Report

This is an explicit, potentially billable operation. For an existing one-instrument CME minute
dataset, first inspect the desired interval. The storage key and vendor symbol are separate:

```python
from ml4t.data.providers import DataBentoProvider
from ml4t.data.storage.metadata_tracker import MetadataTracker
from ml4t.data.update_manager import BackfillManager

cme_feed = feed.with_overrides(calendar="CME Globex Crypto")
key = "futures/minute/BTC"
gaps = GapDetector(feed_spec=cme_feed).detect_gaps_in_storage(
    storage,
    key,
    datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
    datetime(2026, 9, 5, 12, 2, tzinfo=UTC),
)
with DataBentoProvider(stype_in="continuous") as provider:
    result = BackfillManager(storage, MetadataTracker(storage.base_path)).execute_backfill(
        key, gaps, provider, source_symbol="BTC.c.0", feed_spec=cme_feed
    )
print(result.rows_added, result.remaining_gaps)
```

The consumer rechecks the selected ranges against disk before requesting data. It downloads only
currently missing ranges, preserves real stored observations and identity/calendar metadata, and replaces a marked
null placeholder only when an actual source observation is returned. Intraday requests use the
optional `TimestampRangeProvider` capability; unsupported providers fail without falling back to a
full-day download. `calendar=None` disables calendar-based recovery. The first supported adapter is
Databento, with a left-labeled grid and one instrument per storage key. Multi-asset inspection is
supported, but targeted recovery of a panel requires separate per-instrument datasets.

`result.success` means processing completed without errors, not that every observation was recovered.
An empty input report succeeds without downloading or writing; use `rows_added` and `gaps_filled`
to distinguish recovered data from a no-op. Previously an empty report returned `success=False`.
Inspect `errors` and `remaining_gaps`. Empty no-trade responses do not change storage or trigger
unbounded retries. Partial responses retain their residual gaps; `gaps_filled` counts only completely
recovered ranges. A history-recording failure reports an error even if observations were stored;
the residual report still reflects disk. A failed operation that cannot establish coverage returns
`remaining_gaps=None`, not an unverified clean report. Existing observed null values are a separate
validation concern, not permission to overwrite a source observation. `identify_candidates(min_gap_bars=...)` uses bar counts for
intraday data; its daily `min_gap_days` remains a session count.

Coin Metrics is not currently a Data provider. Its candle and metric endpoints use different
identities and schemas; this Databento workflow does not claim to download them. See the
[Databento range contract](../providers/databento.md#exact-timestamp-ranges) before selecting a
continuous symbol or price convention.

### Daily Range Compatibility

Daily plans include missing sessions at either boundary and within the inclusive requested date
range. Closed dates never become missing sessions. Returned `start` and `end` are inclusive session
labels encoded at UTC midnight, not event instants. The legacy `size_days` field and `tolerance_days`
argument count expected sessions in daily mode, not elapsed civil days. A range can span closures
between missing sessions. Backfill selects only scheduled labels from the provider's actual rows;
it does not manufacture observations or insert closed-date bars.

The legacy `exclude_weekends=False` argument selects continuous days only when no canonical calendar
or feed declaration is present. `exclude_weekends=True` now follows NYSE holidays as well as weekends.
Omission follows NYSE, so callers requiring the old elapsed-day counts should pass `calendar="24/7"`.
The legacy undeclared weekly path is elapsed-time based, not an exchange-grid completeness report;
declared weekly grids are not supported. `calendar=None` disables classification.
Session-label dates use their declared timezone, or their existing timezone when
undeclared. Explicit event-time/bar-close/bar-side declarations map to authoritative session labels
before planning, including for provider-row selection during backfill. Out-of-session instants fail
clearly; original provider timestamps are retained in storage. Declare the feed's `timezone` when a
non-UTC label must survive storage's canonical UTC reads.

If a provider's history limit cannot reach the stored dataset, the updater records
`provider_history_limited`. Missing observations remain detectable even when they cannot be retrieved.

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
    check_gaps=True,
    initial_start="2023-01-01",
    initial_end="2024-01-01",
)
```

`DataManager.update()` returns the storage key. It raises when storage is not configured, stored data
is empty, metadata is invalid, or the provider or storage operation fails.

`load()`, `import_data()` and `update()` accept `calendar` and a shared `feed_spec`. New ingestion
without a feed writes version-2 temporal metadata: an omitted calendar remains omitted, while
explicit `None` remains disabled after storage and model projection. An existing version-1 feed
is not automatically reinterpreted; its null calendar still means inheritance. An explicit calendar
override writes version 2. Calendar names and conflicting frequency/bar-type declarations fail
before a replacement generation is published.

Feed timezone, timestamp semantics and bar-grid declarations survive ingestion. A non-null
`session_start_time` override is not supported by the native schedule adapter and is rejected.
The requested frequency and bar type must agree with declared feed metadata. These declarations
do not themselves validate OHLCV completeness or transform timestamps. Gap detection remains
separate from acquisition; neither validation nor gap checking imputes missing market values.

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
- Preserve missing market data. Optional minute-grid expansion marks null placeholder rows with
  the legacy `is_imputed` column; those rows are not recovered source observations.
- Use `ml4t-data validate` after an update when schema, OHLC relationships, duplicates, or anomaly
  checks are part of the ingestion contract.
