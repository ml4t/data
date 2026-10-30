# Trading Sessions

Use session dates when an exchange trading day does not match a calendar day. A session date keeps
all observations from one exchange session together for aggregation, validation, and train/test
splits.

ML4T Data uses
[`pandas_market_calendars`](https://github.com/rsheftel/pandas_market_calendars) for schedules. The
dependency is included with the package.

## Assign Session Dates

`SessionAssigner` accepts Polars `Datetime` instants or `Date` session labels in `timestamp`.
Naive instants use UTC unless the feed declares a timezone; timezone-aware instants are matched in
UTC. Ambiguous or nonexistent local times fail rather than being guessed.

```python
from datetime import datetime

import polars as pl

from ml4t.data.sessions import SessionAssigner

bars = pl.DataFrame(
    {
        "timestamp": [
            datetime(2024, 1, 2, 15, 0),
            datetime(2024, 1, 2, 16, 0),
            datetime(2024, 1, 3, 15, 0),
            datetime(2024, 1, 3, 16, 0),
        ],
        "open": [100.0, 101.0, 102.0, 103.0],
        "high": [101.0, 102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0, 102.0],
        "close": [100.0, 101.0, 102.0, 103.0],
        "volume": [10.0, 20.0, 30.0, 40.0],
    }
)

assigner = SessionAssigner.from_exchange("NYSE")
assigned = assigner.assign_sessions(
    bars,
    bar_frequency="intraday",
    outside_session="raise",
)
```

The result contains a `session_date` column. `outside_session` controls observations that do not
fall within an open interval:

- `"null"` retains the row with a null session date.
- `"raise"` rejects the input and reports sample timestamps.
- `"drop"` removes the row.

Set `bar_frequency="daily"` when timestamps are daily labels. The default, `"auto"`, treats a
series as daily when every non-null timestamp is midnight.

Explicit `FeedSpec.timestamp_semantics` overrides these heuristics: `session_label` preserves the
civil label date, `event_time` matches an instant, and `bar_close` matches the ending bar boundary.
Event intervals include the open and exclude the close; close-labeled bars exclude the open and
include the close. The corresponding boundaries apply to lunch breaks and declared calendar
interruptions. An explicit `bar_label` also selects the declared side; `event_time` alone does not
infer a bar side. This assignment does
not validate intraday grid completeness.

`session_start_time` is retained in metadata, but these schedule adapters reject a non-null override.
Register and select an authoritative calendar with the required intervals instead of silently
replacing a named exchange's session boundaries.

```python
from ml4t.specs import FeedSpec

feed = FeedSpec(calendar="24/7", timestamp_semantics="event_time", time_contract_version=2)
assigned = SessionAssigner(feed_spec=feed).assign_sessions(bars)
unchanged = SessionAssigner(calendar=None).assign_sessions(bars)
```

Use the canonical `calendar` keyword on constructors and facade methods. Explicit selection wins
over feed metadata, then NYSE applies. Explicit `None` returns observations unchanged, without
adding a session column or reconstructing a continuous grid. `calendar="24/7"` remains a real
schedule, including an event exactly at midnight in the new day's session. Unknown names fail.
The positional calendar argument, `calendar_name` constructor keyword and explicit `exchange`
selector remain compatibility forms; contradictory explicit selectors raise an error. Descriptive
exchange identity in feed metadata does not select a calendar.

Data's `Metadata` supports the same versioned temporal declaration. Version 1 retains legacy null
inheritance; version 2 preserves omitted calendar versus explicit null through JSON, copying and
shared FeedSpec projection. Use `model_dump(mode="json")` before a YAML writer, and ordinary
`model_dump_json()` for JSON; do not deliberately exclude null fields when preserving a disable
declaration. Declared timezone, timestamp semantics and fixed-grid fields survive these bridges.

## Select an Exchange Calendar

`SessionAssigner.from_exchange()` recognizes these exchange codes:

| Code | Calendar |
| --- | --- |
| `CME` | CME Globex Crypto |
| `NYSE` | NYSE |
| `NASDAQ` | NASDAQ |
| `LSE` | LSE |
| `TSE` | TSE |
| `HKEX` | HKEX |
| `ASX` | ASX |
| `SSE` | SSE |
| `TSX` | TSX |

For another calendar exposed by `pandas_market_calendars`, pass its name directly:

```python
assigner = SessionAssigner("CME Globex Crypto")
```

Calendar names and schedules can change upstream. Check the installed calendar before assigning a
large dataset.

### CME Crypto and Spot Crypto

`CME` selects CME Globex cryptocurrency futures and options, not all CME products and not
spot-quoted futures. Data preserves the upstream historical schedule through May 29, 2026. After
that day's 16:00-16:30 Chicago maintenance, trading includes weekends and holidays. Regular Globex
maintenance is Monday-Friday 16:00-16:02 Chicago time and Saturday 02:00-04:00. Pre-open periods
are not trading intervals. See the [CME crypto FAQ](https://www.cmegroup.com/articles/faqs/frequently-asked-questions-cryptocurrency-futures.html)
and [launch notice](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/1283194884/Cryptocurrency+Futures+and+Options+Migration+to+24-7+Trading).

CME observations retain the following business trade date across weekends and holidays, rather
than acquiring a separate Saturday or Sunday label. Data uses the upstream `CME_TradeDate`
business calendar, qualified against CME's
[2026 reporting calendar](https://www.cmegroup.com/content/dam/cmegroup/notices/market-regulation/2026/01/msn01-09-2026revisedholidayprocessinglargetrader2026.pdf).
Schedule queries include the next business date when needed; a Saturday observation can therefore
map to Monday, or Tuesday after Labor Day. Assignment and completion share these intervals and
exclude maintenance. The adapter does not change the upstream calendar registry or other products.

Ordinary spot crypto uses `calendar="24/7"`, with civil-day labels and no CME maintenance. Explicit
`calendar=None` disables session handling instead. Select the schedule from the feed's actual
market, not from its symbol or asset-class name.

<a id="complete-minute-sessions"></a>

## Inspect a Nullable Minute Grid

`SessionCompleter` optionally expands a timestamped series to scheduled minutes for inspection.
Missing positions have null market values; they are not recovered observations. It requires a
Datetime `timestamp` column and retains supplied value columns. The result adds `session_date` and
the legacy `is_imputed` marker, which identifies null placeholder rows.

```python
from ml4t.data.sessions import SessionCompleter

completer = SessionCompleter("NYSE")
complete = completer.complete_sessions(
    bars,
    fill_method="none",
    zero_volume=False,
)
```

Only `fill_method="none"` and `zero_volume=False` are supported. Requests to copy prices or invent
zero volume fail clearly. Observed nulls and all other original values remain unchanged.
Use the [disk gap report](incremental-updates.md#inspect-missing-observations-on-disk) when only a list
of missing observations is needed; it does not expand the data.

Session completion rejects duplicate or non-minute-aligned timestamps and observations outside the
selected schedule. Lunch breaks and declared calendar interruptions are excluded from the template;
completion never inserts rows for those closures. Use session assignment first to inspect how
timestamps map to the exchange calendar.

Completion remains a one-minute, open-labeled operation, not a general bar-grid validator. Explicit
close-labeled or session-label metadata is rejected. `calendar=None` returns the input unchanged.
Declared event-count bar types and frequencies other than one minute are also rejected.
General bar-grid anchor/partial-bar policies are preserved as metadata but explicitly rejected by
this native minute completer; they are not silently approximated by its legacy minute template.
`DataManager.complete_sessions(fill_gaps=False)` performs assignment only and never expands rows,
whether or not storage is configured. Feed declarations are accepted through the facade's
`feed_spec` keyword with the same precedence as the low-level classes.

## Run the NASDAQ Futures Bar Example

From a repository checkout with the package installed, run:

```bash
uv run python examples/nasdaq_bars_sessions.py
```

The command reads `examples/data/nasdaq_bars_sample.csv`, resolved relative to the script, and
writes `nasdaq_bars_with_sessions.parquet` in the working directory. It needs no credentials,
network access, or client files. The CSV is hand-authored teaching data distributed under this
repository's MIT license. Its prices, volumes, and trade counts are illustrative, with no vendor
or real contract identity. It is not a substitute for historical NQU25 observations.

The sample contains nine irregular, close-labeled bars across the March 7 and March 10, 2025
`CME_Equity` sessions, including Sunday evening trading after the US daylight-saving change.
All `datetime` values explicitly use UTC (`Z`). Friday's 16:00 Chicago close is 22:00 UTC;
Monday's is 21:00 UTC. A close-labeled observation at the session close or break start belongs to
the preceding open interval. The sample includes those boundaries and observations after the
scheduled break. The selected calendar supplies the actual intervals; a fixed UTC offset or
weekday rule does not determine the session date.

These rows illustrate an already aggregated input. They are not generated from ticks, and no
volume or trade-count threshold, partial-bar policy, or cleaning pipeline is claimed. The example
only adds session dates: it does not aggregate, complete a grid, fill values, or drop observations.
It rejects observations outside the selected schedule. Irregular bar timestamps alone do not
establish that a dataset has no missing bars.

### Supply Your Own Bars

An external Parquet file must contain `datetime` as a non-null Polars Datetime, plus `Open`, `High`,
`Low`, `Last`, `Volume`, and `# of Trades`. Preserve the export's timezone, aggregation threshold,
contract identifier, date range, session-boundary rules, partial-bar policy, and cleaning provenance
alongside the file. Prepare these fields using your licensed source's documented export or
aggregation process; this example does not reconstruct a vendor-specific preprocessing pipeline.

```bash
uv run python examples/nasdaq_bars_sessions.py --input bars.parquet --output assigned.parquet --timezone America/Chicago --bar-label close
```

Naive external timestamps require `--timezone`; use the source timezone rather than guessing.
Aware timestamps retain their instants and need no timezone option. Confirm the source's label
convention: `--bar-label close` is the default and uses `(open, close]` intervals, while
`--bar-label open` uses `[open, close)`. The same convention applies to breaks. The example selects
`CME_Equity` explicitly, not the CME crypto calendar. Use another authoritative product calendar
when the input is from a different market.

The earlier example referenced a private `nqu25_processed.parquet` input. Its original vendor,
contract/date range, timestamp convention, aggregation threshold, filtering, and partial-bar
handling have not been established from public repository history. Neither that input nor its
client preprocessing pipeline is distributed here. The bundled CSV is the runnable alternative;
it does not assert or reproduce the original pipeline's parameters.

## Use Session Dates in Validation

With scikit-learn installed, group observations by `session_date` when a split must not divide one
trading session. The example above contains two sessions, one for each fold:

```python
from sklearn.model_selection import GroupKFold

groups = assigned["session_date"].to_numpy()
features = assigned.select("close").to_numpy()
target = assigned["close"].to_numpy()

for train_index, test_index in GroupKFold(n_splits=2).split(features, target, groups):
    train = assigned[train_index]
    test = assigned[test_index]
```

This groups rows by session. It does not make `GroupKFold` chronological. Use a time-ordered split
over unique session dates when the evaluation must also preserve time order.

## Related APIs

`DataManager.assign_sessions()` and `DataManager.complete_sessions()` expose the same operations
through the main facade. See the [API reference](../api/index.md) for their signatures.
