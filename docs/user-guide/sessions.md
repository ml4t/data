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
include the close. The corresponding boundaries apply to lunch breaks. An explicit `bar_label`
also selects the declared side; `event_time` alone does not infer a bar side. This assignment does
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

## Complete Minute Sessions

`SessionCompleter` expands minute bars to the exchange schedule. It requires `timestamp`, `open`,
`high`, `low`, `close`, and `volume`. The result adds `session_date` and `is_imputed`.

```python
from ml4t.data.sessions import SessionCompleter

completer = SessionCompleter("NYSE")
complete = completer.complete_sessions(
    bars,
    fill_method="forward",
    zero_volume=True,
)
```

The supported fill methods are `"forward"`, `"backward"`, and `"none"`. Forward and backward
filling apply only within a session. `zero_volume=True` sets volume to zero for inserted rows.

Session completion rejects duplicate or non-minute-aligned timestamps and observations outside the
selected schedule. Use session assignment first when you need to inspect how individual timestamps
map to the exchange calendar.

Completion remains a one-minute, open-labeled operation, not a general bar-grid validator. Explicit
close-labeled or session-label metadata is rejected. `calendar=None` returns the input unchanged.
Declared event-count bar types and frequencies other than one minute are also rejected.
`DataManager.complete_sessions(fill_gaps=False)` performs assignment only and never expands rows,
whether or not storage is configured. Feed declarations are accepted through the facade's
`feed_spec` keyword with the same precedence as the low-level classes.

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
