# Trading Sessions

Use session dates when an exchange trading day does not match a calendar day. A session date keeps
all observations from one exchange session together for aggregation, validation, and train/test
splits.

ML4T Data uses
[`pandas_market_calendars`](https://github.com/rsheftel/pandas_market_calendars) for schedules. The
dependency is included with the package.

## Assign Session Dates

`SessionAssigner` requires a Polars `Datetime` column named `timestamp`. Naive timestamps are
interpreted as UTC; timezone-aware timestamps are converted to UTC before they are matched to a
schedule.

```python
from datetime import datetime

import polars as pl

from ml4t.data.sessions import SessionAssigner

bars = pl.DataFrame(
    {
        "timestamp": [
            datetime(2024, 1, 2, 15, 0),
            datetime(2024, 1, 2, 16, 0),
        ],
        "close": [100.0, 101.0],
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

## Use Session Dates in Validation

Group observations by `session_date` when a split must not divide one trading session:

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
