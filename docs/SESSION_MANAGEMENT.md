# Session Management in ML4T Data

Complete guide to assigning and using session dates for futures and intraday data.

## Table of Contents

1. [Overview](#overview)
2. [Why Session Dates Matter](#why-session-dates-matter)
3. [Use Cases](#use-cases)
4. [Supported Exchanges](#supported-exchanges)
5. [Quick Start](#quick-start)
6. [Working with ML4T Data Storage](#working-with-ml4t-data-storage)
7. [Standalone Session Assignment](#standalone-session-assignment)
8. [Cross-Validation with Sessions](#cross-validation-with-sessions)
9. [Complete Examples](#complete-examples)

---

## Overview

Session management in ml4t-data provides tools for:
- **Assigning session dates** to intraday data based on exchange calendars
- **Completing sessions** by filling gaps in minute-level data
- **Cross-validation** that respects session boundaries (prevents data leakage)

### What is a Session Date?

A **session date** is the calendar date assigned to each trading bar based on the exchange's trading hours.

**Key characteristics**:
- For CME futures: Session starts Sunday 5pm CT, ends Friday 4pm CT (23 hours/day)
- Session date = **date when the session ENDS** (e.g., Monday 4pm CT → session_date = Monday)
- Overnight bars (e.g., Monday 11pm CT) still belong to Monday's session
- Critical for time-series cross-validation to avoid data leakage

---

## Why Session Dates Matter

### Problem: Naive Date-Based Splits Cause Data Leakage

```python
# ❌ WRONG: Using calendar date creates leakage
df_train = df.filter(pl.col("timestamp").dt.date() < "2024-06-01")
df_test = df.filter(pl.col("timestamp").dt.date() >= "2024-06-01")

# Problem: Monday's session starts Sunday 5pm
# → Sunday evening bars (5pm-11:59pm) are in df_test
# → But Monday morning bars (12am-4pm) are in df_train
# → Same trading session is split across train/test!
```

### Solution: Session-Based Splits Prevent Leakage

```python
# ✅ CORRECT: Using session_date keeps sessions intact
df_train = df.filter(pl.col("session_date") < "2024-06-01")
df_test = df.filter(pl.col("session_date") >= "2024-06-01")

# Result: All bars from same trading session stay together
# → Sunday 5pm → Monday 4pm (all marked as session_date=Monday)
# → No overlap between train and test sessions
```

---

## Use Cases

### 1. Time-Series Cross-Validation

Use session dates with `GroupKFold` to prevent data leakage:

```python
from sklearn.model_selection import GroupKFold

# Each session stays in one fold
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df["session_date"]):
    # Train and test contain different sessions (no overlap)
    pass
```

### 2. Session-Level Analysis

Aggregate metrics by trading session:

```python
session_stats = df.group_by("session_date").agg([
    pl.col("volume").sum().alias("daily_volume"),
    pl.col("close").last().alias("session_close"),
    (pl.col("close").last() - pl.col("close").first()).alias("session_return")
])
```

### 3. Gap Filling for Complete Sessions

CME futures have 1-hour daily maintenance breaks (4pm-5pm CT). Fill gaps:

```python
df_complete = manager.complete_sessions(
    df,
    exchange="CME",
    fill_gaps=True,      # Fill missing minutes
    zero_volume=True     # Mark filled bars with volume=0
)
```

### 4. Feature Engineering

Build features that respect session boundaries:

```python
# Rolling features within sessions (don't cross session boundaries)
df = df.with_columns([
    pl.col("close")
      .rolling_mean(window_size=60)
      .over("session_date")
      .alias("close_ma60_session")
])
```

---

## Supported Exchanges

ML4T Data uses [pandas_market_calendars](https://github.com/rsheftel/pandas_market_calendars) for exchange calendars:

| Exchange | Code | Calendar Name | Trading Hours (Local Time) |
|----------|------|---------------|----------------------------|
| CME (Globex Crypto) | CME | CME_Globex_Crypto | Sun 5pm - Fri 4pm CT (23h/day) |
| CME (Equity) | CME | CME_Equity | Sun 5pm - Fri 4pm CT (23h/day) |
| NASDAQ | NASDAQ | NASDAQ | 9:30am - 4:00pm ET |
| NYSE | NYSE | NYSE | 9:30am - 4:00pm ET |
| LSE | LSE | LSE | 8:00am - 4:30pm GMT |
| Tokyo | TSE | TSE | 9:00am - 3:00pm JST |
| Hong Kong | HKEX | HKEX | 9:30am - 4:00pm HKT |

**Note**: CME futures have two calendar types:
- `CME_Globex_Crypto`: For BTC/ETH futures
- `CME_Equity`: For equity index futures (ES, NQ, etc.)

---

## Quick Start

### Method 1: Using DataManager (Recommended for ml4t-data storage)

```python
from ml4t-data import DataManager
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage

# Initialize with your storage
storage = HiveStorage(config=StorageConfig(base_path="./data"))
manager = DataManager(storage=storage)

# Load data
df = storage.read("crypto_futures_1min_BTC").collect()

# Assign sessions
df_with_sessions = manager.assign_sessions(df, exchange="CME")

# Complete sessions (fill gaps)
df_complete = manager.complete_sessions(
    df_with_sessions,
    exchange="CME",
    fill_gaps=True,
    zero_volume=True
)
```

### Method 2: Using SessionAssigner Directly

```python
from ml4t.data.sessions import SessionAssigner
import polars as pl

# Load your data
df = pl.read_parquet("my_data.parquet")

# Assign sessions
assigner = SessionAssigner.from_exchange("CME")
df_with_sessions = assigner.assign_sessions(df)
```

---

## Working with ML4T Data Storage

### Example: Wyden CME Crypto Futures

Complete workflow for production data:

```python
from pathlib import Path
import polars as pl
from ml4t-data import DataManager
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage

# Wyden data location
DATA_PATH = Path.home() / "clients/wyden/long-short/data"

# Initialize
storage = HiveStorage(config=StorageConfig(base_path=str(DATA_PATH)))
manager = DataManager(storage=storage)

# Read BTC futures
df = storage.read("crypto_futures_1min_BTC").collect()
print(f"Loaded {len(df):,} rows")  # 2.79M rows

# Assign CME sessions
df_with_sessions = manager.assign_sessions(df, exchange="CME")
print(f"Assigned {df_with_sessions['session_date'].n_unique():,} sessions")

# Complete sessions
df_complete = manager.complete_sessions(
    df_with_sessions,
    exchange="CME",
    fill_gaps=True,
    zero_volume=True
)
print(f"Filled {len(df_complete) - len(df):,} gaps")

# Save
df_complete.write_parquet("btc_futures_with_sessions.parquet")
```

**Full script**: `examples/wyden_cme_sessions_complete_workflow.py`

---

## Standalone Session Assignment

For data NOT stored in ml4t-data (external parquet/CSV files):

### Command-Line Tool

```bash
# Assign sessions to any parquet file
python examples/assign_sessions_standalone.py \
    --input nq_bars.parquet \
    --output nq_bars_with_sessions.parquet \
    --exchange CME \
    --timestamp-column datetime

# Overwrite input file
python examples/assign_sessions_standalone.py \
    --input data.parquet \
    --exchange NASDAQ \
    --inplace
```

### Python API

```python
from examples.assign_sessions_standalone import assign_sessions_to_file

df = assign_sessions_to_file(
    input_path="nq_bars.parquet",
    output_path="nq_bars_with_sessions.parquet",
    exchange="CME",
    timestamp_column="datetime"
)

print(df.select(["datetime", "session_date"]).head())
```

---

## Cross-Validation with Sessions

### GroupKFold (Recommended)

**Why**: Ensures each session appears in exactly one fold.

```python
from sklearn.model_selection import GroupKFold
import numpy as np

# Load data with sessions
df = pl.read_parquet("data_with_sessions.parquet")

# Prepare data
X = df.select([...feature_columns...]).to_numpy()
y = df["target"].to_numpy()
groups = df["session_date"].to_numpy()

# Create folds
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    # Get session info
    train_sessions = np.unique(groups[train_idx])
    test_sessions = np.unique(groups[test_idx])

    # Verify no overlap
    assert len(set(train_sessions) & set(test_sessions)) == 0

    print(f"Fold {fold + 1}:")
    print(f"  Train: {len(train_sessions)} sessions, {len(train_idx):,} bars")
    print(f"  Test: {len(test_sessions)} sessions, {len(test_idx):,} bars")
```

### Time-Series Split with Sessions

```python
from sklearn.model_selection import TimeSeriesSplit

# Sort by session date
df = df.sort("session_date")

# Get unique sessions
unique_sessions = df["session_date"].unique().sort()

# Split sessions (not individual bars)
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_sessions_idx, test_sessions_idx) in enumerate(tscv.split(unique_sessions)):
    train_sessions = unique_sessions[train_sessions_idx]
    test_sessions = unique_sessions[test_sessions_idx]

    # Filter data by sessions
    train_data = df.filter(pl.col("session_date").is_in(train_sessions))
    test_data = df.filter(pl.col("session_date").is_in(test_sessions))

    print(f"Fold {fold + 1}:")
    print(f"  Train: {len(train_sessions)} sessions, {len(train_data):,} bars")
    print(f"  Test: {len(test_sessions)} sessions, {len(test_data):,} bars")
```

---

## Complete Examples

### 1. CME Crypto Futures (Time Bars)

**File**: `examples/wyden_cme_sessions_complete_workflow.py`

```bash
python examples/wyden_cme_sessions_complete_workflow.py
```

Demonstrates:
- Loading data from ml4t-data storage
- Assigning CME sessions
- Completing sessions (filling gaps)
- Session statistics
- Cross-validation setup

### 2. NASDAQ 100 Futures (Trade/Volume Bars)

**File**: `examples/nasdaq_bars_sessions.py`

```bash
python examples/nasdaq_bars_sessions.py
```

Demonstrates:
- Working with bar data (not time-based)
- Irregular timestamps
- Session assignment for external data
- Cross-validation with bar data

### 3. Standalone CLI Tool

**File**: `examples/assign_sessions_standalone.py`

```bash
# Help
python examples/assign_sessions_standalone.py --help

# Assign sessions
python examples/assign_sessions_standalone.py \
    --input ~/clients/chimera/bias_strategy/data/processed/nqu25_processed.parquet \
    --output ~/clients/chimera/bias_strategy/data/processed/nqu25_with_sessions.parquet \
    --exchange CME \
    --timestamp-column datetime
```

---

## API Reference

### DataManager Methods

#### `assign_sessions()`

```python
def assign_sessions(
    self,
    df: pl.DataFrame,
    exchange: str | None = None,
    calendar: str | None = None,
) -> pl.DataFrame:
    """Assign session_date column to DataFrame.

    Args:
        df: DataFrame with timestamp column
        exchange: Exchange code (CME, NYSE, NASDAQ, etc.)
        calendar: Calendar name override (e.g., "CME_Globex_Crypto")

    Returns:
        DataFrame with session_date column added
    """
```

#### `complete_sessions()`

```python
def complete_sessions(
    self,
    df: pl.DataFrame,
    exchange: str | None = None,
    fill_gaps: bool = True,
    zero_volume: bool = True,
) -> pl.DataFrame:
    """Complete trading sessions by filling gaps.

    Args:
        df: DataFrame with timestamp and session_date columns
        exchange: Exchange code (CME, NYSE, etc.)
        fill_gaps: Fill missing minutes within sessions
        zero_volume: Set volume=0 for filled bars

    Returns:
        DataFrame with complete sessions
    """
```

### SessionAssigner Class

```python
from ml4t.data.sessions import SessionAssigner

# From exchange code
assigner = SessionAssigner.from_exchange("CME")

# From calendar name
assigner = SessionAssigner("CME_Globex_Crypto")

# Assign sessions
df_with_sessions = assigner.assign_sessions(
    df,
    start_date="2024-01-01",  # Optional
    end_date="2024-12-31"      # Optional
)
```

---

## Troubleshooting

### Missing pandas_market_calendars

**Error**: `ModuleNotFoundError: No module named 'pandas_market_calendars'`

**Solution**:
```bash
pip install pandas-market-calendars
```

### Wrong Calendar for CME Futures

**Problem**: Using wrong calendar results in incorrect sessions.

**Solution**: Use the right calendar:
- Crypto futures (BTC, ETH): `CME_Globex_Crypto`
- Equity futures (ES, NQ): `CME_Equity`

```python
# Correct for BTC/ETH
assigner = SessionAssigner("CME_Globex_Crypto")

# Correct for NQ
assigner = SessionAssigner("CME_Equity")
```

### Timestamp Column Not Found

**Error**: `ValueError: DataFrame must have 'timestamp' column`

**Solution**: Rename your timestamp column:
```python
df = df.rename({"datetime": "timestamp"})
df_with_sessions = assigner.assign_sessions(df)
df_with_sessions = df_with_sessions.rename({"timestamp": "datetime"})
```

### Session Dates are NULL

**Problem**: All session_date values are NULL.

**Possible causes**:
1. Data outside calendar date range
2. Timestamps are not timezone-aware
3. Wrong calendar for exchange

**Solution**: Check your data:
```python
print(df["timestamp"].min(), df["timestamp"].max())
print(df["timestamp"].dtype)
```

---

## Additional Resources

- **pandas_market_calendars**: https://github.com/rsheftel/pandas_market_calendars
- **CME Trading Hours**: https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html
- **Cross-Validation Best Practices**: See ml4t-data docs on time-series CV

---

## Summary

**Key Takeaways**:
1. ✅ Use session dates for cross-validation to prevent data leakage
2. ✅ Session date = date when session ENDS (e.g., Monday 4pm CT → Monday)
3. ✅ Complete sessions by filling gaps (CME has 1-hour maintenance breaks)
4. ✅ Use GroupKFold with session_date as groups
5. ✅ Works with any data: time bars, volume bars, trade bars

**Next Steps**:
1. Run example scripts to understand workflows
2. Assign sessions to your data
3. Implement proper cross-validation with session groups
4. Build session-aware features for better models
