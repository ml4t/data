# Exporting Stored Data

The command-line interface exports one stored dataset to CSV, JSON, or Parquet. The Python export
manager adds Excel output, batch export, pattern selection, and transformations.

## Command-Line Export

The CLI derives the storage key from the asset class, frequency, and symbol:

```bash
ml4t-data export \
  --symbol AAPL \
  --asset-class equities \
  --frequency daily \
  --storage-path ./data \
  --output ./exports/aapl.csv \
  --format csv
```

Run `ml4t-data export --help` for the accepted values. Create the output directory before running
the command.

## Python Export Manager

`ExportManager` accepts any storage backend that implements `exists()`, `read()`, and `list_keys()`.

```python
from pathlib import Path

from ml4t.data.export.manager import ExportManager
from ml4t.data.storage import create_storage

storage = create_storage("./data", strategy="hive")
manager = ExportManager(storage)
Path("./exports").mkdir(exist_ok=True)

result = manager.export(
    key="equities/daily/AAPL",
    output_path="./exports/aapl.csv",
    format_type="csv",
)
if not result.success:
    raise RuntimeError(result.error)
```

`format_type` accepts `csv`, `json`, `excel`, or `xlsx`. The returned `ExportResult` records the
output path, row count, file size, duration, and any error.

## Filter and Transform

Export options are validated by `ExportConfig`:

```python
result = manager.export(
    key="equities/daily/AAPL",
    output_path="./exports/aapl.csv.gz",
    format_type="csv",
    date_filter=("2024-01-01", "2024-03-31"),
    columns=["timestamp", "close", "volume"],
    add_returns=True,
    add_volatility=True,
    compression="gzip",
)
```

Date and column filters are passed to production storage before collection when the backend accepts
them. Requested columns retain their supplied order, and a missing column fails the export.

CSV supports optional gzip compression. JSON can include export metadata. Excel supports one sheet
per dataset and requires the dependencies installed with ML4T Data.

## Batch and Pattern Export

Export explicit keys to one Excel workbook:

```python
results = manager.export_batch(
    keys=["equities/daily/AAPL", "equities/daily/MSFT"],
    output_path="./exports/equities.xlsx",
    format_type="excel",
)
```

Or select complete storage keys with shell-style patterns:

```python
results = manager.export_pattern(
    pattern="equities/daily/*",
    output_path="./exports/",
    format_type="csv",
)
```

Patterns support `*`, `?`, and bracket expressions. Two keys that resolve to the same export symbol
produce an explicit failure rather than overwriting one another.

## Large Datasets

Use `date_filter` and `columns` to reduce the data before it is collected. Export transformations
and file serialization operate in memory, including gzip CSV output.
