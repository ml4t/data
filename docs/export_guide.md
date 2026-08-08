# Data Export Guide

ML4T Data provides flexible data export functionality to convert stored market data into various formats for analysis and integration with other tools.

## Supported Formats

- **CSV** - Comma-separated values for spreadsheet applications
- **JSON** - JavaScript Object Notation for web applications and APIs
- **Excel** - Microsoft Excel format with multiple sheet support

## CLI Usage

### Basic Export

Export a single dataset to CSV:

```bash
ml4t-data export \
  --symbol equities/daily/AAPL \
  --output ./exports/aapl.csv \
  --format csv \
  --storage-path ./data
```

The CLI supports CSV, JSON, and Parquet for one storage key. Use the Python API for
compression, transformations, batch export, patterns, and Excel workbooks.

## Python API

### Basic Export

```python
from pathlib import Path

from ml4t.data.export.manager import ExportManager
from ml4t.data.storage import create_storage

# Initialize
storage = create_storage("./data", strategy="hive")
manager = ExportManager(storage=storage)
Path("./exports").mkdir(exist_ok=True)

# Export single dataset
result = manager.export(
    key="equities/daily/AAPL",
    output_path="./exports/AAPL.csv",
    format_type="csv"
)

if result.success:
    print(f"Exported to: {result.output_path}")
    print(f"Rows: {result.rows_exported}")
```

### Batch Export

```python
# Export multiple datasets to Excel
results = manager.export_batch(
    keys=["equities/daily/AAPL", "equities/daily/GOOGL"],
    output_path="./report.xlsx",
    format_type="excel"
)
```

### Export with Transformations

```python
# Export with filters and calculations
result = manager.export(
    key="crypto/daily/BTC",
    output_path="./btc_analysis.csv",
    format_type="csv",
    date_filter=("2024-01-01", "2024-03-31"),
    columns=["timestamp", "close", "volume"],
    add_returns=True,
    add_volatility=True
)
```

### Pattern-Based Export

```python
# Export all matching datasets
results = manager.export_pattern(
    pattern="equities/daily/*",
    output_path="./exports/",
    format_type="csv"
)

print(f"Exported {len(results)} datasets")
```

Requested columns are written in the supplied order. An export fails explicitly if any
requested column is absent. Pattern matching uses shell-style `*`, `?`, and bracket
expressions against complete storage keys.

## Export Formats Details

### CSV Format

- Human-readable text format
- Compatible with Excel, Google Sheets, pandas
- Supports compression (gzip)
- One file per dataset

Example output:
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00,100.0,105.0,99.0,104.0,1000000
2024-01-02T00:00:00,104.0,106.0,103.0,105.0,1100000
```

### JSON Format

- Structured data format
- Ideal for web applications
- Supports metadata inclusion
- Can combine multiple datasets

Example output:
```json
{
  "symbol": "AAPL",
  "metadata": {
    "exported_at": "2024-03-15T10:30:00",
    "rows": 252,
    "date_range": {
      "start": "2024-01-01T00:00:00",
      "end": "2024-12-31T00:00:00"
    }
  },
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "open": 100.0,
      "high": 105.0,
      "low": 99.0,
      "close": 104.0,
      "volume": 1000000
    }
  ]
}
```

### Excel Format

- Native Excel format (.xlsx)
- Multiple sheets support
- Automatic formatting
- Metadata sheet included

Features:
- Each dataset in separate sheet
- Sheet names from symbol names
- Auto-fit columns
- Number formatting

## Performance Considerations

### Large Datasets

For datasets with millions of rows:

1. Filter by date so storage reads only matching partitions.
2. Select columns so storage projects only required fields.
3. Use gzip compression when the smaller output justifies its additional memory use.

### Memory Usage

Date and column filters are applied by production storage before collection. The filtered
result is then held in memory while it is transformed and written. Gzip CSV export also
creates the CSV payload in memory, so narrow the date range and columns before exporting a
large dataset.

```python
result = manager.export(
    key="equities/minute/AAPL",
    output_path="./large_export.csv",
    format_type="csv",
    date_filter=("2024-01-01", "2024-01-31"),
    columns=["timestamp", "close", "volume"]
)
```

## Error Handling

Export operations return detailed results:

```python
result = manager.export(key="data/key", output_path="./out", format_type="csv")

if result.success:
    print(f"Success! Exported {result.rows_exported} rows")
    print(f"File size: {result.file_size / 1024 / 1024:.2f} MB")
    print(f"Duration: {result.duration_seconds:.2f} seconds")
else:
    print(f"Export failed: {result.error}")
```

## Best Practices

1. **Choose appropriate format**:
   - CSV for data analysis in Python/R
   - Excel for business reports
   - JSON for web applications

2. **Use compression when required**:
   ```python
   manager.export(
       key="equities/minute/AAPL",
       output_path="./exports/AAPL.csv.gz",
       format_type="csv",
       compression="gzip"
   )
   ```

3. **Filter unnecessary data**:
   ```python
   manager.export(
       key="crypto/minute/BTC",
       output_path="./exports/BTC.csv",
       format_type="csv",
       date_filter=("2024-01-01", "2024-01-31")
   )
   ```

4. **Batch similar exports**:
   ```python
   manager.export_pattern(
       pattern="equities/daily/*",
       output_path="./daily_report.xlsx",
       format_type="excel"
   )
   ```

5. **Validate exports**:
   - Check row counts match expectations
   - Verify date ranges are correct
   - Test with small dataset first

## Troubleshooting

### Excel Export Issues

Excel export uses the included openpyxl dependency. XlsxWriter is optional:

```bash
uv add xlsxwriter
```

### Memory Errors

For large datasets, filter at storage read time before exporting:

```python
manager.export(
    key="large/dataset",
    output_path="./exports/data.csv",
    format_type="csv",
    date_filter=("2024-01-01", "2024-01-31"),
    columns=["timestamp", "close"]
)
```

### Permission Errors

Ensure output directory exists and is writable:

```bash
mkdir -p ./exports
chmod 755 ./exports
```

## Future Enhancements

- [ ] HDF5 format support
- [ ] Parquet pass-through export
- [ ] Custom date/time formatting
- [ ] Export scheduling
- [ ] Export to cloud storage (S3, GCS)
- [ ] Streaming exports for real-time data
