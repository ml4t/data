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
ml4t-data export equities/daily/AAPL --output ./exports/aapl.csv --format csv
```

### Export with Compression

Export with gzip compression:

```bash
ml4t-data export equities/daily/AAPL --output ./exports/ --format csv --compression gzip
```

### Export Multiple Datasets

Use wildcards to export multiple datasets:

```bash
# Export all daily equities data
ml4t-data export "equities/daily/*" --output ./exports/ --format csv

# Export to single Excel file with multiple sheets
ml4t-data export "equities/daily/*" --output ./report.xlsx --format excel
```

### Filter by Date Range

Export only data within a specific date range:

```bash
ml4t-data export crypto/daily/BTC \
  --output ./btc_2024.csv \
  --format csv \
  --start 2024-01-01 \
  --end 2024-03-31
```

### Select Specific Columns

Export only specific columns:

```bash
ml4t-data export equities/daily/AAPL \
  --output ./aapl_prices.csv \
  --format csv \
  --columns timestamp,close,volume
```

### Add Calculated Fields

Add returns and volatility calculations:

```bash
ml4t-data export equities/daily/AAPL \
  --output ./aapl_analysis.csv \
  --format csv \
  --add-returns \
  --add-volatility
```

## Python API

### Basic Export

```python
from ml4t.data.export.manager import ExportManager
from ml4t.data.storage.filesystem import FileSystemBackend

# Initialize
storage = FileSystemBackend(data_root="./data")
manager = ExportManager(storage=storage)

# Export single dataset
result = manager.export(
    key="equities/daily/AAPL",
    output_path="./exports/",
    format="csv"
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
    format="excel"
)
```

### Export with Transformations

```python
# Export with filters and calculations
result = manager.export(
    key="crypto/daily/BTC",
    output_path="./btc_analysis.csv",
    format="csv",
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
    format="csv"
)

print(f"Exported {len(results)} datasets")
```

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

1. **Use compression** - Reduces file size by 70-90%
2. **Filter by date** - Export only needed time periods
3. **Select columns** - Export only required fields
4. **Use chunking** - Automatically handled for large exports

### Memory Usage

The export system processes data in chunks to minimize memory usage:

```python
# Configure batch size for large exports
result = manager.export(
    key="equities/minute/AAPL",
    output_path="./large_export.csv",
    format="csv",
    batch_size=100000  # Process 100k rows at a time
)
```

## Error Handling

Export operations return detailed results:

```python
result = manager.export(key="data/key", output_path="./out", format="csv")

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

2. **Use compression for large files**:
   ```bash
   ml4t-data export equities/minute/AAPL --output ./exports/ --compression gzip
   ```

3. **Filter unnecessary data**:
   ```bash
   ml4t-data export crypto/minute/BTC --start 2024-01-01 --end 2024-01-31
   ```

4. **Batch similar exports**:
   ```bash
   ml4t-data export "equities/daily/*" --output ./daily_report.xlsx --format excel
   ```

5. **Validate exports**:
   - Check row counts match expectations
   - Verify date ranges are correct
   - Test with small dataset first

## Troubleshooting

### Excel Export Issues

If Excel export fails, ensure xlsxwriter is installed:

```bash
pip install xlsxwriter
```

### Memory Errors

For very large datasets, use CSV format with compression:

```bash
ml4t-data export large/dataset --format csv --compression gzip
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
