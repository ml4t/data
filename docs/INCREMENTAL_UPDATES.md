# Incremental Updates & Data Management

This document describes the incremental update system implemented in Sprint 004, including gap detection, file locking, chunked storage, and metadata tracking.

## Overview

The incremental update system allows efficient data updates without re-downloading entire datasets. Key features include:

- **Incremental Updates**: Only fetch new data since last update
- **Gap Detection**: Identify and fill missing data points
- **File Locking**: Safe concurrent access to data files
- **Chunked Storage**: Split large datasets into manageable time-based chunks
- **Metadata Tracking**: Monitor dataset health and update history

## CLI Commands

### Initial Data Load

First time loading data for a symbol:

```bash
# Load historical data
ml4t-data load --provider yahoo --symbol AAPL --start 2023-01-01 --end 2024-01-01

# Load with specific frequency
ml4t-data load -p yahoo -s AAPL --start 2023-01-01 --end 2024-01-01 -f daily
```

### Incremental Updates

Update existing data with new data:

```bash
# Basic update (uses existing provider)
ml4t-data update --symbol AAPL

# Update with options
ml4t-data update -s AAPL --lookback-days 10 --fill-gaps --show-status

# Update without gap filling
ml4t-data update -s AAPL --no-fill-gaps

# Update with different provider
ml4t-data update -s AAPL --provider yahoo
```

Options:
- `--lookback-days/-l`: Days to look back for validation (default: 7)
- `--fill-gaps/--no-fill-gaps`: Enable/disable gap filling (default: enabled)
- `--show-status`: Display detailed update status and history
- `--provider/-p`: Override provider (uses existing if not specified)

### Health Monitoring

Check health status of all datasets:

```bash
# Basic health check
ml4t-data health

# Detailed health check
ml4t-data health --verbose

# Custom staleness threshold
ml4t-data health --stale-days 3 --verbose
```

Output shows:
- Total datasets and their health status (✅ healthy, ⚠️ stale, ❌ error)
- Total rows across all datasets
- Breakdown by asset class
- Individual dataset details (with --verbose)

## Workflow Examples

### Example 1: Daily Stock Data Updates

```bash
# Initial load of AAPL data
ml4t-data load -p yahoo -s AAPL --start 2023-01-01 --end 2024-01-01

# Daily update (run via cron)
ml4t-data update -s AAPL --show-status

# Check health weekly
ml4t-data health --verbose
```

### Example 2: Multiple Symbol Management

```bash
# Load multiple symbols
for symbol in AAPL GOOGL MSFT NVDA; do
    ml4t-data load -p yahoo -s $symbol --start 2023-01-01 --end 2024-01-01
done

# Update all symbols
for symbol in AAPL GOOGL MSFT NVDA; do
    ml4t-data update -s $symbol
done

# Check overall health
ml4t-data health
```

### Example 3: Crypto Data (24/7 Trading)

```bash
# Load crypto data
ml4t-data load -p yahoo -s BTC-USD --start 2023-01-01 --end 2024-01-01 -a crypto

# Update with gap detection (important for 24/7 markets)
ml4t-data update -s BTC-USD -a crypto --fill-gaps --show-status
```

## Technical Details

### Gap Detection

The system distinguishes between expected and unexpected gaps:

- **Stock Markets**: Weekends and after-hours are expected gaps
- **Crypto Markets**: 24/7 trading, all gaps are unexpected
- **Configurable Tolerance**: 10% default tolerance for gap detection

Gap filling methods:
- `forward`: Use last known value (default)
- `backward`: Use next known value
- `interpolate`: Linear interpolation
- `zero`: Fill with zeros

### File Locking

Thread-safe and process-safe file locking ensures data integrity:

- Uses `filelock` library for cross-platform compatibility
- Automatic lock acquisition and release
- Configurable timeout (default: 30 seconds)
- Prevents corruption during concurrent reads/writes

### Chunked Storage

Large datasets are split into time-based chunks:

- **Monthly chunks** (default): 30-day periods
- **Weekly chunks**: 7-day periods
- **Quarterly chunks**: 90-day periods
- **Yearly chunks**: 365-day periods

Benefits:
- Efficient incremental updates (only update relevant chunks)
- Parallel processing capability
- Reduced memory usage for large datasets
- Fast time-range queries

### Metadata Tracking

Each dataset maintains metadata including:

- Update history (last 100 updates)
- Health status (healthy/stale/error)
- Data range and row count
- Provider information
- Error tracking

Health checks consider:
- Days since last update
- Data currency (how far behind current date)
- Recent error frequency

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Provider  │────▶│   Pipeline   │────▶│   Storage   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │ Gap Detector │     │ File Lock   │
                    └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   Metadata   │     │   Chunks    │
                    │   Tracker    │     │   Storage   │
                    └──────────────┘     └─────────────┘
```

### Components

1. **Pipeline** (`src/ml4t-data/pipeline.py`)
   - Orchestrates data flow
   - `run_load()`: Full data load
   - `run_update()`: Incremental update

2. **Gap Detector** (`src/ml4t-data/utils/gaps.py`)
   - Identifies missing data points
   - Market hours awareness
   - Multiple fill strategies

3. **File Locking** (`src/ml4t-data/utils/locking.py`)
   - Thread/process-safe access
   - Automatic cleanup
   - Timeout configuration

4. **Chunked Storage** (`src/ml4t-data/storage/chunked.py`)
   - Time-based data splitting
   - Efficient updates
   - Metadata indexing

5. **Metadata Tracker** (`src/ml4t-data/storage/metadata_tracker.py`)
   - Update history
   - Health monitoring
   - Summary statistics

## Best Practices

### Update Frequency

- **Daily data**: Update once per day after market close
- **Minute data**: Update every few hours during market hours
- **Crypto**: Update more frequently (hourly or more)

### Error Handling

The system includes robust error handling:

```python
# Automatic retries with exponential backoff
@with_retry(max_attempts=3, min_wait=1.0, max_wait=30.0)
def _fetch_data_with_retry(...)
```

### Monitoring

Set up monitoring using the health command:

```bash
# Cron job for daily health check
0 9 * * * ml4t-data health --stale-days 2 >> /var/log/ml4t-data-health.log
```

### Storage Management

Monitor disk usage as datasets grow:

```bash
# Check data directory size
du -sh ~/.ml4t-data/data/

# List all datasets
ml4t-data list

# Remove old data if needed (manual process)
rm -rf ~/.ml4t-data/data/equities/daily/OLD_SYMBOL
```

## Troubleshooting

### Common Issues

1. **"No existing data found"**
   - Run `ml4t-data load` first before using `update`
   - Check the correct asset class and frequency

2. **"Lock timeout"**
   - Another process is accessing the file
   - Check for stuck processes
   - Increase timeout if needed

3. **"Gaps detected"**
   - Normal for some data sources
   - Use `--fill-gaps` to automatically fill
   - Check provider data quality

4. **"Data is stale"**
   - Run update more frequently
   - Check provider connectivity
   - Verify market hours settings

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set log level in .env
echo "ML4T Data_LOG_LEVEL=DEBUG" >> .env

# Run with verbose output
ml4t-data update -s AAPL --show-status
```

## Performance Considerations

### Memory Usage

- Chunked storage keeps memory usage low
- Each chunk is processed independently
- Typical chunk size: 10-50 MB

### Disk Usage

- Parquet compression reduces storage by 50-80%
- Monthly chunks balance size and performance
- Metadata overhead: ~1KB per dataset

### Update Speed

- Incremental updates are 10-100x faster than full loads
- Gap detection adds minimal overhead (<1 second)
- File locking has negligible performance impact

## Future Enhancements

Potential improvements for future sprints:

1. **Parallel Updates**: Update multiple symbols concurrently
2. **Smart Scheduling**: Automatic update scheduling based on asset class
3. **Data Validation**: Detect and flag suspicious data points
4. **Compression Options**: Support for different compression algorithms
5. **Archive Storage**: Move old data to compressed archives
6. **Update Notifications**: Email/webhook alerts for failures
7. **REST API**: HTTP endpoint for remote updates
8. **Data Reconciliation**: Compare and sync with multiple providers

## Summary

The incremental update system provides efficient, reliable data management with:

- ✅ Minimal bandwidth usage (only fetch new data)
- ✅ Data integrity (file locking, gap detection)
- ✅ Scalability (chunked storage)
- ✅ Observability (health monitoring, update history)
- ✅ Flexibility (configurable options, multiple providers)

This foundation enables building robust quantitative trading systems with reliable, up-to-date market data.
