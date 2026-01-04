# Yahoo Finance Provider

## Overview

The Yahoo Finance provider fetches real-time and historical market data using the `yfinance` library. It provides free access to stock prices, volumes, and basic market data.

## Features

- **Free Access**: No API key required
- **Rate Limiting**: Built-in rate limiting to avoid overwhelming the API (default: 0.5 requests/second)
- **Multiple Frequencies**: Supports minute, hourly, daily, weekly, and monthly data
- **Auto-Adjustment**: Automatically adjusts for stock splits and dividends
- **Robust Error Handling**: Retry logic for transient failures

## Usage

### CLI Usage

```bash
# Fetch daily data for Apple stock
ml4t-data load --provider yahoo --symbol AAPL --start 2024-01-01 --end 2024-01-31

# Fetch minute-level data
ml4t-data load --provider yahoo --symbol MSFT --start 2024-01-01 --end 2024-01-01 --frequency minute

# Fetch weekly data
ml4t-data load --provider yahoo --symbol GOOGL --start 2023-01-01 --end 2024-01-01 --frequency weekly
```

### Python API Usage

```python
from ml4t.data.providers.yahoo import YahooFinanceProvider
from ml4t.data.storage.filesystem import FileSystemBackend
from ml4t.data.pipeline import Pipeline
from ml4t.data.core.config import Config

# Initialize provider
provider = YahooFinanceProvider(
    max_requests_per_second=0.5,  # Rate limit
    enable_progress=False          # Progress bars
)

# Fetch data directly
df = provider.fetch_ohlcv(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-01-31",
    frequency="daily"
)

# Or use with pipeline
config = Config()
storage = FileSystemBackend(data_root=config.data_root)
pipeline = Pipeline(provider, storage, config)

# Load data through pipeline
key = pipeline.run_load(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-01-31",
    frequency="daily",
    asset_class="equities"
)
```

## Supported Frequencies

The provider maps standard frequency names to Yahoo Finance intervals:

| Frequency | Yahoo Interval | Description |
|-----------|---------------|-------------|
| `minute` | `1m` | 1-minute bars |
| `5minute` | `5m` | 5-minute bars |
| `15minute` | `15m` | 15-minute bars |
| `30minute` | `30m` | 30-minute bars |
| `hourly` | `1h` | Hourly bars |
| `daily` | `1d` | Daily bars (default) |
| `weekly` | `1wk` | Weekly bars |
| `monthly` | `1mo` | Monthly bars |

## Rate Limiting

The provider includes automatic rate limiting to prevent API throttling:

- Default rate: 0.5 requests per second (1 request every 2 seconds)
- Configurable via `max_requests_per_second` parameter
- Thread-safe implementation
- Automatic waiting when rate limit is reached

## Limitations

1. **Historical Data**:
   - Minute data: Available for last 30 days
   - Hourly data: Available for last 730 days (2 years)
   - Daily/Weekly/Monthly: Full historical data available

2. **API Limitations**:
   - No official API documentation
   - Subject to Yahoo Finance terms of service
   - May experience occasional downtime
   - Data quality varies for less liquid securities

3. **Data Coverage**:
   - Best coverage for US equities
   - Limited international market support
   - No options, futures, or forex data through this provider

## Error Handling

The provider includes robust error handling:

- Automatic retry with exponential backoff for network errors
- Graceful handling of empty responses (e.g., invalid symbols)
- Detailed logging for debugging
- Returns empty DataFrame with correct schema on errors

## Best Practices

1. **Rate Limiting**: Always use rate limiting to avoid being blocked
2. **Batch Requests**: Process multiple symbols sequentially, not in parallel
3. **Cache Data**: Store fetched data locally to minimize API calls
4. **Error Handling**: Always check for empty DataFrames in your code
5. **Respect ToS**: Use data in compliance with Yahoo Finance terms of service

## Troubleshooting

### Common Issues

1. **Empty Data**: Symbol may be invalid or delisted
2. **Rate Limit Errors**: Reduce `max_requests_per_second`
3. **Connection Errors**: Check internet connection; retry logic will handle transient issues
4. **Data Quality**: Some securities may have missing or incorrect data

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG)
)
```
