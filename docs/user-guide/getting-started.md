# Getting Started with ML4T Data

Welcome to ML4T Data (Quantitative Ledger Data Manager)! This guide will help you get up and running quickly.

## Installation

### System Requirements

- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended)
- 10GB free disk space for data storage
- Internet connection for data fetching

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ml4t-data.git
cd ml4t-data

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# For additional features
pip install -e ".[api]"    # API server support
pip install -e ".[dev]"    # Development tools
```

### Verify Installation

```bash
# Check CLI is working
ml4t-data --version

# View available commands
ml4t-data --help
```

## First Steps

### 1. Initialize Configuration

Create your configuration directory and default config:

```bash
ml4t-data init
```

This creates:
- `~/.ml4t-data/` - Main configuration directory
- `~/.ml4t-data/config.yaml` - Configuration file
- `~/.ml4t-data/data/` - Data storage directory

### 2. Configure Data Providers

Edit `~/.ml4t-data/config.yaml` to set up your providers:

```yaml
providers:
  yahoo:
    enabled: true
    rate_limit: 10

  binance:
    enabled: true
    # Optional API credentials for higher limits
    api_key: ${BINANCE_API_KEY}
    api_secret: ${BINANCE_SECRET}

  cryptocompare:
    enabled: true
    api_key: ${CRYPTOCOMPARE_API_KEY}  # Free tier available
```

### 3. Fetch Your First Data

#### Stock Data (Yahoo Finance)

```bash
# Fetch Apple stock data for 2024
ml4t-data fetch AAPL --provider yahoo --start 2024-01-01

# Fetch multiple symbols
ml4t-data fetch AAPL MSFT GOOGL --provider yahoo

# Fetch with specific date range
ml4t-data fetch AAPL --start 2024-01-01 --end 2024-06-30
```

#### Cryptocurrency Data (Binance)

```bash
# Fetch Bitcoin/USDT hourly data
ml4t-data fetch BTC/USDT --provider binance --frequency 1h

# Fetch daily data
ml4t-data fetch ETH/USDT --provider binance --frequency 1d

# Fetch minute data (last 7 days)
ml4t-data fetch BTC/USDT --provider binance --frequency 1m --start 2024-01-01
```

### 4. View and Export Data

#### List Available Data

```bash
# Show all stored symbols
ml4t-data list

# Show details for a specific symbol
ml4t-data info AAPL
```

#### Export Data

```bash
# Export to CSV
ml4t-data export AAPL --format csv --output aapl_2024.csv

# Export to Excel with date range
ml4t-data export AAPL --format excel --start 2024-01-01 --end 2024-06-30

# Export to Parquet (efficient format)
ml4t-data export AAPL --format parquet --output aapl.parquet
```

### 5. Update Existing Data

Keep your data current with incremental updates:

```bash
# Update single symbol
ml4t-data update AAPL

# Update multiple symbols
ml4t-data update AAPL MSFT GOOGL

# Update all symbols from a provider
ml4t-data update --provider yahoo --all
```

## Python API Usage

### Basic Example

```python
from ml4t-data import ML4T Data
import polars as pl

# Initialize ML4T Data
ml4t-data = ML4T Data()

# Fetch data
df = ml4t-data.fetch(
    symbol="AAPL",
    provider="yahoo",
    start="2024-01-01",
    end="2024-12-31"
)

# Display first few rows
print(df.head())

# Basic analysis
print(f"Average close price: ${df['close'].mean():.2f}")
print(f"Maximum high: ${df['high'].max():.2f}")
print(f"Total volume: {df['volume'].sum():,.0f}")
```

### Working with Multiple Symbols

```python
from ml4t-data import ML4T Data

ml4t-data = ML4T Data()

# Fetch multiple symbols
symbols = ["AAPL", "MSFT", "GOOGL"]
data = {}

for symbol in symbols:
    data[symbol] = ml4t-data.fetch(symbol, start="2024-01-01")
    print(f"Fetched {len(data[symbol])} rows for {symbol}")

# Compare performance
for symbol, df in data.items():
    returns = ((df['close'][-1] - df['close'][0]) / df['close'][0] * 100)
    print(f"{symbol}: {returns:.2f}% return")
```

### Data Validation

```python
from ml4t-data import ML4T Data
from ml4t.data.validation import OHLCVValidator

ml4t-data = ML4T Data()
validator = OHLCVValidator()

# Fetch and validate data
df = ml4t-data.fetch("AAPL", start="2024-01-01")
result = validator.validate(df)

if result.passed:
    print("✅ Data validation passed")
else:
    print("❌ Data validation failed:")
    for issue in result.issues:
        print(f"  - {issue.severity}: {issue.message}")
```

## API Server

### Starting the Server

```bash
# Start with default settings
ml4t-data serve

# Specify port and host
ml4t-data serve --host 0.0.0.0 --port 8000

# With authentication enabled
ml4t-data serve --auth
```

### Using the API

Once the server is running, you can access:
- Interactive docs: http://localhost:8000/docs
- API endpoints: http://localhost:8000/api/v1/

#### Example API Calls

```python
import requests

# Get data for a symbol
response = requests.get(
    "http://localhost:8000/api/v1/data/AAPL",
    params={"start": "2024-01-01", "end": "2024-06-30"}
)
data = response.json()

# Update symbol data
response = requests.post(
    "http://localhost:8000/api/v1/update/AAPL",
    headers={"X-API-Key": "your-api-key"}  # If auth enabled
)

# Export data
response = requests.post(
    "http://localhost:8000/api/v1/export",
    json={
        "symbol": "AAPL",
        "format": "csv",
        "start": "2024-01-01"
    }
)
```

## Common Workflows

### Daily Market Data Update

Create a script to update all your symbols daily:

```python
#!/usr/bin/env python
"""Daily market data update script"""

from ml4t-data import ML4T Data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def daily_update():
    ml4t-data = ML4T Data()

    # List of symbols to track
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN",  # Tech stocks
        "JPM", "BAC", "GS",                # Banks
        "BTC/USDT", "ETH/USDT"             # Crypto
    ]

    for symbol in symbols:
        try:
            provider = "binance" if "/" in symbol else "yahoo"
            ml4t-data.update(symbol, provider=provider)
            logger.info(f"✅ Updated {symbol}")
        except Exception as e:
            logger.error(f"❌ Failed to update {symbol}: {e}")

if __name__ == "__main__":
    daily_update()
```

### Data Quality Check

Monitor your data quality:

```python
from ml4t-data import ML4T Data
from ml4t.data.validation import OHLCVValidator, CrossValidator

def check_data_quality(symbol: str):
    ml4t-data = ML4T Data()
    df = ml4t-data.get(symbol)

    # Basic validation
    validator = OHLCVValidator()
    result = validator.validate(df)

    # Cross validation
    cross_validator = CrossValidator()
    cross_result = cross_validator.validate(df)

    # Report
    print(f"\nData Quality Report for {symbol}")
    print("=" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Basic validation: {'✅ Passed' if result.passed else '❌ Failed'}")
    print(f"Cross validation: {'✅ Passed' if cross_result.passed else '❌ Failed'}")

    if not result.passed:
        print("\nIssues found:")
        for issue in result.issues[:5]:  # Show first 5 issues
            print(f"  - {issue.severity}: {issue.message}")

check_data_quality("AAPL")
```

## Troubleshooting

### Common Issues and Solutions

#### Rate Limiting Errors

**Problem**: "Rate limit exceeded" errors

**Solution**: Reduce the rate limit in your config:
```yaml
providers:
  yahoo:
    rate_limit: 5  # Reduce from default 10
```

#### Memory Issues with Large Datasets

**Problem**: High memory usage when processing large datasets

**Solution**: Use chunked processing:
```python
from ml4t.data.performance import DataFrameOptimizer

optimizer = DataFrameOptimizer()
df = optimizer.optimize_memory(df)  # Reduces memory by 40-60%
```

#### Missing Data Periods

**Problem**: Gaps in historical data

**Solution**: Use the gap detection and filling:
```bash
# Detect gaps
ml4t-data gaps AAPL

# Fill gaps automatically
ml4t-data update AAPL --fill-gaps
```

#### Slow Queries

**Problem**: Slow data retrieval

**Solution**: Enable caching:
```python
from ml4t.data.performance import cache_result

@cache_result(ttl_seconds=300)
def get_data(symbol):
    return ml4t-data.get(symbol)
```

## Best Practices

1. **Regular Updates**: Set up a cron job or scheduled task for daily updates
2. **Data Validation**: Always validate data after fetching
3. **Error Handling**: Implement proper error handling in production scripts
4. **Backup**: Regularly backup your data directory
5. **Monitor Storage**: Keep an eye on disk usage
6. **Use Appropriate Frequencies**: Don't fetch minute data if you only need daily

## Next Steps

- Read the [Configuration Guide](configuration.md) for advanced settings
- Check the [CLI Reference](cli-reference.md) for all commands
- Explore [Example Notebooks](../examples/notebooks/) for analysis workflows
- Learn about [Provider Development](../developer-guide/providers.md) to add custom data sources

## Getting Help

- 📖 Documentation: https://yourusername.github.io/ml4t-data/
- 🐛 Report Issues: https://github.com/yourusername/ml4t-data/issues
- 💬 Discussions: https://github.com/yourusername/ml4t-data/discussions
- 📧 Email: support@example.com
