# ml4t-data

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/ml4t-data)](https://pypi.org/project/ml4t-data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Market data acquisition and management library providing unified access to 19 data providers with automated incremental updates and Hive-partitioned storage.

## Features

- **19 Data Providers**: Crypto, equities, forex, futures, factor data with consistent API
- **Unified Interface**: `fetch_ohlcv()` works identically across all providers
- **Automated Updates**: CLI for incremental updates, gap detection, backfilling
- **Smart Storage**: Hive-partitioned Parquet with metadata tracking
- **Data Validation**: OHLC invariant checks, deduplication, anomaly detection
- **Polars-Based**: 10-100x faster than pandas alternatives

## Installation

```bash
pip install ml4t-data
```

## Quick Start

```python
from ml4t.data.providers import YahooFinanceProvider

# No API key needed
provider = YahooFinanceProvider()
data = provider.fetch_ohlcv("AAPL", "2020-01-01", "2024-12-31")

print(data.head())
# shape: (1,258, 7)
# ┌────────────┬────────┬────────┬────────┬────────┬──────────┬────────┐
# │ date       │ open   │ high   │ low    │ close  │ volume   │ symbol │
```

```python
from ml4t.data.providers import CoinGeckoProvider

# Crypto data (no API key needed)
provider = CoinGeckoProvider()
btc = provider.fetch_ohlcv("bitcoin", "2024-01-01", "2024-12-31")
```

## Providers

### Free (No API Key)

| Provider | Coverage | Rate Limit | Best For |
|----------|----------|------------|----------|
| **Yahoo Finance** | Stocks, ETFs, crypto, forex | ~2000/hour | US equities, research |
| **CoinGecko** | 10,000+ cryptocurrencies | 50/min | Crypto historical |
| **FRED** | 850,000 economic series | 120/min | Macro indicators |
| **Fama-French** | Academic factor data | Unlimited | Factor research |
| **AQR** | Alternative factors | Unlimited | QMJ, BAB, TSMOM |

### Free (API Key Required)

| Provider | Free Tier | Paid From | Best For |
|----------|-----------|-----------|----------|
| **EODHD** | 500 calls/day | €20/month | Global markets (60+ exchanges) |
| **Tiingo** | 1000 calls/day | $30/month | High-quality US data |
| **Twelve Data** | 800 calls/day | $10/month | Multi-asset |
| **Alpha Vantage** | 25 calls/day | $50/month | Conservative research |

### Professional

| Provider | Starting Price | Coverage |
|----------|---------------|----------|
| **Databento** | $9/month | CME, CBOE, ICE futures/options |
| **Polygon** | $99/month | Stocks, options, forex, crypto |
| **Finnhub** | $60/month | 70+ global exchanges |

### Specialized

| Provider | Type | Coverage |
|----------|------|----------|
| **Binance** | Crypto exchange | 600+ pairs (geo-restricted) |
| **OANDA** | Forex broker | Major/minor pairs |
| **Wiki Prices** | Static dataset | 3,199 US stocks (1962-2018) |
| **AlgoSeek** | Static dataset | NASDAQ 100 minute bars (2015-2017) |

## CLI Usage

```bash
# Fetch data
ml4t-data fetch AAPL MSFT GOOGL --provider yahoo --start 2020-01-01

# Configuration-driven updates
ml4t-data update-all -c config.yaml

# Preview changes (dry run)
ml4t-data update-all -c config.yaml --dry-run

# Detect and fill gaps
ml4t-data update-all -c config.yaml --detect-gaps

# List stored data
ml4t-data list -c config.yaml

# Check specific symbol
ml4t-data info AAPL -c config.yaml
```

### Configuration File

```yaml
# ml4t-data.yaml
storage:
  path: ~/data/market

datasets:
  sp500_daily:
    provider: yahoo
    symbols_file: symbols/sp500.txt
    frequency: daily
    start_date: 2015-01-01

  crypto:
    provider: coingecko
    symbols: [bitcoin, ethereum, solana]
    frequency: daily
    start_date: 2020-01-01
```

### Automated Updates (Cron)

```bash
# Daily at 5 PM (after market close)
0 17 * * 1-5 ml4t-data update-all -c ~/ml4t-data.yaml
```

## Storage

Data is stored in Hive-partitioned Parquet format:

```
~/data/market/
├── yahoo/
│   └── daily/
│       ├── symbol=AAPL/
│       │   └── data.parquet
│       └── symbol=MSFT/
│           └── data.parquet
└── coingecko/
    └── daily/
        └── symbol=bitcoin/
            └── data.parquet
```

Query with DuckDB (zero-copy):

```python
import duckdb

conn = duckdb.connect()
result = conn.execute("""
    SELECT * FROM read_parquet('~/data/market/yahoo/daily/**/*.parquet')
    WHERE symbol IN ('AAPL', 'MSFT')
    AND date >= '2024-01-01'
""").pl()
```

## Data Validation

Built-in OHLC validation:

```python
from ml4t.data.validation import validate_ohlcv

# Checks: high >= low, high >= open/close, low <= open/close
# Detects: duplicates, gaps, anomalies
issues = validate_ohlcv(data)
if issues:
    print(f"Found {len(issues)} data quality issues")
```

## API Reference

### Providers

```python
from ml4t.data.providers import (
    # Free (no key)
    YahooFinanceProvider,
    CoinGeckoProvider,
    FREDProvider,
    FamaFrenchProvider,
    AQRProvider,

    # Free (key required)
    EODHDProvider,
    TiingoProvider,
    TwelveDataProvider,
    AlphaVantageProvider,

    # Professional
    DataBentoProvider,
    PolygonProvider,
    FinnhubProvider,

    # Specialized
    BinanceProvider,
    OANDAProvider,
)
```

### Common Interface

All providers implement the same interface:

```python
class DataProvider:
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None = None,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """Fetch OHLCV data for a symbol."""

    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str | None = None,
        frequency: str = "daily",
    ) -> dict[str, pl.DataFrame]:
        """Fetch OHLCV data for multiple symbols."""
```

### Storage

```python
from ml4t.data.storage import HiveStorage

storage = HiveStorage("~/data/market")

# Save data
storage.save(data, provider="yahoo", frequency="daily", symbol="AAPL")

# Load data
data = storage.load(provider="yahoo", frequency="daily", symbol="AAPL")

# Query metadata
metadata = storage.get_metadata(provider="yahoo", frequency="daily", symbol="AAPL")
print(f"Last updated: {metadata['last_updated']}")
```

## Integration with ML4T Libraries

ml4t-data is part of the ML4T library ecosystem:

```python
from ml4t.data import DataManager
from ml4t.engineer import compute_features
from ml4t.backtest import Engine, Strategy

# Complete workflow
data = DataManager().fetch("SPY", "2020-01-01", "2023-12-31")
features = compute_features(data, ["rsi", "macd", "atr"])
# ... backtest with features
```

## Ecosystem

- **ml4t-data**: Market data acquisition and storage (this library)
- **ml4t-engineer**: Feature engineering and indicators
- **ml4t-diagnostic**: Statistical validation and evaluation
- **ml4t-backtest**: Event-driven backtesting
- **ml4t-live**: Live trading platform

## Testing

```bash
# Run tests (3,071 tests)
uv run pytest tests/ -q

# Type checking
uv run ty check

# Linting
uv run ruff check src/
```

## Development

```bash
git clone https://github.com/applied-ai/ml4t-data.git
cd ml4t-data

# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/ -q

# Type checking
uv run ty check
```

## Environment Variables

```bash
# API keys
export EODHD_API_KEY="your_key"
export DATABENTO_API_KEY="your_key"
export FINNHUB_API_KEY="your_key"
export TIINGO_API_KEY="your_key"
export FRED_API_KEY="your_key"

# Configuration
export ML4T_DATA_CONFIG="~/ml4t-data.yaml"
export ML4T_DATA_LOG_LEVEL="INFO"
```

## License

MIT License - see [LICENSE](LICENSE) for details.
