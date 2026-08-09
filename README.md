# ml4t-data

[![Python 3.12-3.14](https://img.shields.io/badge/python-3.12--3.14-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/ml4t-data)](https://pypi.org/project/ml4t-data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Unified market data acquisition and storage for quantitative research workflows.

## Part of the ML4T Library Ecosystem

This library is one of six interconnected libraries supporting the machine learning for trading workflow described in [Machine Learning for Trading](https://www.ml4trading.io):

![ML4T Library Ecosystem](docs/images/ml4t_ecosystem_workflow_color.png)

Together they cover data infrastructure, feature engineering, modeling, signal evaluation, strategy backtesting, and live deployment.

## What This Library Does

Quantitative research requires consistent, reproducible access to market data from multiple sources. ml4t-data provides:

- `DataManager` as the unified OHLCV interface for registry providers with that capability
- 20+ provider adapters covering equities, crypto, futures, forex, macro, prediction markets, and factors
- Automated storage in Hive-partitioned Parquet format with metadata tracking
- Incremental updates, gap detection, and backfill via CLI
- Built-in data validation (OHLC invariants, deduplication, anomaly detection)
- Futures module for CME/ICE bulk downloads with continuous contract construction
- COT module for CFTC Commitment of Traders weekly reports
- Resilience: rate limiting, retry with exponential backoff, gap detection

The goal is to support an ongoing research workflow rather than one-off downloads. Data is stored locally, tracked for freshness, and queryable with tools like DuckDB or Polars.

![ml4t-data Architecture](docs/images/ml4t_data_architecture_print.jpeg)

## Installation

ML4T Data supports stable CPython 3.12 through 3.14 on Linux, macOS, and Windows.
Python 3.15 prereleases are tested as informational compatibility checks but are not supported
until the required dependencies publish compatible distributions. The Databento SDK currently
has no CPython 3.15 distribution, so the 3.15 prerelease lane tests the core package without the
Databento extra.

```bash
pip install ml4t-data
```

## Quick Start

### DataManager (Unified Interface)

```python
from ml4t.data import DataManager

dm = DataManager()

# Fetch and store
dm.fetch("AAPL", "2020-01-01", "2024-12-31", provider="yahoo")

# Load from local storage
data = dm.load("AAPL", "2020-01-01", "2024-12-31")

# Batch load multiple symbols
prices = dm.batch_load(["AAPL", "MSFT", "GOOGL"], "2020-01-01", "2024-12-31")

# Incremental update
dm.update("AAPL")

# List what's stored
symbols = dm.list_symbols()
metadata = dm.get_metadata("AAPL")
```

### Direct Provider Access

Providers expose capability-specific methods. OHLCV providers use `fetch_ohlcv`, economic-series
providers may also use `fetch_series`, and factor providers use `fetch`.

```python
from datetime import UTC, datetime, timedelta

from ml4t.data.providers import YahooFinanceProvider, CoinGeckoProvider, FREDProvider

# Equities
provider = YahooFinanceProvider()
data = provider.fetch_ohlcv("AAPL", "2020-01-01", "2024-12-31")

# Crypto
last_complete_day = datetime.now(UTC).date() - timedelta(days=1)
crypto = CoinGeckoProvider().fetch_ohlcv(
    "bitcoin",
    str(last_complete_day - timedelta(days=6)),
    str(last_complete_day),
)

# Economic data
# Reads FRED_API_KEY from the environment
fred = FREDProvider().fetch_series("GDP", "2020-01-01", "2024-12-31")
```

## Data Providers

### No API Key Required

| Provider | Coverage |
|----------|----------|
| Yahoo Finance | US/global equities, ETFs, crypto, forex |
| CoinGecko | 10,000+ cryptocurrencies; daily OHLCV for 29 completed UTC days |
| FXMacroData | FX macro releases, calendars, COT, commodities, sentiment |
| Fama-French | Academic factor data |
| AQR | Research factors (QMJ, BAB, HML Devil, VME, more) |
| Wiki Prices | Frozen US equities history (1962-2018) |
| Kalshi | Prediction market contracts |
| Polymarket | Prediction market history/order book snapshots |
| Binance Public | Bulk crypto data downloads |
| Binance | Crypto exchange data |
| OKX | Crypto perpetuals and funding rates |
| NASDAQ ITCH Sample | Tick-level sample data |

### Authenticated or Metered APIs

| Provider | Coverage |
|----------|----------|
| FRED | 850,000 economic series |
| Alpaca | US equities + crypto (free IEX feed) |
| EODHD | 60+ global exchanges |
| Tiingo | US equities with quality focus |
| Twelve Data | Multi-asset coverage |
| Databento | CME/ICE futures; OPRA options |
| Massive | US equities, options, futures, forex, crypto |
| Finnhub | 70+ global exchanges |
| OANDA | Forex broker data |
| CryptoCompare | Included adapter; not release-qualified for 0.1.0 |

CryptoCompare account registration was unavailable during the 0.1.0 release review, so no live
contract evidence was obtained. The adapter remains available for evaluation, but CryptoCompare is
not part of the release-qualified provider set until its contract passes on a release commit.

## Specialized Modules

### Futures

Bulk download and continuous contract construction for CME/ICE products:

```python
from ml4t.data.futures import FuturesDownloader, ContinuousContractBuilder

# Bulk download via Databento (parent symbology)
downloader = FuturesDownloader(config)
downloader.download()  # Downloads ES, NQ, CL, GC, etc.

# Build continuous contracts with configurable roll logic
builder = ContinuousContractBuilder()
continuous = builder.build(contracts_df, roll_method="volume")
```

Book-focused interface with profiling:

```python
from ml4t.data.futures import FuturesDataManager

fm = FuturesDataManager.from_config("config.yaml")
fm.download_all()
data = fm.load_ohlcv("ES")
profile = fm.generate_profile("ES")
```

### COT (Commitment of Traders)

CFTC weekly positioning data for futures markets:

```python
import polars as pl

from ml4t.data.cot import COTFetcher, combine_cot_ohlcv, create_cot_features

fetcher = COTFetcher()
official_schedule = pl.read_parquet("cot-release-schedule.parquet")
cot_data = fetcher.fetch_product(
    "ES",
    start_year=2015,
    end_year=2024,
    release_schedule=official_schedule,
)

# Point-in-time combination with OHLCV
combined = combine_cot_ohlcv(ohlcv_data, cot_data)

# Generate weekly features without counting forward-filled daily rows as new reports
features = create_cot_features(combined)
```

The schedule must contain one `report_date` and timezone-aware `available_at` timestamp per
report. CFTC publishes a tentative schedule for only the latest 13 months, so retain the exact
release timestamps you use for historical research. Shutdown catch-up releases and other
exceptions must use their actual publication timestamps. See the
[official CFTC release schedule](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm).

### Book Data Managers

Simplified interfaces for the ML4T book workflow:

```python
from ml4t.data.etfs import ETFDataManager
from ml4t.data.crypto import CryptoDataManager

# 50 diversified ETFs via Yahoo Finance
etf_dm = ETFDataManager.from_config("config.yaml")
etf_dm.download_all()
aapl = etf_dm.load_ohlcv("AAPL")

# Crypto premium index via Binance Public
crypto_dm = CryptoDataManager.from_config("config.yaml")
crypto_dm.download_premium_index()
```

## CLI for Automated Updates

```bash
# Fetch specific symbols
ml4t-data fetch -s AAPL -s MSFT -s GOOGL --provider yahoo --start 2020-01-01

# Incremental update
ml4t-data update --symbol AAPL

# Validate data quality
ml4t-data validate --symbol AAPL --anomalies

# Check storage status
ml4t-data status --detailed

# List available data
ml4t-data list-data

# Export to CSV/JSON/Excel
ml4t-data export --symbol AAPL --format-type csv --output aapl.csv

# Get symbol info
ml4t-data info --symbol AAPL
```

Configuration-driven batch updates:

```yaml
storage:
  base_path: ~/data/market

datasets:
  - name: sp500_daily
    provider: yahoo
    symbols_file: symbols/sp500.txt
    frequency: daily
    start_date: 2015-01-01

  - name: crypto
    provider: coingecko
    symbols: [bitcoin, ethereum, solana]
    frequency: daily
    initial_load_days: 29
```

## Storage Format

Data is stored in Hive-partitioned Parquet:

```
~/data/market/
├── yahoo/daily/symbol=AAPL/data.parquet
├── yahoo/daily/symbol=MSFT/data.parquet
└── coingecko/daily/symbol=bitcoin/data.parquet
```

Query with DuckDB or Polars:

```python
import duckdb

result = duckdb.execute("""
    SELECT * FROM read_parquet('~/data/market/yahoo/daily/**/*.parquet')
    WHERE symbol IN ('AAPL', 'MSFT')
    AND date >= '2024-01-01'
""").pl()
```

## Data Validation

```python
from ml4t.data.validation import OHLCVValidator, ValidationReport

validator = OHLCVValidator()
report = validator.validate(data)
# Checks: high >= low, high >= open/close, low <= open/close
# Detects: duplicates, gaps, anomalies
```

Anomaly detection:

```python
from ml4t.data.anomaly import AnomalyManager, ReturnOutlierDetector, VolumeSpikeDetector

manager = AnomalyManager([
    ReturnOutlierDetector(),
    VolumeSpikeDetector(),
])
report = manager.detect(data)
```

## Documentation

- [Getting Started](docs/user-guide/getting-started.md) - quick start guide
- [Configuration](docs/user-guide/configuration.md) - YAML config reference
- [Storage](docs/user-guide/storage.md) - Hive partitioning and backends
- [Incremental Updates](docs/user-guide/incremental-updates.md) - update strategies and gap detection
- [Data Quality](docs/user-guide/data-quality.md) - validation and anomaly detection
- [CLI Reference](docs/user-guide/cli-reference.md) - command-line interface
- [Provider Selection Guide](docs/getting-started/provider-selection.md) - choosing providers
- [Creating a Provider](docs/contributing/creating-a-provider.md) - extending with new sources

## Technical Characteristics

- **Polars-based**: Native Polars DataFrames throughout
- **Capability-specific schemas**: OHLCV, economic-series, factor, and specialized providers
  expose contracts suited to their data
- **Async support**: OHLCV providers implementing the async protocol can use parallel batch operations
- **Metadata tracking**: Last update timestamps, row counts, date ranges
- **Resilience**: Rate limiting, retry with exponential backoff, gap detection
- **Storage layouts**: Partitioned Hive and single-file Parquet storage
- **Type-safe**: Full type annotations throughout

## Related Libraries

- **ml4t-engineer**: Feature engineering and technical indicators
- **ml4t-diagnostic**: Signal evaluation and statistical validation
- **ml4t-backtest**: Event-driven backtesting
- **ml4t-live**: Live trading with broker integration

## Development

```bash
git clone https://github.com/ml4t/data.git ml4t-data
cd ml4t-data
uv sync
uv run pytest tests/ -q
uv run ty check
```

## License

MIT License - see [LICENSE](LICENSE) for details.
