# ML4T Data - Agent Reference

> **Quick Start**: `from ml4t.data import YahooFinanceProvider`

## Purpose
Unified market data acquisition and storage for quantitative finance. Multi-provider support with incremental updates and Hive-partitioned storage.

## Installation
```bash
pip install ml4t-data
```

## Core Usage Patterns

### Fetch OHLCV Data (Single Symbol)
```python
from ml4t.data.providers import YahooFinanceProvider

provider = YahooFinanceProvider()
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31", frequency="daily")
```

### Fetch Multiple Symbols (Async)
```python
from ml4t.data.managers.async_batch import async_batch_load

async with YahooFinanceProvider() as provider:
    df = await async_batch_load(
        provider,
        symbols=["AAPL", "MSFT", "GOOGL"],
        start="2024-01-01",
        end="2024-12-31",
        max_concurrent=10
    )
```

### Store Data (Hive Partitioned)
```python
from ml4t.data.storage import HiveStorage, StorageConfig

storage = HiveStorage(config=StorageConfig(base_path="~/ml4t-data"))
storage.write(df, symbol="AAPL", provider="yahoo", frequency="daily")
```

### Read Stored Data
```python
df = storage.read(symbol="AAPL", provider="yahoo", frequency="daily")
df = storage.read(symbol="AAPL", start="2024-06-01", end="2024-12-31")
```

## Available Providers

### Equity Providers
| Provider | Class | Free Tier | Notes |
|----------|-------|-----------|-------|
| Yahoo Finance | `YahooFinanceProvider` | Unlimited | Best for quick start |
| EODHD | `EODHDProvider` | 500/day | 60+ exchanges |
| Databento | `DatabentoProvider` | Trial | Institutional grade |
| Polygon | `PolygonProvider` | Limited | US equities |
| Tiingo | `TiingoProvider` | 500/day | US alternative |
| TwelveData | `TwelveDataProvider` | 800/day | Multi-asset |
| Finnhub | `FinnhubProvider` | Limited | Real-time quotes |

### Crypto Providers
| Provider | Class | Free Tier | Notes |
|----------|-------|-----------|-------|
| Binance | `BinanceProvider` | Generous | Spot + futures |
| CoinGecko | `CoinGeckoProvider` | 50/min | Market overview |
| CryptoCompare | `CryptoCompareProvider` | 250k/mo | Historical |
| OKX | `OKXProvider` | Unlimited | Exchange data |

### Factor Providers
| Provider | Class | Datasets | Notes |
|----------|-------|----------|-------|
| Fama-French | `FamaFrenchProvider` | 50+ | FF3, FF5, Momentum |
| AQR | `AQRFactorProvider` | 16 | QMJ, BAB, TSMOM |
| FRED | `FREDProvider` | Thousands | Economic data |

### Historical/Fallback
| Provider | Class | Notes |
|----------|-------|-------|
| WikiPrices | `WikiPricesProvider` | US equities 1962-2018 |
| MockProvider | `MockProvider` | Testing only |

## CLI Commands
```bash
# Fetch data
ml4t-data fetch --provider yahoo --symbol AAPL --start 2024-01-01

# Update all datasets
ml4t-data update-all -c ml4t-data.yaml

# List stored data
ml4t-data list --storage ~/ml4t-data

# Show status
ml4t-data status --symbol AAPL
```

## Configuration (YAML)
```yaml
storage:
  path: ~/ml4t-data

datasets:
  us_equities:
    provider: yahoo
    symbols: [AAPL, MSFT, GOOGL, AMZN, META]
    frequency: daily

  sp500:
    provider: yahoo
    symbols_file: sp500.txt  # One symbol per line
    frequency: daily
```

## Key Classes

### Providers
- `BaseProvider` - Abstract base with Template Method pattern
- `AsyncBaseProvider` - Async variant for httpx-based providers
- All providers implement `OHLCVProvider` protocol

### Storage
- `HiveStorage` - Partitioned Parquet (recommended)
- `FlatStorage` - Simple Parquet files
- `MetadataTracker` - Update timestamps

### Utilities
- `AnomalyManager` - Detect data quality issues
- `GapDetector` - Find missing data
- `RateLimiter` - API rate management

## Important Notes

1. **Provider Reuse**: Always reuse provider instances (30ms init cost)
2. **Rate Limits**: Built-in rate limiting and circuit breaker
3. **Validation**: Automatic OHLC invariant checking
4. **Storage Format**: Hive partitions: `provider/frequency/year=YYYY/month=MM/symbol/`

## Error Handling
```python
from ml4t.data.core.exceptions import (
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
    DataValidationError
)

try:
    df = provider.fetch_ohlcv(...)
except RateLimitError:
    # Wait and retry
except SymbolNotFoundError:
    # Invalid symbol
```

## Dependencies
- polars (DataFrames)
- httpx (async HTTP)
- pydantic (validation)
- pyarrow (Parquet I/O)
