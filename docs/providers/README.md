# Data Providers

ml4t-data provides unified access to **21 financial data providers** through a consistent API. All providers inherit from `BaseProvider` and implement the same `fetch_ohlcv()` interface.

For detailed pricing, terms, and gap analysis, see [PROVIDER_AUDIT.md](PROVIDER_AUDIT.md).

---

## Provider Categories

### Equity Providers

| Provider | API Key | Free Tier | Best For |
|----------|---------|-----------|----------|
| [YahooFinance](yahoo.md) | No | Unlimited* | Quick start, US equities |
| [EODHD](eodhd.md) | Yes | 20 calls/day | Global equities (60+ exchanges) |
| [Tiingo](tiingo.md) | Yes | 1,000 req/day | US equities alternative |
| [Finnhub](finnhub.md) | Yes | 60 req/min | Company metrics, real-time |
| [WikiPrices](wiki_prices.md) | No | Local file | Historical (1962-2018) |

*Yahoo Finance: Personal use only per Terms of Service

### Multi-Asset Providers

| Provider | API Key | Free Tier | Best For |
|----------|---------|-----------|----------|
| [Databento](databento.md) | Yes | $125 credit | Institutional futures/equities |
| [Polygon/Massive](polygon.md) | Yes | 5 calls/min | US equities, options |
| [TwelveData](twelve_data.md) | Yes | 800 calls/day | Stocks, Forex, Crypto |

### Crypto Providers

| Provider | API Key | Free Tier | Best For |
|----------|---------|-----------|----------|
| [CoinGecko](coingecko.md) | No | 50 calls/min | Market overview |
| [Binance](binance.md) | No | Generous | Spot + futures trading |
| [BinancePublic](binance_public.md) | No | Unlimited | Bulk historical downloads |
| [CryptoCompare](cryptocompare.md) | Yes | 250k calls/mo | Crypto historical |

### Forex Providers

| Provider | API Key | Free Tier | Best For |
|----------|---------|-----------|----------|
| [Oanda](oanda.md) | Yes | Trial | Institutional forex |

### Economic & Factor Providers

| Provider | API Key | Free Tier | Best For |
|----------|---------|-----------|----------|
| [FRED](fred.md) | Yes | 120 req/min | Macro indicators |
| [AQR](aqr.md) | No | Free | Factor research (QMJ, BAB, TSMOM) |
| [Fama-French](fama_french.md) | No | Free | Academic factors (FF3, FF5) |

### Prediction Markets

| Provider | API Key | Free Tier | Best For |
|----------|---------|-----------|----------|
| [Kalshi](kalshi.md) | No | Free | US-regulated events |
| [Polymarket](polymarket.md) | No | Free | Crypto-based predictions |

### Tick Data & Special Purpose

| Provider | API Key | Purpose |
|----------|---------|---------|
| [NASDAQ ITCH](nasdaq_itch.md) | No | Order book tick data (sample files) |
| [Synthetic](synthetic.md) | No | Generate test data |
| [Mock](mock.md) | No | Unit testing |

---

## Quick Reference Matrix

| Provider | Minute | Hourly | Daily | Options | Fundamentals |
|----------|--------|--------|-------|---------|--------------|
| YahooFinance | ✅ (7d) | ✅ | ✅ | ❌* | ❌* |
| Databento | ✅ | ✅ | ✅ | ✅ (OPRA) | ❌ |
| Polygon/Massive | ✅ | ✅ | ✅ | ✅ | ✅ |
| EODHD | ❌ | ❌ | ✅ | ✅ ($29.99) | ✅ ($59.99) |
| Tiingo | ✅ | ✅ | ✅ | ❌ | ❌ |
| Finnhub | ✅ | ✅ | ✅ | ❌ | ✅ |
| TwelveData | ✅ | ✅ | ✅ | ❌ | ❌ |
| CoinGecko | ❌ | ❌ | ✅ | N/A | N/A |
| Binance | ✅ | ✅ | ✅ | N/A | N/A |
| CryptoCompare | ✅ | ✅ | ✅ | N/A | N/A |
| Oanda | ✅ | ✅ | ✅ | N/A | N/A |
| FRED | ❌ | ❌ | ✅ | N/A | N/A |
| Kalshi | ✅ | ✅ | ✅ | N/A | N/A |
| Polymarket | ✅ | ✅ | ✅ | N/A | N/A |

*Not yet implemented (see [Gap Analysis](PROVIDER_AUDIT.md#gap-analysis-summary))

---

## Quick Start

### Basic Usage

```python
from ml4t.data.providers import YahooFinanceProvider

# Create provider instance (reuse for best performance!)
provider = YahooFinanceProvider()

# Fetch daily OHLCV data
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-01", frequency="daily")

print(df.head())
# shape: (5, 7)
# ┌─────────────────────┬────────┬─────────┬─────────┬─────────┬─────────┬──────────┐
# │ timestamp           ┆ symbol ┆ open    ┆ high    ┆ low     ┆ close   ┆ volume   │
# │ datetime[μs]        ┆ str    ┆ f64     ┆ f64     ┆ f64     ┆ f64     ┆ f64      │
# ╞═════════════════════╪════════╪═════════╪═════════╪═════════╪═════════╪══════════╡
# │ 2024-01-02 00:00:00 ┆ AAPL   ┆ 187.15  ┆ 188.44  ┆ 183.89  ┆ 185.64  ┆ 82488700 │
# │ 2024-01-03 00:00:00 ┆ AAPL   ┆ 184.22  ┆ 185.88  ┆ 183.43  ┆ 184.25  ┆ 58414500 │
# │ ...                 ┆ ...    ┆ ...     ┆ ...     ┆ ...     ┆ ...     ┆ ...      │
# └─────────────────────┴────────┴─────────┴─────────┴─────────┴─────────┴──────────┘

provider.close()
```

### With Context Manager

```python
from ml4t.data.providers import EODHDProvider

with EODHDProvider() as provider:
    df = provider.fetch_ohlcv("AAPL.US", "2024-01-01", "2024-06-01")
    print(f"Fetched {len(df)} rows")
# Provider automatically closed
```

### Academic Factor Data

```python
from ml4t.data.providers import FamaFrenchProvider, AQRFactorProvider

# Fama-French factors
ff = FamaFrenchProvider()
ff3 = ff.fetch("ff3", frequency="monthly")  # Mkt-RF, SMB, HML
ff5 = ff.fetch("ff5", frequency="daily")    # +RMW, CMA

# AQR factors
aqr = AQRFactorProvider()
qmj = aqr.fetch("qmj_factors", region="USA")  # Quality Minus Junk
bab = aqr.fetch("bab_factors")                 # Betting Against Beta
```

### Historical Fallback Pattern

```python
from ml4t.data.providers import YahooFinanceProvider, WikiPricesProvider
import polars as pl

# Pre-2018: Wiki Prices (survivorship-bias free)
wiki = WikiPricesProvider(parquet_path="~/data/wiki_prices.parquet")
historical = wiki.fetch_ohlcv("AAPL", "1990-01-01", "2018-03-27")

# Post-2018: Yahoo Finance
yahoo = YahooFinanceProvider()
recent = yahoo.fetch_ohlcv("AAPL", "2018-03-28", "2024-12-01")

# Combine for 34+ year history
combined = pl.concat([historical, recent]).sort("timestamp")
print(f"Total history: {len(combined)} trading days")
```

---

## Provider Selection Guide

### By Use Case

| Use Case | Recommended Provider |
|----------|---------------------|
| Quick start (free) | YahooFinance |
| US equities (production) | EODHD or Polygon/Massive |
| Global equities | EODHD |
| Futures/options | Databento |
| Crypto | BinancePublic |
| Forex | Oanda |
| Academic factors | Fama-French + AQR |
| Economic indicators | FRED |
| Historical backtesting | WikiPrices |
| Prediction markets | Kalshi (US) or Polymarket |

### By Budget

| Budget | Providers |
|--------|-----------|
| $0 | Yahoo, CoinGecko, BinancePublic, WikiPrices, FRED, AQR, Fama-French |
| <$30/mo | EODHD ($19.99), Polygon Starter ($29) |
| <$100/mo | EODHD All-in-One ($99.99), Polygon Developer ($79) |
| Professional | Databento ($179+), Polygon Advanced ($199) |

---

## API Keys Setup

Create a `.env` file in your project root:

```bash
# Free tier providers
EODHD_API_KEY=your_key_here
TIINGO_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
TWELVE_DATA_API_KEY=your_key_here
CRYPTOCOMPARE_API_KEY=your_key_here
FRED_API_KEY=your_key_here

# Professional providers
POLYGON_API_KEY=your_key_here
OANDA_API_KEY=your_key_here
DATABENTO_API_KEY=your_key_here
```

---

## Storage and Incremental Updates

All providers work with the Hive storage system for persistent data and incremental updates:

```python
from ml4t.data.providers import YahooFinanceProvider
from ml4t.data.storage.hive import HiveStorage, StorageConfig

# Setup storage
storage = HiveStorage(config=StorageConfig(base_path="~/ml4t-data"))

# Fetch and store
provider = YahooFinanceProvider()
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-06-01")
storage.write(df, provider="yahoo", frequency="daily", symbol="AAPL")

# Later: incremental update (only fetches missing data)
from ml4t.data.update_manager import UpdateManager

manager = UpdateManager(storage=storage)
manager.update_symbol(
    provider_name="yahoo",
    symbol="AAPL",
    frequency="daily"
)
```

See [Incremental Updates Guide](../storage/INCREMENTAL_ARCHITECTURE.md) for detailed documentation.

---

## Data Adjustments

| Provider | Split Adjusted | Dividend Adjusted |
|----------|----------------|-------------------|
| YahooFinance | ✅ | ✅ |
| EODHD | ✅ | ✅ |
| Tiingo | ✅ | ✅ |
| WikiPrices | ✅ | ✅ |
| Databento | ✅ (futures) | N/A |
| Others | N/A (crypto/forex) | N/A |

---

## Error Handling

All providers implement consistent error handling:

```python
from ml4t.data.providers import YahooFinanceProvider
from ml4t.data.core.exceptions import (
    SymbolNotFoundError,
    RateLimitError,
    DataValidationError
)

provider = YahooFinanceProvider()

try:
    df = provider.fetch_ohlcv("INVALID_SYMBOL", "2024-01-01", "2024-06-01")
except SymbolNotFoundError:
    print("Symbol not found")
except RateLimitError:
    print("Rate limited - wait and retry")
except DataValidationError as e:
    print(f"Data validation failed: {e}")
```

---

## Performance Tips

1. **Reuse provider instances** - Creating new instances has ~30ms overhead
2. **Use Polars operations** - 10-100x faster than pandas
3. **Batch requests when possible** - Reduces API overhead
4. **Enable caching** - Some providers support response caching

```python
# Good - reuse provider (5ms per call)
provider = YahooFinanceProvider()
for symbol in symbols:
    df = provider.fetch_ohlcv(symbol, start, end)
provider.close()

# Bad - new instance per call (35ms per call)
for symbol in symbols:
    provider = YahooFinanceProvider()  # Don't do this!
    df = provider.fetch_ohlcv(symbol, start, end)
    provider.close()
```

---

## See Also

- [Provider Audit](PROVIDER_AUDIT.md) - Detailed pricing, terms, and gap analysis
- [Creating a Provider](../creating_a_provider.md) - Build custom providers
- [Storage Architecture](../storage/PARTITIONING_GUIDE.md) - Hive partitioning
- [Data Quality](../validation/) - OHLC validation and anomaly detection
