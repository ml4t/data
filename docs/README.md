# ML4T Data Documentation

Complete documentation for the ML4T Data market data library.

---

## 🚀 Getting Started

**New to ML4T Data?** Start here:

1. **[Main README](../README.md)** - Overview, installation, 5-minute quickstart
2. **[Getting Started Guide](user-guide/getting-started.md)** - Detailed setup and first steps
3. **[Provider Selection Guide](provider-selection-guide.md)** - Choose the right data provider with flowchart

---

## 📖 Asset Class Guides

**Complete guides** for each market type with provider comparisons and best practices:

- **[Asset Classes Overview](asset-classes/README.md)** - Decision trees and comparison tables
- **[Cryptocurrency Guide](asset-classes/crypto.md)** - CoinGecko, CryptoCompare (free tier available)
- **[Equities Guide](asset-classes/equities.md)** - US & Global stocks, ETFs (7 providers)
- **[Forex Guide](asset-classes/forex.md)** - OANDA, currency pairs, forex trading
- **[Futures Guide](asset-classes/futures.md)** - Databento, derivatives, institutional data

---

## 🎓 Educational Tutorials

**Step-by-step tutorials** explaining market data concepts and best practices:

- **[Tutorials Overview](tutorials/README.md)** - Learning paths and prerequisites
- **[01: Understanding OHLCV Data](tutorials/01_understanding_ohlcv.md)** - Fundamentals of market data
- **[02: Rate Limiting Best Practices](tutorials/02_rate_limiting.md)** - Avoid API bans, respect quotas
- **[03: Incremental Updates](tutorials/03_incremental_updates.md)** - 100-1000x fewer API calls
- **[04: Data Quality Validation](tutorials/04_data_quality.md)** - Ensure data integrity
- **[05: Multi-Provider Strategies](tutorials/05_multi_provider.md)** - Resilient production systems

---

## 🛠️ Contributing & Extending

**Add your own providers** or contribute to ML4T Data:

- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Code style, PR process, testing guidelines
- **[Creating a Provider](creating_a_provider.md)** - Step-by-step tutorial with Stooq example (850+ lines)
- **[Extending ML4T Data](extending_ml4t-data.md)** - Architecture and design patterns
- **[Provider Template](../provider_template/)** - Ready-to-use templates with 50+ TODOs

---

## 📚 Technical Reference

### Core Features

- **[Incremental Updates](INCREMENTAL_UPDATES.md)** - Gap detection and backfilling system
- **[Async Storage](ASYNC_STORAGE.md)** - High-performance async Parquet storage
- **[Export Guide](export_guide.md)** - Export data to CSV, Parquet, HDF5
- **[Integration Testing](INTEGRATION_TESTING.md)** - Testing real API calls

### Provider-Specific

- **[Databento Reference](providers/databento_reference.md)** - Futures and derivatives
- **[Yahoo Finance](yahoo_provider.md)** - Free stock data (legacy)
- **[Crypto Providers](crypto_providers.md)** - Cryptocurrency data sources

### API & Deployment

- **[REST API Guide](api/rest-api.md)** - FastAPI server for production
- **[API Guide](api_guide.md)** - HTTP API reference
- **[Production Deployment](deployment/production-guide.md)** - Deploy ML4T Data in production
- **[CLI Reference](user-guide/cli-reference.md)** - Command-line interface

---

## 🗂️ Documentation by Topic

### For Beginners

**Start here** if you're new to market data or ML4T Data:

1. [Main README](../README.md) - 5-minute quickstart
2. [Getting Started Guide](user-guide/getting-started.md)
3. [Tutorial 01: Understanding OHLCV](tutorials/01_understanding_ohlcv.md)
4. [Asset Classes Overview](asset-classes/README.md)

### For Developers

**Building with ML4T Data**:

1. [Provider Selection Guide](provider-selection-guide.md)
2. [Tutorial 02: Rate Limiting](tutorials/02_rate_limiting.md)
3. [Tutorial 03: Incremental Updates](tutorials/03_incremental_updates.md)
4. [Tutorial 04: Data Quality](tutorials/04_data_quality.md)
5. [Tutorial 05: Multi-Provider](tutorials/05_multi_provider.md)

### For Contributors

**Extending ML4T Data**:

1. [CONTRIBUTING.md](../CONTRIBUTING.md)
2. [Creating a Provider](creating_a_provider.md)
3. [Extending ML4T Data](extending_ml4t-data.md)
4. [Provider Template](../provider_template/)
5. [Integration Testing](INTEGRATION_TESTING.md)

### For Production Systems

**Deploy ML4T Data at scale**:

1. [Incremental Updates](INCREMENTAL_UPDATES.md)
2. [Async Storage](ASYNC_STORAGE.md)
3. [REST API Guide](api/rest-api.md)
4. [Production Deployment](deployment/production-guide.md)
5. [Tutorial 05: Multi-Provider](tutorials/05_multi_provider.md)

---

## 📊 Provider Comparison

Quick comparison of all 14 providers:

| Provider | Crypto | Stocks | Forex | Futures | Free Tier | Best For |
|----------|--------|--------|-------|---------|-----------|----------|
| **CoinGecko** | ✅ | ❌ | ❌ | ❌ | 50/min | Crypto (beginners) |
| **CryptoCompare** | ✅ | ❌ | ❌ | ❌ | 100K/mo | Crypto (advanced) |
| **Tiingo** | ❌ | ✅ US | ❌ | ❌ | 1000/day | US stocks |
| **IEX Cloud** | ❌ | ✅ US | ❌ | ❌ | 50K msgs | US stocks + fundamentals |
| **Alpha Vantage** | ⚠️ | ✅ | ⚠️ | ❌ | 25/day | Low-volume research |
| **EODHD** | ❌ | ✅ Global | ❌ | ❌ | 500/day | Global (best value) |
| **Finnhub** | ⚠️ | ✅ Global | ⚠️ | ❌ | Real-time | Professional stocks |
| **OANDA** | ❌ | ❌ | ✅ | ❌ | Demo | Professional forex |
| **Twelve Data** | ✅ | ✅ | ✅ | ❌ | 800/day | Multi-asset |
| **Polygon** | ✅ | ✅ | ✅ | ✅ | 5/min | Multi-asset (paid) |
| **Databento** | ❌ | ⚠️ | ❌ | ✅ | None | Institutional futures |
| **Yahoo** | ❌ | ✅ | ❌ | ❌ | Unlimited | Free stocks (unreliable) |
| **Binance** | ✅ | ❌ | ❌ | ✅ | 1200/min | Crypto exchange data |

**See [Asset Classes Overview](asset-classes/README.md) for detailed comparison and recommendations.**

---

## 🎯 Common Tasks

### Get Started with Crypto

```bash
# No API key needed!
pip install ml4t-data
```

```python
from ml4t.data.providers import CoinGeckoProvider

provider = CoinGeckoProvider()
btc = provider.fetch_ohlcv("BTC", "2024-01-01", "2024-12-31")
```

**Next**: [Cryptocurrency Guide](asset-classes/crypto.md)

### Get Started with US Stocks

```python
from ml4t.data.providers import TiingoProvider

provider = TiingoProvider(api_key="your_key")
aapl = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")
```

**Next**: [Equities Guide](asset-classes/equities.md)

### Get Started with Forex

```python
from ml4t.data.providers import OandaProvider

provider = OandaProvider(api_key="your_key", account_type="practice")
eurusd = provider.fetch_ohlcv("EUR_USD", "2024-01-01", "2024-12-31", frequency="H1")
```

**Next**: [Forex Guide](asset-classes/forex.md)

### Get Started with Futures

```python
from ml4t.data.providers import DatabentoProvider

provider = DatabentoProvider(api_key="your_key", dataset="GLBX.MDP3")
es = provider.fetch_ohlcv("ES.v.0", "2024-01-01", "2024-12-31")  # S&P 500 futures
```

**Next**: [Futures Guide](asset-classes/futures.md)

---

## 🔍 Find Documentation By...

### By Asset Class

- [Cryptocurrency](asset-classes/crypto.md)
- [Equities (Stocks)](asset-classes/equities.md)
- [Forex (FX)](asset-classes/forex.md)
- [Futures & Options](asset-classes/futures.md)

### By Provider

- **Crypto**: [CoinGecko](asset-classes/crypto.md#coingecko-recommended-for-beginners), [CryptoCompare](asset-classes/crypto.md#cryptocompare-advanced-features)
- **US Stocks**: [Tiingo](asset-classes/equities.md#tiingo-recommended-for-us-stocks), [IEX Cloud](asset-classes/equities.md#iex-cloud-fundamentals--ohlcv), [Alpha Vantage](asset-classes/equities.md#alpha-vantage-low-volume-research)
- **Global Stocks**: [EODHD](asset-classes/equities.md#eodhd-best-value-for-global-coverage), [Finnhub](asset-classes/equities.md#finnhub-professional-grade---paid-for-historical)
- **Forex**: [OANDA](asset-classes/forex.md#oanda-recommended-for-forex)
- **Futures**: [Databento](asset-classes/futures.md#databento-recommended-for-futures)
- **Multi-Asset**: [Twelve Data](asset-classes/equities.md#twelve-data-multi-asset-with-forex), [Polygon](asset-classes/equities.md#polygon-multi-asset-alternative)

### By Concept

- **OHLCV Basics**: [Tutorial 01](tutorials/01_understanding_ohlcv.md)
- **Rate Limiting**: [Tutorial 02](tutorials/02_rate_limiting.md)
- **Incremental Updates**: [Tutorial 03](tutorials/03_incremental_updates.md), [INCREMENTAL_UPDATES.md](INCREMENTAL_UPDATES.md)
- **Data Quality**: [Tutorial 04](tutorials/04_data_quality.md)
- **Multi-Provider**: [Tutorial 05](tutorials/05_multi_provider.md)
- **Storage**: [ASYNC_STORAGE.md](ASYNC_STORAGE.md)
- **API**: [REST API Guide](api/rest-api.md), [API Guide](api_guide.md)

### By Task

- **Adding a provider**: [Creating a Provider](creating_a_provider.md)
- **Understanding architecture**: [Extending ML4T Data](extending_ml4t-data.md)
- **Writing tests**: [Integration Testing](INTEGRATION_TESTING.md)
- **Deploying**: [Production Guide](deployment/production-guide.md)
- **Exporting data**: [Export Guide](export_guide.md)
- **Using CLI**: [CLI Reference](user-guide/cli-reference.md)

---

## 💡 Tips for Using This Documentation

### Documentation Structure

```
docs/
├── README.md                    # You are here - documentation index
├── asset-classes/               # Market-specific guides (crypto, stocks, forex, futures)
├── tutorials/                   # Step-by-step educational content
├── user-guide/                  # Getting started, CLI reference
├── api/                         # REST API documentation
├── deployment/                  # Production deployment guides
├── providers/                   # Provider-specific references
├── creating_a_provider.md       # Tutorial for adding providers
├── extending_ml4t-data.md           # Architecture documentation
├── provider-selection-guide.md  # Choose the right provider
└── [technical docs]             # Incremental updates, async storage, etc.
```

### How to Navigate

1. **New users**: Start with [Main README](../README.md) → [Asset Classes](asset-classes/README.md)
2. **Developers**: [Provider Selection](provider-selection-guide.md) → [Tutorials](tutorials/README.md)
3. **Contributors**: [CONTRIBUTING](../CONTRIBUTING.md) → [Creating a Provider](creating_a_provider.md)
4. **Production**: [Tutorials](tutorials/README.md) → [Production Guide](deployment/production-guide.md)

### Cross-References

Throughout the documentation, you'll find:
- 📖 Links to related guides
- ✅ Prerequisites and next steps
- ⚠️ Common gotchas and warnings
- 💡 Pro tips and best practices

---

## 📞 Getting Help

- **Questions**: Open a [GitHub Issue](https://github.com/quantlab/ml4t-data-python/issues)
- **Bugs**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for bug reports
- **Feature Requests**: Use GitHub Discussions
- **Security**: See SECURITY.md for vulnerability reporting

---

## 📝 Documentation Contributions

Found a typo? Want to improve a guide? See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Documentation style guide
- How to submit improvements
- Writing tutorials and examples

---

**Last Updated**: 2025-11-04
**Documentation Version**: 1.0 (Work Unit 003 complete)
