# ML4T Data - High-Performance Market Data Management Library

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/ml4t-data)](https://pypi.org/project/ml4t-data/)
[![Tests](https://github.com/ml4t/data/actions/workflows/ml4t-data.yml/badge.svg)](https://github.com/ml4t/data/actions)
[![Coverage](https://codecov.io/gh/ml4t/data/branch/main/graph/badge.svg)](https://codecov.io/gh/ml4t/data)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ML4T Data** is a production-ready market data acquisition and management system providing unified access to **19 data providers** with enterprise-grade reliability and **automated incremental updates**. Get OHLCV data from crypto, equities, forex, futures, and factor data in 3 lines of code.

## 5-Minute Quickstart

```python
# Install
pip install ml4t-data

# Get Bitcoin historical data (no API key needed!)
from ml4t.data.providers import CoinGeckoProvider

provider = CoinGeckoProvider()
btc_data = provider.fetch_ohlcv("bitcoin", "2024-01-01", "2024-12-31")

# That's it! You have a Polars DataFrame with OHLCV data
print(btc_data.head())
```

**Want more?** See [Choose Your Provider](#choose-your-provider) to find the best data source for your needs.

## Quick Start for ML4T Book Readers

Reading **Machine Learning for Trading (Third Edition)**? This library is your reference implementation for market data acquisition throughout the book.

### 30-Second Setup

```bash
# Install
pip install ml4t-data

# Download 10 years of S&P 500 data (free, no API key!)
ml4t-data fetch AAPL MSFT GOOGL NVDA META \
  --start 2015-01-01 \
  --provider yahoo \
  --output ~/ml4t-book/data
```

That's it! You now have production-grade OHLCV data ready for Chapter 3 examples.

### Three-Tier Learning Path

The book follows a natural progression from free data to professional infrastructure:

#### **Tier 1: Free Data** (Chapters 1-5)
Start learning with zero cost:
- **Yahoo Finance**: Unlimited US stocks, no API key
- **CoinGecko**: 10,000+ cryptocurrencies
- **Wiki Prices**: Historical S&P 500 dataset (fallback)

**Use for**: Learning fundamentals, running first models, backtesting basics

#### **Tier 2: Educational** (Chapters 6-12)
Scale up with free credits and generous limits:
- **AlgoSeek**: FREE NASDAQ 100 minute bars (2015-2017, 5GB)
- **EODHD**: 500 calls/day free (perfect for daily updates)
- **DataBento**: $10 free credits for new users
- **Tiingo**: 1000 calls/day free

**Use for**: Intraday analysis, global markets, production-scale projects

#### **Tier 3: Professional** (Advanced Chapters)
Production trading infrastructure:
- **DataBento**: $9+/month for institutional data
- **Polygon**: $99+/month for real-time feeds
- **Finnhub**: $60+/month for global coverage

**Use for**: Live trading, high-frequency strategies, institutional workflows

### Book-Wide Data Configuration

All book chapters reference a central config:

```yaml
# ~/ml4t-book/ml4t-data.yaml
storage:
  path: ~/ml4t-book/data

datasets:
  # Chapter 3-5: Free tier
  sp500_daily:
    provider: yahoo
    symbols_file: config/symbols/sp500.txt
    frequency: daily
    start_date: 2015-01-01

  # Chapter 6+: Educational tier
  nasdaq100_minute:
    provider: algoseek  # Download from algoseek.com
    frequency: minute
```

Run daily updates:
```bash
ml4t-data update-all -c ~/ml4t-book/ml4t-data.yaml
```

### Progressive Scaling

The book teaches you to scale from 5 symbols to 5000+:

1. **Chapter 3**: 5 tech stocks → Learn the basics
2. **Chapter 4**: 55 stocks → File-based symbol loading
3. **Chapter 5**: 500 stocks (S&P 500) → Production scale
4. **Chapter 6+**: Minute data, futures, global markets

### Why ml4t-data for the Book?

**Compared to DIY notebooks** (2nd edition approach):
- ✅ 10x less code (unified API vs 10+ provider notebooks)
- ✅ Production-ready (OHLC validation, gap detection, incremental updates)
- ✅ 10-100x faster (Polars vs pandas)
- ✅ No manual orchestration (automated updates via CLI)

**Compared to native SDKs**:
- ✅ One interface for 19 providers (easy to compare data quality)
- ✅ Free tier friendly (Yahoo + EODHD get you far)
- ✅ Built-in storage (Hive partitioning, not scattered CSVs)
- ✅ Book examples work out-of-the-box

### Learning Progression Guide

This guide shows how the book progressively builds your data engineering skills, from fetching your first stock quote to managing production-scale data pipelines.

#### Phase 1: Foundations (Chapters 1-5)

**Learning Objective**: Master the fundamentals with zero-cost data

| Chapter | Data Scale | Providers | Key Skills |
|---------|-----------|-----------|------------|
| **Ch 3: Data Universe** | 5 stocks | Yahoo Finance, CoinGecko | Provider comparison, OHLC validation, storage patterns |
| **Ch 4: Alpha Factors** | 55 stocks | Yahoo Finance | File-based symbol loading, feature engineering on stored data |
| **Ch 5: Portfolio Optimization** | 500 stocks (S&P 500) | Yahoo + EODHD | Production scale, DuckDB querying, Hive partitions |

**Data Tier**: Free (no API keys, no limits)
**Storage**: ~50MB-500MB on disk
**Time Investment**: 3-5 hours to complete setup

**Chapter 3 Notebooks** (6 hands-on tutorials):
1. **Free Data Quickstart** - Fetch your first OHLCV data (5 stocks, Yahoo)
2. **Data Quality** - OHLC validation, anomaly detection, deduplication
3. **Scaling to S&P 500** - File-based symbols, batch downloads, 500 stocks
4. **Intraday Minute Data** - AlgoSeek ingestion (2015-2017 NASDAQ 100)
5. **Storage & Automation** - Hive partitioning, incremental updates, cron jobs
6. **DuckDB Querying** - Zero-copy analytics, time-series operations

**Outcome**: You can fetch, validate, store, and query production-scale daily OHLCV data.

#### Phase 2: Scaling Up (Chapters 6-10)

**Learning Objective**: Add intraday data and global markets

| Chapter | Data Scale | Providers | Key Skills |
|---------|-----------|-----------|------------|
| **Ch 6: ML Preprocessing** | S&P 500 minute | AlgoSeek (static) | Intraday patterns, minute bar analysis |
| **Ch 7: Linear Models** | S&P 500 daily | Yahoo, EODHD | Feature engineering at scale, train/test splits |
| **Ch 8: Time Series** | Multi-asset | Yahoo, CoinGecko | Cross-asset correlation, regime detection |
| **Ch 9: Bayesian ML** | S&P 500 + ETFs | Yahoo, EODHD | Hierarchical data (stocks + sectors) |
| **Ch 10: Decision Trees** | S&P 500 daily | Yahoo, EODHD | High-dimensional features, tree-based models |

**Data Tier**: Free + Educational (AlgoSeek 5GB static, EODHD 500/day)
**Storage**: ~5GB-20GB on disk
**API Keys Needed**: EODHD (free tier), optional DataBento credits

**Outcome**: You can work with minute bars, manage multi-gigabyte datasets, and orchestrate daily updates.

#### Phase 3: Professional Infrastructure (Chapters 11-16)

**Learning Objective**: Build production trading pipelines

| Chapter | Data Scale | Providers | Key Skills |
|---------|-----------|-----------|------------|
| **Ch 11: Neural Networks** | S&P 500 minute | DataBento trial | Real-time features, streaming data |
| **Ch 12: Deep Learning** | Multi-asset minute | DataBento, EODHD | GPU-friendly storage, large-batch training |
| **Ch 13: Transformers** | Time-series sequences | DataBento | Windowed data, sequence prediction |
| **Ch 14: NLP** | Equities + news | Finnhub, Polygon | Multi-modal data, sentiment features |
| **Ch 15: Reinforcement Learning** | Futures contracts | DataBento | Continuous contracts, roll dates |
| **Ch 16: Portfolio Management** | Multi-asset live | DataBento, Polygon | Real-time updates, production deployment |

**Data Tier**: Professional (paid subscriptions)
**Storage**: 50GB+ on disk
**API Keys Needed**: DataBento ($9+/mo), optional Polygon/Finnhub

**Outcome**: You have production-grade infrastructure for live trading systems.

#### Provider Maturity Path

As you progress through the book, your data needs evolve:

```
Chapters 1-5:  Yahoo Finance (free, daily, US stocks)
               └─> Learn: fetch, validate, store

Chapters 6-10: Yahoo + EODHD (500/day free tier)
               └─> Add: global markets, scheduled updates

Chapters 11-13: DataBento trial ($10 credits)
                └─> Add: minute bars, institutional quality

Chapters 14-16: DataBento + Polygon/Finnhub (paid)
                └─> Add: real-time, multi-asset, derivatives
```

**Decision Point** (End of Chapter 10):
- **Academic/Learning**: Stay on free tier (Yahoo + EODHD covers 80% of use cases)
- **Professional Trading**: Upgrade to DataBento ($9/mo for basic, $49/mo for comprehensive)

#### Symbol Scaling Roadmap

Learn to manage datasets of increasing complexity:

| Scale | Symbols | Storage | Update Time | Chapters |
|-------|---------|---------|-------------|----------|
| **Starter** | 5 stocks | 5MB | 10 seconds | 3 |
| **Portfolio** | 55 stocks | 50MB | 1 minute | 4 |
| **S&P 500** | 500 stocks | 500MB | 10 minutes | 5-10 |
| **Russell 3000** | 3000 stocks | 3GB | 60 minutes | 11+ |
| **Global** | 10,000+ stocks | 20GB+ | Hours | 14+ |

**Best Practice**: Start small (5 stocks), validate your workflow, then scale up. Don't download S&P 500 on day one.

#### Common Learning Paths

**Path 1: Academic / Book Learning** (Recommended for most readers)
```
Ch 3 → Free tier (Yahoo)
Ch 4-5 → Add EODHD (500/day free)
Ch 6-10 → AlgoSeek (static 5GB dataset)
Ch 11+ → Optional: DataBento trial ($10 credits)
```
**Total Cost**: $0 (covers 95% of book exercises)

**Path 2: Professional / Live Trading**
```
Ch 3-5 → Free tier (learn fundamentals)
Ch 6-10 → EODHD paid (€19.99/mo for unlimited)
Ch 11+ → DataBento ($49/mo for comprehensive)
```
**Total Cost**: ~$70/month (production-ready infrastructure)

**Path 3: Hybrid** (Best value)
```
Ch 3-10 → Free tier (Yahoo + EODHD 500/day)
Ch 11+ → DataBento when you need minute bars or derivatives
```
**Total Cost**: $0 until Ch 11, then $9-49/mo

#### Getting Stuck? Start Here

**If you're on Chapter 3**: Use [examples/book_chapter3.py](examples/book_chapter3.py) - 5 tech stocks, Yahoo Finance, zero setup

**If you're on Chapter 5+**: Use [examples/sp500_pipeline.yaml](examples/sp500_pipeline.yaml) - S&P 500 daily updates

**If you need help**: See [Book Forum](https://github.com/stefan-jansen/ml4t/discussions) for questions and community support

### CLI Workflows for Book Readers

The `ml4t-data` CLI is designed for configuration-driven workflows. Define your datasets once in YAML, then run one command for all updates.

#### Essential Commands

**1. Fetch Initial Data** (one-time setup)
```bash
# Fetch 5 tech stocks for Chapter 3 (no config needed)
ml4t-data fetch AAPL MSFT GOOGL NVDA META \
  --start 2015-01-01 \
  --provider yahoo \
  --output ~/ml4t-book/data

# Verify data was saved
ls -lh ~/ml4t-book/data/yahoo/daily/
```

**2. Configuration-Driven Updates** (recommended for Chapter 4+)
```bash
# Create config file once (see Book-Wide Data Configuration above)
# Then run daily updates:
ml4t-data update-all -c ~/ml4t-book/ml4t-data.yaml

# Dry-run to preview what will update
ml4t-data update-all -c ~/ml4t-book/ml4t-data.yaml --dry-run

# Update specific dataset only
ml4t-data update-all -c ~/ml4t-book/ml4t-data.yaml --dataset sp500_daily
```

**3. List Stored Data**
```bash
# See what data you have
ml4t-data list -c ~/ml4t-book/ml4t-data.yaml

# Example output:
# Dataset: sp500_daily
#   Provider: yahoo
#   Symbols: 503 (from sp500.txt)
#   Last updated: 2025-11-24 09:15:00
#   Storage: 487MB (Hive partitioned)
```

**4. Inspect Specific Symbol**
```bash
# Check when AAPL was last updated
ml4t-data info AAPL --config ~/ml4t-book/ml4t-data.yaml

# Output: symbol, provider, last_date, row_count, file_size
```

#### Automated Daily Updates (Cron)

Set up once, get fresh data every morning before market open.

**Step 1: Test your config**
```bash
# Dry-run to verify config is correct
ml4t-data update-all -c ~/ml4t-book/ml4t-data.yaml --dry-run
```

**Step 2: Create update script**
```bash
# ~/ml4t-book/scripts/update_data.sh
#!/bin/bash
set -e

# Load API keys (if needed)
export EODHD_API_KEY="your_key_here"

# Run update with logging
ml4t-data update-all \
  -c ~/ml4t-book/ml4t-data.yaml \
  >> ~/ml4t-book/logs/data_updates.log 2>&1

# Optional: Send notification on failure
if [ $? -ne 0 ]; then
  echo "Data update failed at $(date)" | mail -s "ML4T Data Alert" you@example.com
fi
```

**Step 3: Make executable and test**
```bash
chmod +x ~/ml4t-book/scripts/update_data.sh
~/ml4t-book/scripts/update_data.sh  # Test it works
```

**Step 4: Schedule with cron**
```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 6 AM):
0 6 * * * /home/yourusername/ml4t-book/scripts/update_data.sh

# For EODHD (market close is ~4 PM ET, wait until 5 PM):
0 17 * * 1-5 /home/yourusername/ml4t-book/scripts/update_data.sh
```

**Cron Schedule Examples**:
- `0 6 * * *` - Every day at 6 AM
- `0 17 * * 1-5` - Weekdays at 5 PM (after US market close)
- `0 */6 * * *` - Every 6 hours
- `0 0 * * 0` - Weekly on Sunday midnight

#### API Key Management

**Best Practice: Environment Variables**
```bash
# Add to ~/.bashrc or ~/.zshrc
export EODHD_API_KEY="your_key_here"
export DATABENTO_API_KEY="your_key_here"
export FINNHUB_API_KEY="your_key_here"

# Reload shell
source ~/.bashrc
```

**Alternative: .env File**
```bash
# Create ~/ml4t-book/.env
EODHD_API_KEY=your_key_here
DATABENTO_API_KEY=your_key_here

# Load in your script
set -a
source ~/ml4t-book/.env
set +a

ml4t-data update-all -c ~/ml4t-book/ml4t-data.yaml
```

**Security Best Practices**:
- ✅ Never commit `.env` files to git (add to `.gitignore`)
- ✅ Use environment variables for production
- ✅ Rotate keys if exposed
- ✅ Use separate keys for dev/production
- ❌ Don't hardcode keys in YAML configs
- ❌ Don't share keys in screenshots or logs

#### Common Issues & Solutions

**Issue: "Symbol not found" error**
```bash
# Symptom: Provider returns 404 or "symbol not found"
# Cause: Symbol delisted, wrong format, or provider doesn't have it

# Solution 1: Check symbol format
# Yahoo: AAPL (US stocks), BRK-B (not BRK.B)
# EODHD: AAPL.US (must include exchange)
# Binance: BTCUSDT (not BTC-USD)

# Solution 2: Try different provider
ml4t-data fetch AAPL --provider yahoo      # Works
ml4t-data fetch AAPL.US --provider eodhd   # Also works

# Solution 3: Check if symbol was delisted
# Remove from your symbol file if no longer trading
```

**Issue: Rate limit exceeded**
```bash
# Symptom: "Rate limit exceeded, retry in X seconds"
# Cause: Too many requests to provider

# Solution 1: Reduce batch size
# In YAML config, add smaller symbol files or reduce parallelism

# Solution 2: Use different provider
# Yahoo has no rate limits, EODHD allows 500/day free

# Solution 3: Upgrade to paid tier
# EODHD: €19.99/mo for unlimited calls
```

**Issue: Missing data / gaps**
```bash
# Symptom: Some dates missing in your data
# Cause: Provider outage, API changes, weekends/holidays

# Solution: Use --detect-gaps flag
ml4t-data update-all -c ~/ml4t-book/ml4t-data.yaml --detect-gaps

# This will identify gaps and attempt backfill
```

**Issue: Slow updates**
```bash
# Symptom: Updates taking too long (>10 min for S&P 500)
# Cause: Too many symbols, slow provider, or network issues

# Solution 1: Check what's slow
ml4t-data update-all -c ~/ml4t-book/ml4t-data.yaml --verbose

# Solution 2: Incremental updates (only new data)
# Already default behavior, but verify in config:
# update_mode: INCREMENTAL

# Solution 3: Parallel updates (advanced)
# Add to config:
# max_workers: 5  # Fetch 5 symbols in parallel
```

**Issue: Storage growing too large**
```bash
# Symptom: Data directory is 50GB+
# Cause: Too much history or too many symbols

# Solution 1: Check storage usage
du -sh ~/ml4t-book/data/*

# Solution 2: Clean old data
ml4t-data clean --before 2020-01-01 -c ~/ml4t-book/ml4t-data.yaml

# Solution 3: Reduce symbols
# Edit your symbol files to focus on active portfolios
```

**Issue: "Config file not found"**
```bash
# Symptom: ml4t-data can't find your YAML
# Cause: Wrong path or file not created

# Solution 1: Use absolute paths
ml4t-data update-all -c /home/you/ml4t-book/ml4t-data.yaml

# Solution 2: Set default config location
export ML4T_DATA_CONFIG=~/ml4t-book/ml4t-data.yaml
ml4t-data update-all  # Now uses default

# Solution 3: Navigate to config directory
cd ~/ml4t-book
ml4t-data update-all -c ml4t-data.yaml
```

#### Quick Reference

| Task | Command |
|------|---------|
| Fetch initial data | `ml4t-data fetch AAPL MSFT --provider yahoo` |
| Daily updates | `ml4t-data update-all -c config.yaml` |
| Preview changes | `ml4t-data update-all -c config.yaml --dry-run` |
| Update one dataset | `ml4t-data update-all -c config.yaml --dataset sp500` |
| List data | `ml4t-data list -c config.yaml` |
| Check symbol | `ml4t-data info AAPL -c config.yaml` |
| Detect gaps | `ml4t-data update-all -c config.yaml --detect-gaps` |
| Clean old data | `ml4t-data clean --before 2020-01-01 -c config.yaml` |
| Verbose logging | `ml4t-data update-all -c config.yaml --verbose` |

**Environment Variables**:
- `ML4T_DATA_CONFIG` - Default config file path
- `ML4T_DATA_LOG_LEVEL` - Log verbosity (DEBUG, INFO, WARNING, ERROR)
- `<PROVIDER>_API_KEY` - API keys (EODHD_API_KEY, DATABENTO_API_KEY, etc.)

### Get Help

- **Documentation**: Full guides at [docs/](docs/)
- **Examples**: Ready-to-run scripts in [examples/](examples/)
- **Issues**: Report problems at [GitHub Issues](https://github.com/stefan-jansen/ml4t-data/issues)
- **Book Forum**: Discuss with other readers at [ML4T Community](https://github.com/stefan-jansen/ml4t/discussions)

---

## When to Use This Library

### ✅ Perfect For

**Production Data Pipelines:**
```bash
# Set up automated daily updates via cron
# crontab: 0 9 * * * ml4t-data update-all -c ~/ml4t-data.yaml

ml4t-data update-all -c ml4t-data.yaml
```
- **Incremental updates**: Fetch only new data since last update
- **Gap detection**: Automatically identify and fill missing dates
- **Cron-friendly**: CLI designed for automated scheduled updates
- **Multi-dataset management**: Update equities, crypto, futures in one command

**Cross-Provider Workflows:**
```python
# Same code, different data sources
providers = {
    'yahoo': YahooFinanceProvider(),
    'databento': DataBentoProvider(api_key),
    'eodhd': EODHDProvider(api_key)
}

for name, provider in providers.items():
    data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")
    analyze_quality(data, name)
```
- **Unified interface**: `fetch_ohlcv()` works across all 12 providers
- **Data quality comparison**: Test multiple sources with same code
- **Provider fallback**: Switch providers without code changes

**Backtesting & Research:**
- **Polars DataFrames**: 10-100x faster than pandas alternatives
- **OHLC validation**: Catch data errors before they corrupt backtest results
- **Hive-partitioned storage**: Efficient access to years of historical data
- **Metadata tracking**: Know exactly when data was last updated

### ⚠️ Consider Native SDKs When

**You Need Advanced Provider Features:**
- **Databento**: Trades, MBO, quotes, symbology API, streaming
- **Interactive Brokers**: Real-time tick data, order execution
- **Specialized endpoints**: Fundamentals, news, options chains

**Performance is Critical:**
- Native SDKs have 5-20% less overhead
- For sub-second latency requirements, use direct connections

**You're Doing One-Off Analysis:**
```python
# For quick notebook exploration, native yfinance is fine
import yfinance as yf
df = yf.download("AAPL", "2020-01-01", "2024-12-31")
```

**You Need Provider-Specific Data Formats:**
- Some providers return custom objects (e.g., Databento's DBN format)
- Our wrappers standardize to OHLCV, which may lose information

### 💡 Hybrid Approach (Recommended)

Use ml4t-data for **data pipeline infrastructure**, native SDKs for **advanced features**:

```python
# Daily updates via ml4t-data
# cron: ml4t-data update-all -c config.yaml

# Analysis with native SDK for advanced features
import databento as db
client = db.Historical(api_key)
trades = client.timeseries.get_range(
    dataset='GLBX.MDP3',
    symbols=['ES.FUT'],
    schema='trades',  # Not available via our wrapper
    start='2024-01-01'
)
```

**Best of both worlds:**
- Automated infrastructure (ml4t-data)
- Full provider capabilities (native SDKs)

## Why ML4T Data?

- 🤖 **Automated Updates**: Cron-friendly CLI for incremental updates, gap detection, and backfilling
- 🚀 **19 Data Providers**: Crypto, equities, forex, futures, economic, factor data - all with consistent API
- ⚡ **10-100x Faster**: Polars-based processing beats pandas-based alternatives
- 🛡️ **Production-Ready**: Circuit breakers, rate limiting, automatic retries
- 💾 **Smart Storage**: Hive-partitioned Parquet with metadata tracking
- 🔍 **Data Quality**: Automatic validation of OHLC invariants and deduplication
- 🎯 **Type-Safe**: Full type hints, mypy strict mode
- 📚 **Well-Documented**: Step-by-step guides, examples, and templates

## Choose Your Provider

Not sure which data source to use? Here's a comprehensive guide to help you select the right provider based on your budget, asset class, and data needs.

### Quick Tier Overview

| Tier | Providers | Book Tier | Best For | Monthly Cost | Cost/Benefit |
|------|-----------|-----------|----------|--------------|--------------|
| **Free (Generous)** | Yahoo, CoinGecko, FRED, Fama-French, AQR | 🟢 Tier 1 | Learning fundamentals, factor models, macro research | $0 | ⭐⭐⭐⭐⭐ Perfect for Chapters 1-5 |
| **Free (Limited)** | EODHD, Tiingo, Twelve Data, Kalshi | 🟡 Tier 2 | Daily updates, S&P 500 scale, global markets | $0 | ⭐⭐⭐⭐ Great for Chapters 6-10 |
| **Free (Historical)** | Wiki Prices, NASDAQ ITCH | 🟢 Tier 1 | Survivorship-bias analysis, microstructure | $0 | ⭐⭐⭐⭐⭐ Static datasets |
| **Educational** | AlgoSeek (static), DataBento (credits) | 🟡 Tier 2 | Intraday analysis, minute data, learning production patterns | $0-10 | ⭐⭐⭐⭐ Educational tier with credits |
| **Paid (Affordable)** | EODHD, Tiingo, Twelve Data | 🔵 Tier 3 | Production pipelines, unlimited calls, extended history | $10-30 | ⭐⭐⭐ Good value for serious projects |
| **Professional** | Databento, Polygon, Finnhub | 🔵 Tier 3 | Live trading, institutional data, high-frequency | $60+ | ⭐⭐ Best quality, premium price |

### Provider Pricing Breakdown

#### Free Tier Providers (No API Key Required)

**Yahoo Finance** - Best for US/Global Stocks 🟢 **Book Tier 1**
- **Rate Limit**: ~2000 requests/hour (undocumented)
- **Coverage**: Stocks, ETFs, indices, crypto, forex
- **Free Tier**: Unlimited historical data
- **Symbol Format**: Standard ticker (AAPL, BTC-USD, EURUSD=X)
- **Book Use Cases**:
  - ✅ Chapter 3-5: Learn data fundamentals with S&P 500
  - ✅ Chapter 3: Provider comparison notebooks
  - ✅ Chapter 4: Alpha factor engineering
  - ✅ Chapter 5: Portfolio optimization examples
- **Best For**: Backtesting, research, learning fundamentals
- **Limitations**: No official API (may break), no support
- **Cost/Benefit**: ⭐⭐⭐⭐⭐ Perfect for getting started
- **[Get Started](https://finance.yahoo.com)** (no registration needed)

**CoinGecko** - Best for Crypto 🟢 **Book Tier 1**
- **Rate Limit**: 10-50 calls/min (free), 500 calls/min (paid $129/mo)
- **Coverage**: 10,000+ cryptocurrencies
- **Free Tier**: Unlimited historical data, no API key needed
- **Symbol Format**: Coin ID (bitcoin, ethereum, binancecoin)
- **Book Use Cases**:
  - ✅ Chapter 3: Crypto data quality comparison
  - ✅ Chapter 8+: Cryptocurrency strategy backtesting
  - ✅ Advanced: Multi-asset portfolio allocation
- **Best For**: Crypto historical analysis, portfolio tracking, learning crypto markets
- **Limitations**: No real-time WebSocket on free tier
- **Cost/Benefit**: ⭐⭐⭐⭐⭐ Best free crypto data
- **[Get Started](https://www.coingecko.com/en/api/pricing)** (free tier available)

#### Free Tier Providers (API Key Required)

**EODHD** - Best Value for Global Coverage 🟡 **Book Tier 2**
- **Free Tier**: 500 API calls/day, 1-year historical depth
- **Paid Tier**: €19.99/month, unlimited calls, 30+ years history
- **Coverage**: 150,000+ tickers across 60+ global exchanges
- **Symbol Format**: TICKER.EXCHANGE (AAPL.US, VOD.LSE, BMW.FRA)
- **Rate Limit**: No rate limit on paid tier
- **Book Use Cases**:
  - ✅ Chapter 6+: Scale to daily S&P 500 updates (500 calls = 500 symbols/day)
  - ✅ Chapter 7: Global market strategies (LSE, FRA, TSE)
  - ✅ Chapter 9: Multi-market portfolio construction
  - ✅ Perfect for production: Upgrade to paid for unlimited
- **Best For**: Global stocks, best price/coverage ratio, production scaling
- **Cost/Benefit**: ⭐⭐⭐⭐ Best value for scaling beyond US markets
- **[Get API Key](https://eodhd.com/register)** | **[Pricing](https://eodhd.com/pricing)**

**Tiingo** - Best for High-Quality US Data 🟡 **Book Tier 2**
- **Free Tier**: 1000 calls/day, 500 unique symbols/month
- **Paid Tier**: $30/month, 20,000 calls/hour
- **Coverage**: US stocks, crypto, IEX real-time
- **Symbol Format**: Standard ticker (AAPL)
- **Rate Limit**: 1000/day (free), 20K/hour (paid)
- **Book Use Cases**:
  - ✅ Chapter 6+: High-quality daily S&P 500 updates
  - ✅ Chapter 7: Data quality validation (compare with Yahoo)
  - ✅ Chapter 10: Real-time strategy testing (IEX integration)
  - ✅ Excellent for: Thesis/research projects requiring data quality
- **Best For**: Daily updates, research workflows, high-quality backtesting
- **Cost/Benefit**: ⭐⭐⭐⭐ Premium quality at free tier scale
- **[Get API Key](https://www.tiingo.com/account/api/token)** | **[Pricing](https://www.tiingo.com/about/pricing)**

**IEX Cloud** - Best for Fundamentals
- **Free Tier**: 50,000 message credits/month
- **Paid Tiers**: $0-999/month (usage-based)
- **Coverage**: US stocks, fundamentals, news
- **Symbol Format**: Standard ticker (AAPL)
- **Rate Limit**: Based on message credits (1 OHLCV bar ≈ 1 credit)
- **Best For**: Fundamental data, company info, news
- **[Get API Key](https://iexcloud.io/cloud-login#/register)** | **[Pricing](https://iexcloud.io/pricing/)**

**Alpha Vantage** - Best for Conservative Research
- **Free Tier**: 25 API calls/day, 5 calls/minute
- **Paid Tier**: $49.99/month, 75 calls/minute
- **Coverage**: US/global stocks, forex, crypto, fundamentals
- **Symbol Format**: Standard ticker (AAPL)
- **Rate Limit**: 5/min, 25/day (free) | 75/min (paid)
- **Best For**: Multi-asset research with low call volume
- **Limitations**: Very restrictive free tier
- **[Get API Key](https://www.alphavantage.co/support/#api-key)** | **[Pricing](https://www.alphavantage.co/premium/)**

**Twelve Data** - Best for Multi-Asset
- **Free Tier**: 800 API calls/day, 8 calls/minute
- **Paid Tier**: $9.99/month (Basic), up to $79.99/month (Pro)
- **Coverage**: Stocks, forex, crypto, ETFs, indices
- **Symbol Format**: Standard ticker, some use TICKER:EXCHANGE
- **Rate Limit**: 8/min, 800/day (free) | 800/min (paid)
- **Best For**: Stocks + forex + crypto in one API
- **Bonus**: Built-in technical indicators
- **[Get API Key](https://twelvedata.com/pricing)** | **[Pricing](https://twelvedata.com/pricing)**

**CryptoCompare** - Best for Crypto Real-Time
- **Free Tier**: Good limits (varies by endpoint)
- **Paid Tier**: Custom pricing
- **Coverage**: 5,000+ cryptocurrencies, multiple exchanges
- **Symbol Format**: Ticker pairs (BTC, ETH)
- **Rate Limit**: Varies by tier
- **Best For**: Real-time crypto data, aggregated prices
- **[Get API Key](https://min-api.cryptocompare.com/pricing)** (optional for free tier)

#### Educational Tier (Static Datasets + Free Credits)

**AlgoSeek NASDAQ 100 Minute Bars** - FREE Educational Dataset 🟡 **Book Tier 2**
- **Type**: Static dataset download (NOT live API)
- **Coverage**: NASDAQ 100 stocks, minute bars, 2015-2017
- **Data Size**: ~5GB compressed
- **Format**: CSV with 54 fields (OHLCV + bid-ask spread + tick stats)
- **Book Use Cases**:
  - ✅ Chapter 6: Intraday pattern analysis
  - ✅ Chapter 7: Microstructure and order flow
  - ✅ Chapter 9: High-frequency backtesting basics
  - ✅ Excellent for: Learning minute-frequency strategies WITHOUT paying for live data
- **Best For**: Intraday analysis, learning minute bars, thesis projects
- **Cost/Benefit**: ⭐⭐⭐⭐ FREE 5GB of institutional-quality minute data
- **[Download Dataset](https://www.algoseek.com/ml4t-book-data.html)** | **[Documentation PDF](https://us-equity-market-data-docs.s3.amazonaws.com/algoseek.US.Equity.TAQ.Minute.Bars.pdf)**
- **Integration**: Use `ml4t-data ingest algoseek` to convert to Hive partitions

**DataBento** - Institutional Data with Educational Credits 🟡 **Book Tier 2** → 🔵 **Tier 3**
- **Free Tier**: $10 free credits for new users
- **Paid Tier**: $9+/month (usage-based), institutional tiers available
- **Coverage**: CME, CBOE, ICE futures and options, tick-level equities
- **Symbol Format**: Provider-specific (ES.FUT, SPX.OPT)
- **Book Use Cases**:
  - ✅ Chapter 8: Futures continuous contracts
  - ✅ Chapter 9: Options pricing and volatility
  - ✅ Chapter 11: Multi-asset portfolio strategies
  - ✅ Use free credits ($10) to download 5-10 years of daily futures
- **Best For**: Futures/options learning → production trading
- **Cost/Benefit**: ⭐⭐⭐⭐ Free credits get you started, affordable for serious use
- **[Get API Key](https://databento.com/signup)** | **[Pricing](https://databento.com/pricing)** | **[Docs](https://databento.com/docs/)**

#### Professional Providers (Paid Only or Limited Free)

**Finnhub** - Professional Trading Platform 🔵 **Book Tier 3**
- **Free Tier**: Real-time quotes (60/min), 30 OHLCV calls/day (impractical for backtesting)
- **Paid Tier**: $59.99/month (Starter) to custom enterprise
- **Coverage**: 70+ global exchanges, stocks, forex, crypto
- **Symbol Format**: Standard ticker (AAPL)
- **Rate Limit**: 60/min (free), higher on paid tiers
- **Book Use Cases**:
  - ✅ Advanced chapters: Global equities (70+ exchanges)
  - ✅ Production: Live trading with real-time quotes
  - ✅ Note: Free tier NOT suitable for historical backtesting
- **Best For**: Professional trading systems, global markets, real-time data
- **Cost/Benefit**: ⭐⭐ Premium price for premium global coverage
- **Important**: Historical OHLCV requires paid subscription
- **[Get API Key](https://finnhub.io/register)** | **[Pricing](https://finnhub.io/pricing)**

**Polygon** - Professional Multi-Asset 🔵 **Book Tier 3**
- **Free Tier**: None (paid only)
- **Paid Tier**: $99/month (Starter), up to $399/month (Advanced)
- **Coverage**: Stocks, options, forex, crypto
- **Symbol Format**: Standard ticker
- **Rate Limit**: Varies by tier
- **Book Use Cases**:
  - ✅ Advanced chapters: Multi-asset strategies
  - ✅ Production: Live trading infrastructure
  - ✅ Options: Greeks, chains, real-time pricing
- **Best For**: Production trading systems, multi-asset portfolios
- **Cost/Benefit**: ⭐⭐ Professional quality, premium price
- **[Pricing](https://polygon.io/pricing)**

**OANDA** - Professional Forex
- **Free Tier**: Demo account (limited)
- **Coverage**: Major and minor forex pairs, CFDs
- **Symbol Format**: Pair format (EUR_USD)
- **Best For**: Professional forex trading and research
- **[Get API Key](https://developer.oanda.com/)**

**Databento** - Institutional Derivatives
- **Free Tier**: None (institutional pricing)
- **Coverage**: CME, CBOE, ICE futures and options, tick-level data
- **Symbol Format**: Provider-specific (ES.FUT.C.0)
- **Best For**: Institutional derivatives, high-frequency trading
- **[Contact Sales](https://databento.com/pricing)**

**Binance** - Crypto Exchange (Geo-Restricted)
- **Free Tier**: Unlimited (no API key for public endpoints)
- **Coverage**: 600+ crypto spot and futures pairs
- **Symbol Format**: Pair format (BTCUSDT)
- **Best For**: Crypto trading (where available)
- **Important**: Returns HTTP 451 in restricted regions (requires VPN)
- **Rate Limit**: 1200 requests/minute
- **[API Docs](https://www.binance.com/en/binance-api)**

#### Economic & Factor Data Providers (Free)

**FRED** - Federal Reserve Economic Data 🟢 **Free**
- **Free Tier**: 120 requests/minute (free API key required)
- **Coverage**: 850,000+ economic time series (macro indicators, rates, employment)
- **Symbol Format**: Series ID (VIXCLS, DGS10, UNRATE, SP500)
- **Book Use Cases**:
  - ✅ Chapter 5: Macro regime indicators
  - ✅ Chapter 7: Economic feature engineering
  - ✅ Advanced: Macro factor models
- **Best For**: Economic indicators, interest rates, macro research
- **Cost/Benefit**: ⭐⭐⭐⭐⭐ Free institutional-grade economic data
- **[Get API Key](https://fred.stlouisfed.org/docs/api/api_key.html)** | **[Series Search](https://fred.stlouisfed.org/)**

**Fama-French** - Academic Factor Data 🟢 **Free**
- **Free Tier**: Unlimited (no API key, downloads from Ken French Data Library)
- **Coverage**: 50+ factor datasets (FF3, FF5, Momentum, industry portfolios)
- **Symbol Format**: Dataset ID (ff3, ff5, mom, ind_48)
- **Book Use Cases**:
  - ✅ Chapter 5: Factor model construction
  - ✅ Chapter 7: Factor exposure analysis
  - ✅ Advanced: Multi-factor portfolio backtesting
- **Best For**: Academic research, factor investing, portfolio attribution
- **Cost/Benefit**: ⭐⭐⭐⭐⭐ Publication-grade factor data for free
- **[Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)**

**AQR** - Alternative Factor Data 🟢 **Free**
- **Free Tier**: Unlimited (no API key, downloads from AQR website)
- **Coverage**: 16 factor datasets (QMJ, BAB, TSMOM, commodity factors)
- **Symbol Format**: Dataset ID (qmj_factors, bab_factors, tsmom)
- **Book Use Cases**:
  - ✅ Chapter 5: Quality, low-beta, and momentum factors
  - ✅ Advanced: Cross-asset factor strategies
- **Best For**: Alternative factors beyond Fama-French, cross-asset research
- **Cost/Benefit**: ⭐⭐⭐⭐⭐ Institutional-quality alternative factors
- **[AQR Datasets](https://www.aqr.com/Insights/Datasets)**

#### Prediction Markets & Alternative Data

**Kalshi** - US Prediction Market 🟡 **Free Tier**
- **Free Tier**: API access with account
- **Coverage**: Political, economic, weather events
- **Symbol Format**: Event ticker
- **Book Use Cases**:
  - ✅ Chapter 5: Alternative data for sentiment
  - ✅ Advanced: Event-driven strategies
- **Best For**: Probability estimates for discrete events
- **[API Docs](https://kalshi.com/developer)**

**Polymarket** - Crypto Prediction Market 🟢 **Free**
- **Free Tier**: Public API, no key required
- **Coverage**: Crypto-native prediction markets
- **Symbol Format**: Market ID
- **Book Use Cases**:
  - ✅ Chapter 5: Crypto sentiment data
  - ✅ Advanced: DeFi integration
- **Best For**: Crypto-adjacent event probabilities
- **[API Docs](https://docs.polymarket.com)**

#### Historical & Tick Data Providers

**Wiki Prices** - Historical US Equities 🟢 **Free (Local File)**
- **Free Tier**: Local Parquet file (631MB, one-time download)
- **Coverage**: 3,199 US stocks, 1962-2018, 15.4M rows
- **Symbol Format**: Standard ticker (AAPL)
- **Book Use Cases**:
  - ✅ Chapter 3: Survivorship bias analysis
  - ✅ Chapter 5: Long-term backtesting (30+ years)
  - ✅ Fallback when Yahoo unavailable
- **Best For**: Historical research, survivorship-bias-free analysis
- **Cost/Benefit**: ⭐⭐⭐⭐⭐ Curated historical dataset

**NASDAQ ITCH** - Tick-Level Order Book 🟢 **Free Samples**
- **Free Tier**: Sample files from NASDAQ
- **Coverage**: Full order book, tick-by-tick messages
- **Symbol Format**: Standard ticker
- **Book Use Cases**:
  - ✅ Chapter 4: Market microstructure
  - ✅ Chapter 4: Order book reconstruction
- **Best For**: Learning market microstructure, academic research
- **[Sample Data](https://emi.nasdaq.com/ITCH/)**

### Provider Selection by Use Case

#### "I'm learning and need free data"
**Recommendation**: Yahoo Finance (stocks) + CoinGecko (crypto)
- **Why**: No API keys, unlimited historical data, broad coverage
- **Cost**: $0/month

#### "I need daily updates for my portfolio"
**Recommendation**: Tiingo (1000/day) or EODHD (500/day free tier)
- **Why**: Generous free tiers, excellent data quality
- **Cost**: $0/month

#### "I'm building a production backtest system"
**Recommendation**: EODHD Paid (€19.99/month) or Tiingo Paid ($30/month)
- **Why**: Unlimited calls, reliable, affordable
- **Cost**: €19.99-30/month

#### "I need global stock coverage"
**Recommendation**: EODHD (60+ exchanges for €19.99/month)
- **Why**: Best price/coverage ratio, includes LSE, FRA, TSE, HKEX, etc.
- **Cost**: €19.99/month

#### "I need stocks + forex + crypto"
**Recommendation**: Twelve Data (800/day free) or Alpha Vantage (25/day free)
- **Why**: Multi-asset in one API
- **Cost**: $0/month (free tier) or $9.99/month (paid)

#### "I'm building a professional trading system"
**Recommendation**: Finnhub ($59.99/month) or Polygon ($99/month)
- **Why**: Real-time + historical, professional-grade infrastructure
- **Cost**: $60-99/month

#### "I need derivatives data (futures/options)"
**Recommendation**: Databento (institutional pricing)
- **Why**: Only provider with tick-level CME/CBOE/ICE data
- **Cost**: Contact for pricing

### Symbol Format Reference

Different providers use different symbol conventions:

| Provider | Stocks | Crypto | Forex | Format Notes |
|----------|--------|--------|-------|--------------|
| **Yahoo** | AAPL | BTC-USD | EURUSD=X | Suffix for crypto/forex |
| **EODHD** | AAPL.US | N/A | N/A | Ticker.Exchange required |
| **Tiingo** | AAPL | btcusd | N/A | Standard ticker |
| **CoinGecko** | N/A | bitcoin | N/A | Coin ID (not ticker) |
| **Twelve Data** | AAPL | BTC/USD | EUR/USD | Some use TICKER:EXCHANGE |
| **Alpha Vantage** | AAPL | BTC | EURUSD | Standard ticker |
| **Finnhub** | AAPL | BINANCE:BTCUSDT | OANDA:EUR_USD | EXCHANGE:SYMBOL |
| **OANDA** | N/A | N/A | EUR_USD | Underscore separator |
| **Binance** | N/A | BTCUSDT | N/A | No separator |

### Rate Limiting Best Practices

**Free Tier Users:**
1. Use incremental updates (only fetch new data)
2. Cache aggressively (use `HiveStorage`)
3. Batch requests when possible
4. Respect rate limits (ml4t-data handles this automatically)
5. Consider upgrading if hitting limits regularly

**Example: Minimize API Calls**
```python
from ml4t.data import DataManager
from ml4t.data.storage import HiveStorage, StorageConfig

# Setup storage for caching
storage = HiveStorage(StorageConfig(base_path="./data"))
manager = DataManager(storage=storage)

# First load: Downloads 30 days by default (1 API call)
manager.load("AAPL", provider="tiingo")

# Incremental update: Only fetches NEW data (1 API call)
manager.update("AAPL")  # Fetches only data since last update

# No API calls - reads from cache!
df = storage.read("equities/daily/AAPL")
```

### Getting Started Examples

**Free Tier (Yahoo Finance - No API Key)**
```python
from ml4t.data import DataManager

manager = DataManager()

# US stocks
data = manager.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")

# Crypto
btc = manager.fetch("BTC-USD", "2024-01-01", "2024-12-31", provider="yahoo")

# Forex
eur = manager.fetch("EURUSD=X", "2024-01-01", "2024-12-31", provider="yahoo")
```

**Free Tier (CoinGecko - No API Key)**
```python
from ml4t.data.providers import CoinGeckoProvider

provider = CoinGeckoProvider()  # No API key needed!
btc = provider.fetch_ohlcv("bitcoin", "2024-01-01", "2024-12-31")
eth = provider.fetch_ohlcv("ethereum", "2024-01-01", "2024-12-31")
```

**Free Tier (With API Key - Tiingo)**
```python
import os
os.environ["TIINGO_API_KEY"] = "your_key_here"  # Get from tiingo.com

from ml4t.data import DataManager

manager = DataManager()
data = manager.fetch("AAPL", "2024-01-01", "2024-12-31", provider="tiingo")
```

**Paid Tier (EODHD - Global Coverage)**
```python
import os
os.environ["EODHD_API_KEY"] = "your_key_here"  # Get from eodhd.com

from ml4t.data import DataManager

manager = DataManager()

# US stocks
us_data = manager.fetch("AAPL", "2024-01-01", "2024-12-31",
                        provider="eodhd", exchange="US")

# London Stock Exchange
lse_data = manager.fetch("VOD", "2024-01-01", "2024-12-31",
                         provider="eodhd", exchange="LSE")

# Frankfurt Stock Exchange
fra_data = manager.fetch("BMW", "2024-01-01", "2024-12-31",
                         provider="eodhd", exchange="FRA")
```

**📖 Asset Class Guides**: For detailed guides with examples, best practices, and gotchas:
- **[Cryptocurrency Guide](docs/asset-classes/crypto.md)** - CoinGecko, CryptoCompare
- **[Equities Guide](docs/asset-classes/equities.md)** - US & Global stocks, ETFs
- **[Forex Guide](docs/asset-classes/forex.md)** - OANDA, currency pairs
- **[Futures Guide](docs/asset-classes/futures.md)** - Databento, derivatives

Or use the **[Provider Selection Flowchart](docs/provider-selection-guide.md)** for guided selection.

## Key Features

- 🔌 **14 Providers**: CoinGecko, IEX Cloud, Twelve Data, Alpha Vantage, Tiingo, Finnhub, EODHD, CryptoCompare, OANDA, Databento, Polygon, Yahoo, Binance, and more
- 📈 **Incremental Updates**: Smart gap detection and backfilling - only fetch what's new
- 🔍 **Data Quality**: Anomaly detection for price staleness, outliers, and volume spikes
- 🔒 **Transaction Support**: ACID guarantees for batch operations
- 🛠️ **Extensible**: Add your own providers with templates and guides
- 📊 **Standard Format**: All providers return the same Polars DataFrame schema

## Multi-Asset Support 🚀

**NEW**: Load and analyze hundreds of symbols simultaneously with blazing-fast performance.

```python
from ml4t.data import DataManager
from ml4t.data.universe import Universe

manager = DataManager()

# Load entire S&P 500 in seconds
df = manager.batch_load_universe(
    universe='SP500',
    start='2024-01-01',
    end='2024-12-31',
    provider='yahoo'
)

print(f"Loaded {len(df):,} rows for {df['symbol'].n_unique()} symbols")
# Output: Loaded 126,000 rows for 500 symbols
```

### Why Multi-Asset?

- **📊 100x Faster**: Load 100 symbols from cache in <1 second (vs 12-50 seconds from network)
- **🎯 Pre-defined Universes**: S&P 500, NASDAQ-100, Crypto Top-100, Forex Majors
- **⚡ Parallel Fetching**: Configurable workers for optimal throughput
- **💾 Cache-First**: Intelligent storage loading with automatic fallback
- **🔄 Stacked Format**: Polars-native format scales to 1000+ symbols

### Quick Examples

**Load Multiple Symbols**:
```python
# Basic batch loading
df = manager.batch_load(
    symbols=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
    start='2024-01-01',
    end='2024-12-31',
    provider='yahoo',
    max_workers=8  # Parallel fetching
)
```

**Use Pre-defined Universes**:
```python
# S&P 500
df = manager.batch_load_universe('SP500', '2024-01-01', '2024-12-31')

# NASDAQ-100
df = manager.batch_load_universe('NASDAQ100', '2024-01-01', '2024-12-31')

# Crypto Top-100
df = manager.batch_load_universe('CRYPTO_TOP_100', '2024-01-01', '2024-12-31')

# Forex Majors
df = manager.batch_load_universe('FOREX_MAJORS', '2024-01-01', '2024-12-31')
```

**Fast Cache Loading (100x Speedup)**:
```python
from ml4t.data.storage import HiveStorage, StorageConfig

# Setup storage
storage = HiveStorage(StorageConfig(base_path="./data"))
manager = DataManager(storage=storage)

# Load from cache (extremely fast!)
df = manager.batch_load_from_storage(
    symbols=['AAPL', 'MSFT', 'GOOG'],
    start='2024-01-01',
    end='2024-12-31'
)
# Loads in ~0.01 seconds instead of 5-10 seconds!
```

**Cross-Sectional Analysis**:
```python
import polars as pl

# Calculate returns per symbol
df = df.with_columns(
    pl.col('close').pct_change().over('symbol').alias('returns')
)

# Daily cross-sectional statistics
stats = df.group_by('timestamp').agg([
    pl.col('returns').mean().alias('mean_return'),
    pl.col('returns').std().alias('std_return'),
    pl.col('volume').sum().alias('total_volume'),
])
```

### Performance Benchmarks

| Operation | Symbols | Time | Rows/Second | Speedup |
|-----------|---------|------|-------------|---------|
| **batch_load_from_storage()** | 100 | 0.165s | 152,000 | **76-303x faster** |
| **batch_load()** (network) | 100 | 12-50s | 500-2,000 | Baseline |
| **Format conversion** | 50 | 0.009s | 1.4M | - |

### Learn More

- **[Complete Multi-Asset Guide](docs/multi_asset_guide.md)** - Comprehensive documentation
- **[Quickstart Example](examples/multi_asset_quickstart.py)** - Working code examples
- **[Format Conversion Example](examples/format_conversion_example.py)** - Stacked ↔ Wide conversion

---

## Complete Provider Comparison

All providers share the same simple API. Choose based on your budget and needs:

### Asset Class Coverage Matrix

| Provider | Crypto | US Stocks | Global Stocks | Forex | Futures | API Key | Free Tier | Best For |
|----------|--------|-----------|---------------|-------|---------|---------|-----------|----------|
| **Yahoo** | ✅ | ✅ | ✅ | ✅ | ❌ | No | Unlimited | Best free option, learning |
| **CoinGecko** | ✅ | ❌ | ❌ | ❌ | ❌ | No | Unlimited | Crypto historical data |
| **EODHD** | ❌ | ✅ | ✅ | ❌ | ❌ | Yes | 500/day | Global stocks, best value |
| **Tiingo** | ✅ | ✅ | ❌ | ❌ | ❌ | Yes | 1000/day | High-quality stock data |
| **IEX Cloud** | ❌ | ✅ | ❌ | ❌ | ❌ | Yes | 50K/mo | US equities + fundamentals |
| **Alpha Vantage** | ✅ | ✅ | ✅ | ✅ | ❌ | Yes | 25/day | Multi-asset research |
| **Twelve Data** | ✅ | ✅ | ❌ | ✅ | ❌ | Yes | 800/day | Multi-asset + indicators |
| **Finnhub** | ✅ | ✅ | ✅ | ✅ | ❌ | Yes | 30 OHLCV/day* | Professional grade |
| **CryptoCompare** | ✅ | ❌ | ❌ | ❌ | ❌ | Optional | Good | Crypto real-time + historical |
| **Polygon** | ✅ | ✅ | ❌ | ✅ | ❌ | Yes | Paid only | Professional multi-asset |
| **OANDA** | ❌ | ❌ | ❌ | ✅ | ❌ | Yes | Demo only | Professional forex |
| **Databento** | ❌ | ✅ | ❌ | ❌ | ✅ | Yes | Paid only | Institutional futures/options |
| **Binance** | ✅ | ❌ | ❌ | ❌ | ✅ | No | Unlimited** | Crypto (geo-restricted) |

*Finnhub free tier has only 30 OHLCV calls/day (impractical for backtesting). Paid plan required for historical data.
**Binance returns HTTP 451 in restricted regions (VPN required)

### Rate Limits Comparison

| Provider | Free Tier Limit | Paid Tier Limit | Cost (Paid) | Notes |
|----------|-----------------|-----------------|-------------|-------|
| **Yahoo** | ~2000/hour | N/A | $0 | Undocumented, unofficial API |
| **CoinGecko** | 10-50/min | 500/min | $129/mo | No daily limit on free tier |
| **EODHD** | 500/day | Unlimited | €19.99/mo | Best value for global coverage |
| **Tiingo** | 1000/day | 20K/hour | $30/mo | 500 unique symbols/month (free) |
| **IEX Cloud** | 50K msg/mo | Usage-based | $0-999/mo | 1 OHLCV bar ≈ 1 message credit |
| **Alpha Vantage** | 5/min, 25/day | 75/min | $49.99/mo | Most restrictive free tier |
| **Twelve Data** | 8/min, 800/day | 800/min | $9.99-79.99/mo | Built-in technical indicators |
| **Finnhub** | 60/min quotes, 30 OHLCV/day | Higher | $59.99+/mo | Free tier impractical for backtesting |
| **CryptoCompare** | Varies | Varies | Custom | Good free tier for crypto |
| **Polygon** | N/A | Varies | $99-399/mo | No free tier |
| **OANDA** | Demo account | Production | N/A | Professional forex data |
| **Databento** | N/A | Institutional | Contact | Tick-level derivatives data |
| **Binance** | 1200/min | Same | $0 | Geo-restricted (HTTP 451) |

---

## Advanced Features

ML4T Data includes production-grade features for data quality, reliability, and enterprise use cases.

### Session Management 📅

**For Futures & Cross-Validation**: Assign session dates and fill gaps for CME futures and other exchange-traded products.

```python
from ml4t-data import DataManager

manager = DataManager(storage_path="./data")

# Load CME Bitcoin futures with 1-minute bars
key = manager.load(
    symbol="BTC",
    start="2024-01-01",
    end="2024-12-31",
    provider="databento",
    frequency="1min",
    exchange="CME",
    calendar="CME_Globex_Crypto"
)

# Read data and assign session dates
df = manager.storage.read(key).collect()
df = manager.assign_sessions(df, exchange="CME")

# Fill gaps for complete sessions (forward-fill OHLC, zero volume)
df = manager.complete_sessions(
    df,
    exchange="CME",
    fill_gaps=True,
    zero_volume=True
)

# Now ready for session-aware cross-validation!
# Each timestamp has a session_date column
```

**Why Session Management?**
- **Futures Trading**: CME sessions run Sunday 5pm → Friday 4pm (23 hours/day)
- **Cross-Validation**: Split data by session dates, not calendar dates
- **Gap Filling**: Some providers skip zero-volume periods - fill for continuous analysis
- **Exchange Calendars**: Supports 40+ exchanges via `pandas-market-calendars`

**Supported Exchanges**:
- CME (futures), NYSE, NASDAQ (equities)
- LSE, TSE, HKEX, ASX (international)
- Full holiday calendars and DST handling

**Example: CME Futures Workflow**
```python
# 1. Load minute-level futures data
df = manager.load("BTC", "2024-01-01", "2024-12-31",
                  provider="databento", frequency="1min",
                  exchange="CME")

# 2. Assign session dates
df = manager.assign_sessions(df, exchange="CME")

# 3. Fill gaps with forward-filled OHLC
df = manager.complete_sessions(df, exchange="CME")

# 4. Group by session for analysis
sessions = df.group_by("session_date").agg([
    pl.col("volume").sum().alias("total_volume"),
    pl.col("close").last().alias("session_close")
])
```

See [`examples/cme_futures_sessions.py`](examples/cme_futures_sessions.py) for complete workflow.

---

### Alternative Bar Types 📊

**Beyond Time-Based Bars**: Support for volume bars, trade bars, dollar bars, and tick bars.

```python
from ml4t-data import DataManager

manager = DataManager(storage_path="./data")

# Volume bars (aggregate by volume threshold)
manager.load(
    symbol="BTC",
    start="2024-01-01",
    end="2024-12-31",
    provider="databento",
    bar_type="volume",
    bar_threshold=1000,  # 1000 BTC per bar
    exchange="CME"
)

# Trade bars (aggregate by number of trades)
manager.load(
    symbol="ES",
    start="2024-01-01",
    end="2024-12-31",
    provider="databento",
    bar_type="trade",
    bar_threshold=500,  # 500 trades per bar
    exchange="CME"
)

# Dollar bars (aggregate by dollar volume)
manager.load(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    provider="polygon",
    bar_type="dollar",
    bar_threshold=1000000,  # $1M per bar
    exchange="NASDAQ"
)
```

**Bar Types**:
- `time` - Traditional time-based (default: "1min", "5min", "daily")
- `volume` - Aggregate by volume threshold (better for low-liquidity assets)
- `trade` - Aggregate by trade count (institutional order flow analysis)
- `dollar` - Aggregate by dollar volume (constant information per bar)
- `tick` - Aggregate by tick count (microstructure analysis)

**Metadata Storage**: Bar specifications automatically saved in `.metadata/{key}.json`:
```json
{
  "bar_type": "volume",
  "bar_params": {"threshold": 1000},
  "exchange": "CME",
  "calendar": "CME_Globex_Crypto"
}
```

**Use update() seamlessly** - metadata preserved across updates:
```python
# Metadata loaded automatically
manager.update(symbol="BTC")  # Uses stored bar_type, threshold, exchange
```

---

### Bulk Operations 🔄

**Update Multiple Symbols**: Efficiently update all stored data with filters.

```python
from ml4t-data import DataManager

manager = DataManager(storage_path="./data")

# Update all Yahoo Finance data
results = manager.update_all(provider="yahoo")
print(f"Updated {len(results)} symbols")

# Update all CME futures
results = manager.update_all(exchange="CME")

# Update all equities
results = manager.update_all(asset_class="equities")

# Discover stored symbols
symbols = manager.list_symbols(provider="databento", exchange="CME")
print(f"Found {len(symbols)} CME symbols")

# Get metadata for any symbol
metadata = manager.get_metadata("BTC")
print(f"Provider: {metadata.provider}")
print(f"Last updated: {metadata.last_updated}")
print(f"Exchange: {metadata.exchange}")
```

**Discovery Methods**:
```python
# List symbols with filters
symbols = manager.list_symbols(
    provider="yahoo",        # Filter by provider
    asset_class="equities",  # Filter by asset class
    exchange="NYSE"          # Filter by exchange
)

# Get metadata for symbol
metadata = manager.get_metadata("AAPL")
# Returns: Metadata(provider, symbol, bar_type, exchange, last_updated, ...)

# Update all with filters
results = manager.update_all(
    provider="yahoo",
    asset_class="equities"
)
# Returns: {"AAPL": "equities/daily/AAPL", "MSFT": "equities/daily/MSFT", ...}
```

**Use Cases**:
- Nightly updates: `update_all()` across entire portfolio
- Data discovery: `list_symbols()` to see what's stored
- Monitoring: `get_metadata()` to check last update times
- Batch migrations: Filter by provider or exchange

---

### Data Import & Seeding 📥

**Import External Data**: Seed ml4t-data storage with data from other sources, then use `update()` to keep current.

```python
from ml4t-data import DataManager
import polars as pl

manager = DataManager(storage_path="./data")

# Import data from DataBento local files
df = pl.read_parquet("databento_btc_2020_2023.parquet")

manager.import_data(
    data=df,
    symbol="BTC",
    provider="databento",
    frequency="1min",
    asset_class="crypto_futures",
    exchange="CME",
    calendar="CME_Globex_Crypto"
)

# Now use update() to fetch only new data
manager.update("BTC")  # Fetches 2024-01-01 onwards using stored metadata
```

**Import from Any Source**:
```python
# Yahoo Finance CSV export
df = pl.read_csv("yahoo_aapl.csv")
manager.import_data(df, symbol="AAPL", provider="yahoo", frequency="daily")

# Custom data pipeline
df = your_custom_etl_pipeline()
manager.import_data(df, symbol="CUSTOM", provider="internal", frequency="1hour")

# Multiple symbols
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    df = pl.read_parquet(f"historical/{symbol}.parquet")
    manager.import_data(df, symbol=symbol, provider="yahoo")
```

**Requirements**:
- DataFrame must have: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Timestamps must be timezone-aware (UTC)
- No duplicates allowed

**After Import**:
- Metadata saved automatically in `.metadata/{key}.json`
- Use `update()` to fetch incremental data
- Use `update_all()` to update all imported symbols
- All ml4t-data features work (validation, anomaly detection, sessions)

---

### Data Quality Monitoring 🔍

Automatic anomaly detection for price data quality assurance:

```python
from ml4t.data.anomaly import AnomalyManager, PriceStalenessDetector, ReturnOutlierDetector, VolumeSpikeDetector

# Create anomaly manager with detectors
manager = AnomalyManager()
manager.add_detector(PriceStalenessDetector(max_gap_days=3))
manager.add_detector(ReturnOutlierDetector(threshold=5.0))  # 5 std devs
manager.add_detector(VolumeSpikeDetector(threshold=10.0))  # 10x average

# Run detection
report = manager.detect(df)

# Check results
if report.has_anomalies:
    print(f"Found {len(report.anomalies)} anomalies")
    for anomaly in report.anomalies:
        print(f"  {anomaly.severity}: {anomaly.description}")
```

**Available Detectors**:
- **PriceStalenessDetector** - Detects gaps in price data (missing dates)
- **ReturnOutlierDetector** - Detects abnormal price movements (flash crashes, errors)
- **VolumeSpikeDetector** - Detects unusual trading volume (data errors, events)

**CLI Usage**:
```bash
ml4t-data validate --anomalies --symbol AAPL  # Check for anomalies
ml4t-data health --check-anomalies           # Include in health check
```

---

### Transaction Support 🔒

ACID guarantees for multi-symbol batch operations:

```python
from ml4t-data import DataManager

# Enable transactions for batch operations
manager = DataManager(use_transactions=True)

# All-or-nothing batch fetch
try:
    results = manager.batch_fetch(
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start="2024-01-01",
        end="2024-12-31",
        provider="yahoo"
    )
    # All symbols fetched successfully, data committed
except Exception as e:
    # Any failure rolls back all changes
    print(f"Batch failed, rolled back: {e}")
```

**Use Cases**:
- Multi-symbol batch loading (ensure all succeed or none)
- Historical backfills (rollback on incomplete data)
- Data migrations (ensure consistency across operations)

**Trade-off**: ~15% overhead for ACID guarantees

---

### Async Batch Loading ⚡

**3-10x Faster Multi-Symbol Fetches**: Use async/await to fetch multiple symbols concurrently.

```python
import asyncio
from ml4t.data.managers.async_batch import async_batch_load
from ml4t.data.providers.yahoo import YahooFinanceProvider

async def fetch_portfolio():
    async with YahooFinanceProvider() as provider:
        df = await async_batch_load(
            provider,
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            start="2024-01-01",
            end="2024-12-31",
            max_concurrent=10,  # Limit concurrent requests
        )
    return df

# Run it
df = asyncio.run(fetch_portfolio())
print(f"Fetched {len(df)} rows for {df['symbol'].n_unique()} symbols")
```

**Performance Comparison**:

| Method | 100 Symbols | 500 Symbols | Speedup |
|--------|-------------|-------------|---------|
| Sequential `fetch_ohlcv()` | 50-100s | 250-500s | Baseline |
| `async_batch_load()` | 5-15s | 25-75s | **3-10x faster** |

**Provider Async Support**:

All 12 OHLCV providers support async via `async_batch_load()`:

| Provider | Async Type | Notes |
|----------|------------|-------|
| Yahoo | Thread-wrapped | Uses `asyncio.to_thread()` |
| EODHD | Native httpx | True async HTTP |
| Binance | Native httpx | True async HTTP |
| CoinGecko | Native httpx | True async HTTP |
| OKX | Native httpx | True async HTTP |
| TwelveData | Native httpx | True async HTTP |
| CryptoCompare | Native httpx | True async HTTP |
| DataBento | Thread-wrapped | SDK is sync |
| Finnhub | Thread-wrapped | SDK is sync |
| Oanda | Thread-wrapped | SDK is sync |
| Polygon | Thread-wrapped | SDK is sync |
| Tiingo | Thread-wrapped | SDK is sync |

**Native Async** providers use `httpx.AsyncClient` for true non-blocking I/O.
**Thread-wrapped** providers use `asyncio.to_thread()` to avoid blocking the event loop.

**Direct Async Usage** (for custom workflows):

```python
# Use provider's native async method directly
async with YahooFinanceProvider() as provider:
    # Fetch multiple symbols concurrently
    tasks = [
        provider.fetch_ohlcv_async("AAPL", "2024-01-01", "2024-12-31"),
        provider.fetch_ohlcv_async("MSFT", "2024-01-01", "2024-12-31"),
        provider.fetch_ohlcv_async("GOOGL", "2024-01-01", "2024-12-31"),
    ]
    results = await asyncio.gather(*tasks)

# Combine results
import polars as pl
df = pl.concat(results)
```

**Best Practices**:
- Use `max_concurrent` to respect provider rate limits (default: 10)
- Prefer `async_batch_load()` over manual `asyncio.gather()` for error handling
- Native async providers are slightly faster than thread-wrapped ones
- All providers handle rate limiting automatically

---

### Schema Migration (v2.0+) 📦

Future-proof schema evolution for production deployments:

```bash
# When upgrading with schema changes
ml4t-data migrate --from-version 1.0 --to-version 2.0
```

**Handles**:
- New columns (e.g., `adjusted_close`, `exchange`)
- Type changes (e.g., Float32 → Float64)
- Timezone updates (e.g., naive → UTC)
- Automatic backups before migration

**Note**: No migrations currently needed. This feature activates in v2.0+ when schema changes are introduced.

---

## Provider Implementation Examples

The sections below show how to use each provider in your code. All providers share the same API pattern for consistency.

### Crypto Providers

#### CoinGecko (FREE - No API Key)

**Best for**: Crypto historical data, beginners, no rate limits
**Tier**: Free (Generous)
**Cost**: $0/month

```python
from ml4t.data.providers import CoinGeckoProvider

provider = CoinGeckoProvider()  # No API key needed!
data = provider.fetch_ohlcv("bitcoin", "2024-01-01", "2024-12-31")
```

**Key Details**:
- ✅ 10,000+ cryptocurrencies
- ✅ Historical OHLCV data
- ✅ No API key required
- ✅ 10-50 calls/minute (free tier)
- ✅ Unlimited historical depth
- ⚠️ Symbol format: Coin ID (bitcoin, ethereum) NOT ticker (BTC, ETH)

**Rate Limits**: 10-50/min (free), 500/min ($129/mo paid tier)
**[Get API Key](https://www.coingecko.com/en/api/pricing)** (optional) | **[Documentation](https://www.coingecko.com/en/api/documentation)**

#### CryptoCompare

**Best for**: Crypto aggregated prices, real-time data
**Tier**: Free (Limited)
**Cost**: $0/month (free tier) or custom pricing

```bash
export CRYPTOCOMPARE_API_KEY="your_key"  # Optional for free tier
```

```python
from ml4t.data.providers import CryptoCompareProvider

provider = CryptoCompareProvider(api_key="your_key")
data = provider.fetch_ohlcv("BTC", "2024-01-01", "2024-12-31")
```

**Key Details**:
- ✅ 5,000+ cryptocurrencies
- ✅ Aggregated prices from multiple exchanges
- ✅ Real-time + historical
- ✅ Free tier available (no API key needed for basic use)
- ✅ WebSocket support for real-time feeds

**Rate Limits**: Varies by tier and endpoint
**[Get API Key](https://min-api.cryptocompare.com/pricing)** | **[Documentation](https://min-api.cryptocompare.com/documentation)**

#### Binance (FREE - Geo-Restricted)

**Best for**: Crypto spot and futures (where available)
**Tier**: Free (Unlimited)
**Cost**: $0/month

```python
from ml4t.data.providers import BinanceProvider

provider = BinanceProvider()  # No API key needed for public data!
data = provider.fetch_ohlcv("BTCUSDT", "2024-01-01", "2024-12-31")
```

**Key Details**:
- ✅ 600+ crypto spot and futures pairs
- ✅ No API key required for public endpoints
- ✅ 1200 requests/minute rate limit
- ✅ High-quality, official exchange data
- ⚠️ Symbol format: No separator (BTCUSDT, not BTC-USDT)
- ❌ Returns HTTP 451 in geo-restricted regions (VPN required)

**Rate Limits**: 1200/min (no daily limit)
**Geo-Restriction**: HTTP 451 in US and other regions
**[API Documentation](https://www.binance.com/en/binance-api)**

---

### US Equities Providers

#### Yahoo Finance (FREE - No API Key)

**Best for**: Learning, backtesting, research
**Tier**: Free (Generous)
**Cost**: $0/month

```python
from ml4t.data import DataManager

manager = DataManager()
data = manager.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")
```

**Key Details**:
- ✅ No API key required
- ✅ ~2000 requests/hour (undocumented limit)
- ✅ Unlimited historical data
- ✅ US stocks, ETFs, indices, forex, crypto
- ✅ Adjusted close for splits/dividends
- ⚠️ Unofficial API (may break without notice)
- ⚠️ No official support or SLA

**Rate Limits**: ~2000/hour (estimated, undocumented)
**[Get Started](https://finance.yahoo.com)** (no registration needed)

#### Tiingo (FREE - 1000/day)

**Best for**: High-quality US stock data, IEX integration
**Tier**: Free (Limited) or Paid (Affordable)
**Cost**: $0/month (free) or $30/month (paid)

```bash
# Get free key at: https://www.tiingo.com/account/api/token
export TIINGO_API_KEY="your_key"
```

```python
from ml4t.data.providers import TiingoProvider
from ml4t.data.storage import HiveStorage, StorageConfig

provider = TiingoProvider(api_key="your_key")

# Fetch daily data
data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")
```

**Key Details**:
- ✅ 1000 API calls/day (free tier)
- ✅ 500 unique symbols/month (free tier)
- ✅ 20,000 calls/hour (paid tier)
- ✅ Excellent data quality (IEX-powered)
- ✅ Incremental updates supported
- ✅ Daily, weekly, monthly frequencies
- ✅ Crypto support (BTC, ETH)

**Rate Limits**: 1000/day (free), 20K/hour (paid)
**[Get API Key](https://www.tiingo.com/account/api/token)** | **[Pricing](https://www.tiingo.com/about/pricing)** | **[Documentation](https://api.tiingo.com/documentation/)**

#### IEX Cloud (FREE - 50K messages/month)

**Best for**: US equities with fundamentals and news
**Tier**: Free (Limited) or Paid (Usage-Based)
**Cost**: $0-999/month (usage-based)

```bash
# Get free key at: https://iexcloud.io/
export IEX_CLOUD_API_KEY="your_key"
```

```python
from ml4t.data.providers import IEXCloudProvider

provider = IEXCloudProvider(api_key="your_key")

# Fetch OHLCV
data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")

# Company fundamentals
info = provider.fetch_company_info("AAPL")
```

**Key Details**:
- ✅ 50,000 message credits/month (free)
- ✅ Real-time + historical OHLCV
- ✅ Company fundamentals and financials
- ✅ News and press releases
- ✅ 1 OHLCV record ≈ 1 credit
- ✅ Pay-as-you-grow pricing

**Rate Limits**: 50K messages/month (free), higher on paid tiers
**[Get API Key](https://iexcloud.io/cloud-login#/register)** | **[Pricing](https://iexcloud.io/pricing/)** | **[Documentation](https://iexcloud.io/docs/)**

#### Alpha Vantage (FREE - 25/day)

**Best for**: Multi-asset research, fundamentals
**Tier**: Free (Very Limited) or Paid (Affordable)
**Cost**: $0/month (free) or $49.99/month (paid)

```bash
# Get free key at: https://www.alphavantage.co/support/#api-key
export ALPHA_VANTAGE_API_KEY="your_key"
```

```python
from ml4t.data.providers import AlphaVantageProvider

provider = AlphaVantageProvider(api_key="your_key")
data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31", frequency="daily")
```

**Key Details**:
- ✅ US and global stocks
- ✅ Daily, weekly, monthly frequencies
- ✅ Company fundamentals
- ✅ Forex and crypto support
- ⚠️ Very restrictive free tier (25 calls/day, 5 calls/min)
- ✅ 75 calls/min on paid tier

**Rate Limits**: 5/min, 25/day (free) | 75/min (paid)
**[Get API Key](https://www.alphavantage.co/support/#api-key)** | **[Pricing](https://www.alphavantage.co/premium/)** | **[Documentation](https://www.alphavantage.co/documentation/)**

---

### Global Equities Providers

#### EODHD (FREE 500/day or €19.99/month)

**Best for**: Global stock coverage at best price/performance ratio
**Tier**: Free (Limited) or Paid (Affordable)
**Cost**: $0/month (500/day) or €19.99/month (unlimited)

```bash
# Get free key at: https://eodhd.com/register
export EODHD_API_KEY="your_key"
```

```python
from ml4t.data.providers import EODHDProvider

# Default exchange (US)
provider = EODHDProvider(api_key="your_key", exchange="US")

# US stocks
us_data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")

# London Stock Exchange
lse_data = provider.fetch_ohlcv("VOD", "2024-01-01", "2024-12-31", exchange="LSE")

# Frankfurt Stock Exchange
fra_data = provider.fetch_ohlcv("BMW", "2024-01-01", "2024-12-31", exchange="FRA")
```

**Global Coverage**:
- **US**: 51,000+ stocks, ETFs, mutual funds
- **London (LSE)**: 2,000+ stocks
- **Frankfurt (FRA)**: 1,000+ stocks
- **Tokyo (TSE)**: 3,700+ stocks
- **+56 more exchanges worldwide**

**Free Tier** (500 API calls/day):
- ✅ 1 year historical depth
- ✅ 150,000+ tickers across 60+ exchanges
- ✅ Daily, weekly, monthly OHLCV
- ✅ Adjusted close prices (splits/dividends)

**Paid Tier** (€19.99/month):
- ✅ Unlimited API calls
- ✅ 30+ years historical coverage
- ✅ Real-time data
- ✅ Fundamentals and financials
- ✅ Best value for global coverage

**Symbol Format**: `TICKER.EXCHANGE` (AAPL.US, VOD.LSE, BMW.FRA)
**Rate Limits**: 500/day (free), unlimited (paid)
**[Get API Key](https://eodhd.com/register)** | **[Pricing](https://eodhd.com/pricing)** | **[Documentation](https://eodhd.com/financial-apis/)**

#### Finnhub (Professional - Paid Required for OHLCV)

**Best for**: Professional trading systems, global markets
**Tier**: Professional (Paid Required)
**Cost**: $59.99+/month

```bash
# Get key at: https://finnhub.io/register
export FINNHUB_API_KEY="your_key"
```

```python
from ml4t.data.providers import FinnhubProvider

provider = FinnhubProvider(api_key="your_key")

# Real-time quotes (FREE TIER)
quote = provider.fetch_quote("AAPL")

# Historical OHLCV (PAID SUBSCRIPTION REQUIRED)
data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")
```

**Free Tier** (Very Limited):
- ✅ Real-time quotes (60/min)
- ✅ 50 symbols via WebSocket
- ⚠️ Historical OHLCV: Only 30 candle calls/day (impractical for backtesting)
- **Best for**: Real-time market data, NOT historical analysis

**Paid Tier** ($59.99+/month):
- ✅ Historical daily/weekly/monthly OHLCV
- ✅ 70+ global exchanges
- ✅ Forex, crypto historical data
- ✅ Company fundamentals and financials
- ✅ Professional-grade infrastructure

**Symbol Format**: Standard ticker (AAPL) or EXCHANGE:SYMBOL for specific exchanges
**Rate Limits**: 60/min (free quotes), higher on paid tiers
**[Get API Key](https://finnhub.io/register)** | **[Pricing](https://finnhub.io/pricing)** | **[Documentation](https://finnhub.io/docs/api)**

---

### Multi-Asset Providers

#### Twelve Data (FREE 800/day)

**Best for**: Stocks + Forex + Crypto in one API with built-in indicators
**Tier**: Free (Limited) or Paid (Affordable)
**Cost**: $0/month (free) or $9.99-79.99/month (paid)

```bash
# Get free key at: https://twelvedata.com/
export TWELVE_DATA_API_KEY="your_key"
```

```python
from ml4t.data.providers import TwelveDataProvider

provider = TwelveDataProvider(api_key="your_key")

# Stocks
stock_data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")

# Forex
forex_data = provider.fetch_ohlcv("EUR/USD", "2024-01-01", "2024-12-31")

# Crypto
crypto_data = provider.fetch_ohlcv("BTC/USD", "2024-01-01", "2024-12-31")

# Real-time quotes (batch)
quotes = provider.fetch_quote(["AAPL", "MSFT", "GOOGL"])

# Technical indicators (built-in)
rsi = provider.fetch_technical_indicator(
    "AAPL", "rsi", "2024-01-01", "2024-12-31", time_period=14
)
```

**Key Details**:
- ✅ Stocks, forex, crypto, ETFs, indices
- ✅ 800 API calls/day (free tier)
- ✅ 800 calls/min (paid tier)
- ✅ Built-in technical indicators (RSI, MACD, Bollinger Bands, etc.)
- ✅ Batch quote requests
- ⚠️ 8 requests/minute on free tier (strict rate limit)

**Rate Limits**: 8/min, 800/day (free) | 800/min (paid)
**[Get API Key](https://twelvedata.com/pricing)** | **[Pricing](https://twelvedata.com/pricing)** | **[Documentation](https://twelvedata.com/docs)**

#### Polygon (Professional Multi-Asset)

**Best for**: Production trading systems, professional-grade data
**Tier**: Professional (Paid Only)
**Cost**: $99-399/month

```bash
export POLYGON_API_KEY="your_key"
```

```python
from ml4t.data.providers import PolygonProvider

provider = PolygonProvider(api_key="your_key")

# Stocks
stock_data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")

# Crypto
crypto_data = provider.fetch_ohlcv("X:BTCUSD", "2024-01-01", "2024-12-31")

# Forex
forex_data = provider.fetch_ohlcv("C:EURUSD", "2024-01-01", "2024-12-31")
```

**Key Details**:
- ✅ Stocks, options, forex, crypto
- ✅ Real-time + historical data
- ✅ Professional-grade infrastructure
- ✅ WebSocket support
- ❌ No free tier (paid subscriptions only)

**Rate Limits**: Varies by tier (no free tier)
**[Pricing](https://polygon.io/pricing)** | **[Documentation](https://polygon.io/docs)**

---

### Forex Providers

#### OANDA (Professional)

**Best for**: Professional forex trading and research
**Tier**: Professional
**Cost**: Demo account (limited) or production account

```bash
export OANDA_API_KEY="your_key"
export OANDA_ACCOUNT_ID="your_account_id"
```

```python
from ml4t.data.providers import OANDAProvider

provider = OANDAProvider(api_key="your_key", account_id="your_account_id")
data = provider.fetch_ohlcv("EUR_USD", "2024-01-01", "2024-12-31")
```

**Key Details**:
- ✅ Major and minor forex pairs
- ✅ Real-time + historical data
- ✅ Professional-grade accuracy
- ✅ Multiple timeframes (M1, M5, H1, D)
- ✅ CFDs and commodities
- ⚠️ Symbol format: Underscore separator (EUR_USD, not EURUSD)

**Rate Limits**: 120 requests/second
**[Get API Key](https://developer.oanda.com/)** | **[Documentation](https://developer.oanda.com/rest-live-v20/introduction/)**

---

### Futures & Options Providers

#### Databento (Institutional - Paid)

**Best for**: Institutional-grade derivatives, tick-level data
**Tier**: Institutional (Paid Only)
**Cost**: Contact for pricing

```bash
export DATABENTO_API_KEY="your_key"
```

```python
from ml4t.data.providers import DatabentoProvider

provider = DatabentoProvider(api_key="your_key")

# CME futures
es_data = provider.fetch_ohlcv("ES.FUT.C.0", "2024-01-01", "2024-12-31")

# Bitcoin futures
btc_data = provider.fetch_ohlcv("BTC.FUT.CME.0", "2024-01-01", "2024-12-31")
```

**Key Details**:
- ✅ CME, CBOE, ICE futures and options
- ✅ Tick-level data (nanosecond precision)
- ✅ Multiple data schemas (OHLCV, trades, book)
- ✅ Institutional quality and SLA
- ✅ Historical and real-time
- ❌ Paid subscription required (no free tier)

**Symbol Format**: Provider-specific (ES.FUT.C.0 for CME E-mini S&P 500)
**Rate Limits**: Based on subscription tier
**[Contact Sales](https://databento.com/pricing)** | **[Documentation](https://databento.com/docs)**

---

### CFTC Commitment of Traders (COT) Data

#### Free Weekly Positioning Data

**Best for**: Sentiment analysis, positioning-based ML features
**Tier**: Free (no API key required)
**Cost**: FREE from CFTC public data

```bash
# List all 36 supported products
ml4t-data download-cot --list-products

# Download specific products (2020-present)
ml4t-data download-cot -p ES -p CL -p GC --start-year 2020

# Download to custom location
ml4t-data download-cot -p ES -p NQ -o ~/ml4t-data/cot
```

**Key Details**:
- ✅ **FREE** - Direct from CFTC, no API key needed
- ✅ 36 products: equity indices, currencies, bonds, crypto, commodities
- ✅ Weekly positioning by trader type (hedge funds, commercials, etc.)
- ✅ Net positioning columns pre-computed
- ✅ Hive-partitioned Parquet storage

**Available Products**:

| Category | Products |
|----------|----------|
| **Equity Indices** | ES, NQ, RTY, YM |
| **Currencies** | 6E, 6J, 6B, 6C, 6A, 6S, 6M, 6N |
| **Interest Rates** | ZN, ZB, ZF, ZT, SR3 |
| **Crypto** | BTC, ETH |
| **Energy** | CL, NG, RB, HO |
| **Metals** | GC, SI, HG, PL |
| **Agriculture** | ZC, ZW, ZS, ZM, ZL |
| **Livestock** | LE, HE, GF |

**Data Columns**:

*Financial Futures (ES, NQ, bonds, FX, crypto)*:
- `dealer_net` - Bank/swap dealer positioning
- `asset_mgr_net` - Institutional investor positioning
- `lev_money_net` - Hedge fund positioning
- `nonrept_net` - Small trader positioning

*Commodity Futures (CL, NG, GC, grains)*:
- `commercial_net` - Commercial hedger positioning
- `managed_money_net` - Hedge fund/CTA positioning
- `nonrept_net` - Small trader positioning

**ML Feature Ideas**:
```python
import polars as pl

# Load COT data
cot = pl.read_parquet("~/ml4t-data/cot/product=ES/data.parquet")

# Net positioning features
cot = cot.with_columns([
    # Week-over-week change in hedge fund positioning
    (pl.col("lev_money_net") - pl.col("lev_money_net").shift(1)).alias("lev_money_change"),

    # Z-score of positioning (extreme detection)
    ((pl.col("lev_money_net") - pl.col("lev_money_net").rolling_mean(52)) /
     pl.col("lev_money_net").rolling_std(52)).alias("lev_money_zscore"),

    # Commercial vs speculative divergence
    (pl.col("dealer_net") - pl.col("lev_money_net")).alias("comm_spec_divergence"),
])
```

**Why COT over Daily Open Interest**:

| | Daily OI (Databento) | Weekly COT (Free) |
|--|---------------------|-------------------|
| **Cost** | ~$300+ for 1 year | **FREE** |
| **Frequency** | Daily | Weekly |
| **Content** | Total OI only | OI breakdown by trader type |
| **ML Value** | Low | **High** (sentiment signals) |

**[CFTC COT Reports](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)** | **[Report Types Explained](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes/index.htm)**

---

## Incremental Updates

All providers support the `ProviderUpdater` pattern for smart, incremental updates:

```python
from ml4t.data.providers import TiingoProvider, TiingoUpdater
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.backend import StorageConfig

# Setup
storage = HiveStorage(StorageConfig(base_path="./data"))
provider = TiingoProvider(api_key="your_key")
updater = TiingoUpdater(provider, storage)

# First update: Downloads default history (30 days for stocks, 90 days for crypto)
result = updater.update_symbol("AAPL", incremental=True)
print(f"Downloaded {result['records_fetched']} records")

# Second update: Only fetches NEW data since last update
result = updater.update_symbol("AAPL", incremental=True)
if result.get("skip_reason") == "already_up_to_date":
    print("Already up to date!")
else:
    print(f"Added {result['records_added']} new records")

# Dry run (preview without saving)
result = updater.update_symbol("AAPL", incremental=True, dry_run=True)
print(f"Would download {result['records_fetched']} records")
```

**Benefits**:
- Only fetches new data (saves API calls)
- Automatic gap detection
- Respects rate limits
- Dry-run mode for testing

---

## Running Examples

```bash
# 5-minute quickstart
python examples/quickstart.py

# Individual provider examples
python examples/eodhd_example.py
python examples/finnhub_example.py

# Run integration tests (requires API keys)
pytest tests/integration/test_tiingo.py -v
pytest tests/integration/test_eodhd.py -v
pytest tests/integration/test_coingecko.py -v
```

## Learning Resources

### Interactive Notebooks (Hands-On!)
- **[01: Quickstart Notebook](examples/notebooks/01_quickstart.ipynb)** 📓 - Get started in 5 minutes
- More notebooks: [Provider comparison, Incremental updates, Multi-asset, Data quality](examples/notebooks/)

### Tutorials (Start Here!)
- **[01: Understanding OHLCV Data](docs/tutorials/01_understanding_ohlcv.md)** - Learn the fundamentals
- **[02: Rate Limiting Best Practices](docs/tutorials/02_rate_limiting.md)** - Avoid API bans
- **[03: Incremental Updates](docs/tutorials/03_incremental_updates.md)** - 100-1000x fewer API calls
- **[04: Data Quality Validation](docs/tutorials/04_data_quality.md)** - Ensure data integrity
- **[05: Multi-Provider Strategies](docs/tutorials/05_multi_provider.md)** - Build resilient pipelines

### Developer Guides
- **[Provider Selection Guide](docs/provider-selection-guide.md)** - Choose the right provider
- **[Creating a Provider](docs/creating_a_provider.md)** - Add new data sources
- **[Extending ML4T Data](docs/extending_ml4t_data.md)** - Architecture and patterns
- **[Contributing](CONTRIBUTING.md)** - Join the project
## Installation

```bash
# Basic installation
pip install ml4t-data

# With specific providers
pip install ml4t-data[yahoo]          # Yahoo Finance support
pip install ml4t-data[databento]      # Databento support
pip install ml4t-data[all-providers]  # All providers

# Development installation
pip install ml4t-data[dev]            # Include testing tools
```

## Quick Start

### Basic Usage (Fetch Only)
```python
from ml4t.data.data_manager import DataManager

# Create manager (no storage needed for fetch-only)
manager = DataManager()

# Fetch equity data
spy_data = manager.fetch(
    symbol="SPY",
    start="2023-01-01",
    end="2023-12-31",
    provider="yahoo"
)

# Cryptocurrency data (auto-routed to crypto provider)
btc_data = manager.fetch(
    symbol="BTC-USD",
    start="2023-01-01",
    end="2023-12-31"
)

# Futures data (requires API key)
es_data = manager.fetch(
    symbol="ES.FUT.C.0",
    start="2023-01-01",
    end="2023-12-31",
    provider="databento"
)
```

### With Storage (Load & Update)
```python
from ml4t.data.data_manager import DataManager
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.backend import StorageConfig

# Create storage backend
storage = HiveStorage(StorageConfig(base_path="./data"))

# Create manager with storage and transactions
manager = DataManager(
    storage=storage,
    use_transactions=True,  # ACID guarantees
    enable_validation=True   # Data quality checks
)

# Initial load - fetches and stores data
key = manager.load(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    provider="yahoo"
)
print(f"Stored at: {key}")  # e.g., "equities/daily/AAPL"

# Incremental update - only fetches new data
key = manager.update(
    symbol="AAPL",
    fill_gaps=True  # Automatically detect and fill gaps
)
```

### Configuration-Driven Updates (Recommended for Production)

For managing multiple datasets, use YAML configuration files with the CLI:

```yaml
# ml4t-data.yaml
storage:
  path: ~/ml4t-data

datasets:
  # Small dataset with inline symbols (good for 5-50 symbols)
  demo_stocks:
    provider: yahoo
    frequency: daily
    symbols: [AAPL, MSFT, GOOGL, JPM, XOM]

  # Large dataset with file-based symbols (good for 500+ symbols)
  sp500_full:
    provider: yahoo
    frequency: daily
    symbols_file: sp500.txt  # One symbol per line, # for comments

  # Crypto dataset
  crypto:
    provider: cryptocompare
    frequency: hourly
    symbols: [BTC, ETH]
```

**Usage:**
```bash
# Update all datasets
ml4t-data update-all -c ml4t-data.yaml

# Update specific dataset only
ml4t-data update-all -c ml4t-data.yaml --dataset sp500_full

# Preview what would be updated (dry-run)
ml4t-data update-all -c ml4t-data.yaml --dry-run

# List all stored data
ml4t-data list -c ml4t-data.yaml

# Automate with cron (daily at 6 PM)
# crontab -e
0 18 * * * source ~/.bashrc && ml4t-data update-all -c ~/ml4t-data.yaml
```

**Symbol file format:**
```txt
# sp500.txt - One symbol per line
# Lines starting with # are comments

# Technology
AAPL
MSFT
GOOGL

# Financial
JPM
BAC
WFC
```

**Benefits:**
- ✅ **Scalable**: Handle 5,000+ symbols without cluttering config
- ✅ **Version Control Friendly**: Track symbol changes with git
- ✅ **Reusable**: Share symbol lists across projects
- ✅ **Clean**: Config stays small and readable
- ✅ **Flexible**: Mix inline and file-based symbols

**Getting Started:**
```bash
# Copy starter config
cp examples/configs/ml4t-starter.yaml ~/ml4t-config.yaml

# Edit with your symbols and API keys
vim ~/ml4t-config.yaml

# Run initial update
ml4t-data update-all -c ~/ml4t-config.yaml

# Fetch still works without storage
df = manager.fetch("GOOGL", "2024-01-01", "2024-12-31")
```

### CLI Interface
```bash
# Fetch data
ml4t-data fetch SPY --start 2023-01-01 --end 2023-12-31

# Update existing data
ml4t-data update SPY

# Check data status
ml4t-data status SPY --detailed

# Validate data quality
ml4t-data validate SPY --check-gaps --check-ohlc

# Download futures data (Databento)
ml4t-data download-futures -c configs/futures_download.yaml

# Download COT positioning data (FREE)
ml4t-data download-cot -p ES -p CL -p GC --start-year 2020
```

## Storage Features

### Hive Partitioned Storage
```python
from ml4t.data.storage import create_storage

storage = create_storage("/data", strategy="hive")

# Store data with year/month partitioning
storage.write(data_df, "BTC-USD")

# Read with date filtering (efficient)
filtered_data = storage.read(
    "BTC-USD",
    start_date=datetime(2023, 6, 1),
    end_date=datetime(2023, 12, 31)
)
```

### Metadata Tracking
```python
# Get data lineage and quality metrics
metadata = storage.get_metadata("BTC-USD")
print(f"Last updated: {metadata['last_updated']}")
print(f"Row count: {metadata['row_count']}")
print(f"Schema: {metadata['schema']}")
```

## Advanced Usage

### Provider Configuration
```python
# Configure with API keys
config = Config(
    data_root="/path/to/data",
    provider_configs={
        "databento": {"api_key": "your_api_key"},
        "oanda": {"api_key": "your_api_key", "account_id": "your_account"},
    }
)

manager = DataManager(config)
```

### Incremental Updates
```python
# Automatically detect and fill gaps
update_manager = UpdateManager(storage)
update_manager.update_symbol(
    symbol="SPY",
    provider="yahoo",
    lookback_days=30  # Check last 30 days for gaps
)
```

### Data Validation
```python
from ml4t.data.validation import DataValidator

validator = DataValidator()

# Check OHLC invariants
issues = validator.validate_ohlc(data_df)
if issues:
    print(f"Found {len(issues)} validation issues")

# Detect anomalies
anomalies = validator.detect_anomalies(data_df)
```

## Supported Providers

| Provider | Asset Classes | Free Tier | Rate Limit | Status |
|----------|--------------|-----------|------------|--------|
| Yahoo Finance | Stocks, ETFs, Indices | Unlimited | 2000/hour | ✅ Stable |
| Binance | Crypto Spot, Futures | Yes | 1200/min | ✅ Stable |
| CryptoCompare | Crypto Spot | 100K/month | 50/min | ✅ Stable |
| Databento | Futures, Options | Trial | Varies | ✅ Stable |
| OANDA | Forex | Demo Account | 120/sec | ✅ Stable |

## API Documentation

### DataManager API

```python
from ml4t-data import DataManager, Config

class DataManager:
    """Unified interface for market data management."""

    def fetch_data(
        self,
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        provider: str = "yahoo",
        **kwargs
    ) -> pl.DataFrame:
        """Fetch OHLCV data from specified provider."""

    def update_data(
        self,
        symbol: str,
        provider: str = "yahoo"
    ) -> UpdateResult:
        """Update existing data with latest bars."""

    def get_status(
        self,
        symbol: str
    ) -> DataStatus:
        """Get data availability and quality metrics."""
```

### Storage API

```python
from ml4t.data.storage import HiveStorage

class HiveStorage:
    """Hive-partitioned Parquet storage."""

    def write(
        self,
        data: pl.DataFrame,
        symbol: str
    ) -> None:
        """Write data with year/month partitioning."""

    def read(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> pl.DataFrame:
        """Read data with efficient date filtering."""
```
```

### 2. Binance Provider Inheritance (CRITICAL)
**Issue**: Wrong base class causes failures

**Current Code** (src/ml4t/data/providers/binance.py:17):
```python
from ml4t.data.providers.base import Provider  # ❌ Wrong import
```

**Fix Needed**:
```python
from ml4t.data.providers.base import BaseProvider  # ✅ Correct import
```

### 3. Storage Performance (HIGH)
**Issue**: HiveStorage forces eager evaluation at line 62-65

**Current Code**:
```python
df = lazy_data.collect()  # ❌ Defeats lazy evaluation
```

**Fix Needed**: Process partitions lazily

### 4. Rate Limiting (MEDIUM)
**Issue**: Per-instance rate limiting allows bypassing limits

**Current**: Each provider instance has its own limiter
**Fix Needed**: Global rate limiting per provider type

## Provider Status Details

### ✅ CryptoCompare (Working)
- **API**: Free tier 100k calls/month
- **Assets**: Major cryptocurrencies
- **Features**: OHLCV, multiple exchanges
- **Issues**: None major

### ✅ Databento (Working)
- **API**: Professional data (paid)
- **Assets**: Futures, options, equities
- **Features**: Multiple schemas, tick data
- **Issues**: Requires API key

### ✅ OANDA (Working)
- **API**: Forex data
- **Assets**: Currency pairs
- **Features**: Real-time and historical
- **Issues**: Limited free tier

### ❌ Yahoo Finance (Best Provider, Not Registered)
- **API**: Free, no key required
- **Assets**: Stocks, ETFs, crypto, forex
- **Features**: Excellent coverage, reliable
- **Issues**: Not in PROVIDER_CLASSES map
- **Workaround**: Import directly

### ❌ Binance (Broken)
- **API**: Free, high rate limits
- **Assets**: Cryptocurrency spot/futures
- **Features**: Real-time data, no API key needed
- **Issues**: Wrong inheritance, not registered
- **Fix**: Change import + register

### ❌ Mock Provider (Not Registered)
- **Purpose**: Testing and development
- **Features**: Synthetic data generation
- **Issues**: Not in PROVIDER_CLASSES map
- **Use**: Development only

## Performance Benchmarks

### Storage Performance (1M rows)
| Operation | Current | Optimal | Issue |
|-----------|---------|---------|--------|
| Write | 2.3s | 0.8s | Forced eager eval |
| Read (filtered) | 0.4s | 0.2s | Manual pruning |
| Metadata query | 0.05s | 0.05s | ✅ Good |

### Provider Performance
| Provider | Latency | Rate Limit | Free Tier |
|----------|---------|------------|-----------|
| CryptoCompare | 200ms | 10/min | 100k/month |
| Yahoo Finance | 150ms | 2000/hour | Unlimited |
| Binance | 100ms | 1200/min | Unlimited |

## Configuration

### Basic Configuration
```python
from ml4t-data import Config

config = Config(
    data_root="/data/ml4t-data",
    log_level="INFO",

    # Storage settings
    storage_strategy="hive",
    compression="snappy",

    # Provider settings
    default_provider="cryptocompare",
    rate_limit_requests=10,
    rate_limit_period=60,

    # Retry settings
    max_retries=3,
    backoff_factor=2.0
)
```

### Environment Variables
```bash
export ML4T_DATA_ROOT="/data/ml4t-data"
export CRYPTOCOMPARE_API_KEY="your_key"
export DATABENTO_API_KEY="your_key"
export OANDA_API_KEY="your_key"
```

## Development

### Running Tests
```bash
# All tests
pytest tests/tests/

# Specific provider tests
pytest tests/tests/test_yahoo_provider.py

# Real API tests (minimal usage)
pytest tests/tests/integration/test_real_api_integration.py

# Mock provider tests (safe for development)
pytest tests/tests/test_mock_provider.py
```

### Adding a New Provider

1. **Create provider class** inheriting from `BaseProvider`
2. **Implement abstract methods**: `_fetch_raw_data()`, `_transform_data()`
3. **Add rate limiting configuration**
4. **Write comprehensive tests** (mock + minimal real API)
5. **Register in PROVIDER_CLASSES** (don't forget this!)
6. **Update documentation**

### Code Quality
```bash
make quality  # Run all checks
make format   # Format code
make lint     # Lint code
make type     # Type check
```

## Integration with QuantLab

ML4T Data outputs standardized DataFrames compatible with other QuantLab libraries:

```python
# Standard schema
DataFrame {
    timestamp: datetime64[ns, UTC],  # Event timestamp
    open: float64,                   # Open price
    high: float64,                   # High price
    low: float64,                    # Low price
    close: float64,                  # Close price
    volume: float64,                 # Volume traded
    symbol: str,                     # Asset identifier
}
```

## Migration from Monorepo

If you're migrating from the QuantLab monorepo:

```bash
# Old monorepo import
from ml4t.data import DataManager

# New standalone import
from ml4t-data import DataManager
```

All APIs remain backward compatible.

## Documentation

**📚 [Complete Documentation Index](docs/README.md)** - All guides, tutorials, and references organized by topic

Quick links:
- **[Asset Class Guides](docs/asset-classes/README.md)** - Crypto, Stocks, Forex, Futures
- **[Tutorials](docs/tutorials/README.md)** - Step-by-step learning
- **[Creating a Provider - Extend ML4T Data

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ml4t/data.git
cd data

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

## Support

- **Documentation**: [https://ml4t-data.readthedocs.io](https://ml4t-data.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/ml4t/data/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ml4t/data/discussions)
- **Security**: Report vulnerabilities to security@ml4trading.io

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use ML4T Data in your research, please cite:

```bibtex
@software{ml4t-data,
  title = {ML4T Data: High-Performance Market Data Management},
  author = {ML4T Team},
  year = {2024},
  url = {https://github.com/ml4t/data}
}
```
