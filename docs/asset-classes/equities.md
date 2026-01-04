# Equities (Stocks) Data Guide

**Asset Classes**: US Stocks, Global Stocks, ETFs, Indices
**Available Providers**: 7 providers (5 US, 2 Global, 3 Multi-asset)
**Difficulty**: Moderate (requires API keys, understanding of market hours)
**Recommended For**: Stock trading strategies, portfolio backtesting, factor research

---

## Overview

ML4T Data provides **multiple free-tier providers** for equity data, ranging from US-only to global multi-exchange coverage. Stock data has unique characteristics compared to crypto:

- **Market Hours**: Trading typically 9:30 AM - 4:00 PM ET (NYSE/NASDAQ)
- **Weekends & Holidays**: No data on weekends, market holidays
- **Corporate Actions**: Splits, dividends require adjusted close prices
- **Exchange Specificity**: Same symbol may trade on multiple exchanges

**Quick Provider Comparison**:

| Provider | US Stocks | Global | Free Tier | API Key | Best For |
|----------|-----------|--------|-----------|---------|----------|
| **Tiingo** | ✅ | ❌ | 1000/day | Required | High-quality US stocks |
| **IEX Cloud** | ✅ | ❌ | 50K/month | Required | Fundamentals + OHLCV |
| **Alpha Vantage** | ✅ | ⚠️ Limited | 25/day | Required | Research (low volume) |
| **EODHD** | ✅ | ✅ 60+ exchanges | 500/day | Required | Global coverage |
| **Finnhub** | ✅ | ✅ 70+ exchanges | Real-time only | Required | Professional (paid OHLCV) |
| **Twelve Data** | ✅ | ✅ | 800/day | Required | Multi-asset flexibility |
| **Polygon** | ✅ | ✅ | 5 calls/min | Required | Real-time + historical |

---

## US Stocks Providers

### Tiingo (Recommended for US Stocks)

#### Why Tiingo?

- ✅ **High-quality data** - Sourced from IEX Exchange
- ✅ **1000 API calls/day** free tier (generous)
- ✅ **500 unique symbols/month** free tier
- ✅ **Adjusted close** - Handles splits and dividends automatically
- ✅ **Excellent coverage** - All major US exchanges (NYSE, NASDAQ, AMEX)

**Best for**: US stock backtesting, academic research, portfolio analysis

#### Quick Start

```python
from ml4t.data.providers import TiingoProvider

# Get free API key from: https://www.tiingo.com/account/api/token
provider = TiingoProvider(api_key="your_tiingo_api_key")

# Fetch Apple daily data
df = provider.fetch_ohlcv(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)

print(df.head())
```

**Output**:
```
shape: (252, 7)
┌─────────────────────┬──────────┬───────────┬───────────┬───────────┬───────────┬──────────────┐
│ timestamp           │ symbol   │ open      │ high      │ low       │ close     │ volume       │
│ ---                 │ ---      │ ---       │ ---       │ ---       │ ---       │ ---          │
│ datetime[μs]        │ str      │ f64       │ f64       │ f64       │ f64       │ f64          │
╞═════════════════════╪══════════╪═══════════╪═══════════╪═══════════╪═══════════╪══════════════╡
│ 2024-01-02 00:00:00 │ AAPL     │ 187.15    │ 188.44    │ 183.89    │ 185.64    │ 8.26e7       │
│ 2024-01-03 00:00:00 │ AAPL     │ 184.22    │ 185.88    │ 183.43    │ 184.25    │ 5.80e7       │
└─────────────────────┴──────────┴───────────┴───────────┴───────────┴───────────┴──────────────┘
```

#### Features

**Supported Frequencies**:
```python
# Daily (most common)
df = provider.fetch_ohlcv(symbol="AAPL", frequency="daily")

# Weekly
df = provider.fetch_ohlcv(symbol="AAPL", frequency="weekly")

# Monthly
df = provider.fetch_ohlcv(symbol="AAPL", frequency="monthly")
```

**Adjusted Close Handling**:
```python
# Tiingo automatically provides adjusted close prices
df = provider.fetch_ohlcv(symbol="AAPL", start="2020-01-01", end="2024-01-01")

# The 'close' column accounts for:
# - Stock splits (e.g., AAPL 4:1 split in Aug 2020)
# - Dividend distributions
# - Other corporate actions
```

**Symbol Coverage**:
- All NYSE stocks (~2,800 symbols)
- All NASDAQ stocks (~3,300 symbols)
- AMEX, OTC markets
- ETFs and mutual funds

#### Rate Limits

**Free Tier**:
- 1000 API calls per day
- 500 unique symbols per month
- No intraday data

**Paid Tier** ($30/month):
- 20,000 API calls per hour
- Unlimited symbols
- Intraday data (1-minute, 5-minute bars)

**Usage Strategy**:
```python
# Example: Fetch S&P 500 constituents (500 symbols)
# Fits within free tier: 500 symbols, 500 API calls

sp500_symbols = ["AAPL", "MSFT", "GOOGL", ...]  # 500 symbols

for symbol in sp500_symbols:
    df = provider.fetch_ohlcv(
        symbol=symbol,
        start="2024-01-01",
        end="2024-12-31",
        frequency="daily"
    )
    # 500 API calls total - well within 1000/day limit
```

#### Incremental Updates

```python
from ml4t.data.provider_updater import ProviderUpdater

# Initialize updater
updater = ProviderUpdater(provider="tiingo")

# First run: Fetch all history
df = updater.update(
    symbol="AAPL",
    frequency="daily",
    lookback_days=365  # 1 year of data
)
# Uses 1 API call

# Subsequent runs: Only fetch new data
df = updater.update(
    symbol="AAPL",
    frequency="daily",
    lookback_days=7  # Only check last 7 days
)
# Uses 1 API call, merges with existing data
# 100-1000x more efficient than re-fetching history!
```

#### When to Use Tiingo

✅ **Good for**:
- US stock backtesting
- Daily/weekly/monthly strategies
- Academic research
- Portfolio analysis
- 100-500 symbols daily updates

❌ **Not ideal for**:
- Global stocks (US only)
- Intraday strategies (free tier)
- High-frequency trading
- Real-time quotes

---

### IEX Cloud (Fundamentals + OHLCV)

#### Why IEX Cloud?

- ✅ **50,000 message credits/month** free tier
- ✅ **Company fundamentals** - Financials, earnings, news
- ✅ **Real-time + historical** OHLCV data
- ✅ **IEX Exchange** - The exchange that inspired Flash Boys

**Best for**: Fundamental + technical analysis, news-driven strategies

#### Quick Start

```python
from ml4t.data.providers import IEXCloudProvider

# Get free API key from: https://iexcloud.io/
provider = IEXCloudProvider(api_key="your_iex_cloud_api_key")

# Fetch OHLCV data
df = provider.fetch_ohlcv(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)

# Fetch company fundamentals
info = provider.fetch_company_info("AAPL")
print(info)
# {'companyName': 'Apple Inc.', 'sector': 'Technology', ...}
```

#### Features

**OHLCV Data**:
- Daily, weekly, monthly frequencies
- Adjusted for splits and dividends
- Real-time quotes (delayed 15 minutes on free tier)

**Fundamentals** (bonus):
- Company information
- Financial statements (income, balance sheet, cash flow)
- Earnings reports
- News and press releases
- Analyst recommendations

**Message Credits System**:
```python
# 1 OHLCV record ≈ 1 credit
# Fetch 1 year of daily data (252 trading days) = 252 credits
df = provider.fetch_ohlcv(symbol="AAPL", start="2024-01-01", end="2024-12-31")
# Cost: ~252 credits

# Free tier: 50,000 credits/month
# Can fetch ~198 symbols × 252 days = 50,000 credits
```

#### Rate Limits

**Free Tier**:
- 50,000 message credits per month
- ~198 symbols with 1 year of daily data
- Resets monthly

**Paid Tiers**:
- Start: $9/month (1M messages)
- Grow: $99/month (100M messages)
- Scale: $999/month (1B messages)

#### When to Use IEX Cloud

✅ **Good for**:
- Fundamental + technical combined strategies
- News-driven backtests
- Moderate symbol counts (50-200)
- Real-time quote monitoring

❌ **Not ideal for**:
- Large universes (500+ symbols)
- Global stocks (US only)
- High-frequency updates

---

### Alpha Vantage (Low-Volume Research)

#### Why Alpha Vantage?

- ✅ **Multi-asset support** - Stocks, forex, crypto
- ✅ **Fundamentals available** - Company overviews, earnings
- ✅ **Technical indicators** - Built-in SMA, EMA, RSI, etc.
- ⚠️ **Very restrictive free tier** - Only 25 calls/day

**Best for**: Research projects with low API usage, multi-asset portfolios

#### Quick Start

```python
from ml4t.data.providers import AlphaVantageProvider

# Get free API key from: https://www.alphavantage.co/support/#api-key
provider = AlphaVantageProvider(api_key="your_alpha_vantage_api_key")

# Fetch daily data
df = provider.fetch_ohlcv(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)
```

#### Rate Limits

**Free Tier** (Very Restrictive):
- 5 API calls per minute
- 25 API calls per day
- No intraday data

**Paid Tier** ($49.99/month):
- 75 API calls per minute
- Intraday data (1min, 5min, 15min, 30min, 60min)
- Extended historical data

**Critical: Use with ProviderUpdater**:
```python
from ml4t.data.provider_updater import ProviderUpdater

# DON'T: Fetch full history every time (wastes API quota)
for symbol in symbols:
    df = provider.fetch_ohlcv(symbol, start="2020-01-01", end="2024-12-31")
    # 100 symbols = 100 API calls = 4 days on free tier!

# DO: Use incremental updates
updater = ProviderUpdater(provider="alpha_vantage")
for symbol in symbols:
    df = updater.update(symbol, frequency="daily", lookback_days=7)
    # Only fetch last 7 days, merge with cached data
    # 100 symbols = 100 API calls = 1 day batch, but then only 100 calls/month
```

#### When to Use Alpha Vantage

✅ **Good for**:
- Research projects with minimal API usage
- Multi-asset portfolios (stocks + forex + crypto)
- Technical indicator exploration
- Small symbol universes (5-10 symbols)

❌ **Not ideal for**:
- Production systems (too restrictive)
- Large symbol universes
- Daily batch updates (will hit 25/day limit)
- Time-sensitive applications

---

## Global Stocks Providers

### EODHD (Best Value for Global Coverage)

#### Why EODHD?

- ✅ **60+ global exchanges** - US, Europe, Asia, emerging markets
- ✅ **150,000+ tickers** worldwide
- ✅ **500 API calls/day free** or **€19.99/month unlimited**
- ✅ **Most affordable global data** vs competitors
- ✅ **Adjusted close** for splits and dividends

**Best for**: Global equity strategies, international portfolios, emerging markets

#### Quick Start

```python
from ml4t.data.providers import EODHDProvider

# Get free API key from: https://eodhd.com/register
provider = EODHDProvider(
    api_key="your_eodhd_api_key",
    exchange="US"  # Default exchange
)

# US stocks (AAPL.US format)
us_data = provider.fetch_ohlcv(
    symbol="AAPL.US",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)

# London Stock Exchange (VOD.LSE)
lse_data = provider.fetch_ohlcv(
    symbol="VOD.LSE",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)

# Frankfurt Stock Exchange (BMW.FRA)
fra_data = provider.fetch_ohlcv(
    symbol="BMW.FRA",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)
```

#### Global Exchange Coverage

**Major Exchanges**:

| Region | Exchange | Code | Symbols | Example |
|--------|----------|------|---------|---------|
| **US** | NYSE, NASDAQ | US | 51,000+ | AAPL.US, MSFT.US |
| **UK** | London | LSE | 2,000+ | VOD.LSE, BP.LSE |
| **Germany** | Frankfurt | FRA | 1,000+ | BMW.FRA, SAP.FRA |
| **Japan** | Tokyo | TSE | 3,700+ | 7203.TSE (Toyota) |
| **France** | Euronext Paris | PA | 700+ | AI.PA (Air Liquide) |
| **Canada** | Toronto | TO | 3,000+ | SHOP.TO (Shopify) |
| **China** | Shanghai | SS | 1,500+ | 600519.SS (Moutai) |
| **Hong Kong** | HKEX | HK | 2,000+ | 0700.HK (Tencent) |
| **India** | NSE | NSE | 2,000+ | RELIANCE.NSE |
| **Australia** | ASX | AU | 2,000+ | BHP.AU |

**Plus**: Brazil (SA), South Korea (KO), Singapore (SI), Switzerland (SW), and 50+ more!

#### Symbol Format

EODHD uses **SYMBOL.EXCHANGE** notation:

```python
# US stocks
"AAPL.US", "MSFT.US", "TSLA.US"

# London Stock Exchange
"VOD.LSE"   # Vodafone
"BP.LSE"    # British Petroleum
"HSBA.LSE"  # HSBC

# Frankfurt
"BMW.FRA"   # BMW
"SAP.FRA"   # SAP
"DTE.FRA"   # Deutsche Telekom

# Tokyo (use numerical codes)
"7203.TSE"  # Toyota
"6758.TSE"  # Sony
"9984.TSE"  # SoftBank
```

#### Free Tier vs Paid Tier

**Free Tier** (500 API calls/day):
- ✅ 1 year historical depth
- ✅ 150,000+ tickers across 60+ exchanges
- ✅ Daily, weekly, monthly OHLCV
- ✅ Adjusted close (splits/dividends)
- ⚠️ 500 calls/day limit

**Paid Tier** (€19.99/month - **Best Value**):
- ✅ **Unlimited API calls**
- ✅ Extended historical coverage (up to 30 years)
- ✅ Real-time data
- ✅ Fundamentals and financials
- ✅ Dividends and splits calendar
- ✅ Bulk downloads

**Cost Comparison**:
| Provider | Global Coverage | Free Tier | Paid Tier |
|----------|-----------------|-----------|-----------|
| EODHD | 60+ exchanges | 500/day | €19.99/month |
| Finnhub | 70+ exchanges | Real-time only | $59.99/month |
| Polygon | Limited global | 5/min | $199/month |
| **Winner** | EODHD | EODHD | **EODHD** (cheapest) |

#### Rate Limiting Strategy

```python
# Free tier: 500 calls/day
# Strategy: Batch updates wisely

# Example 1: Daily updates for 500 global stocks
symbols = [f"AAPL.US", "VOD.LSE", "BMW.FRA", ...]  # 500 symbols

for symbol in symbols:
    df = provider.fetch_ohlcv(symbol, start="2024-01-01", end="2024-12-31")
    # 500 API calls = 1 day's quota (perfect fit!)

# Example 2: Use incremental updates to reduce calls
from ml4t.data.provider_updater import ProviderUpdater

updater = ProviderUpdater(provider="eodhd")

# First run: 500 calls (fetch full history)
for symbol in symbols:
    df = updater.update(symbol, frequency="daily", lookback_days=365)

# Daily runs: Only fetch new data (much fewer calls)
for symbol in symbols:
    df = updater.update(symbol, frequency="daily", lookback_days=7)
    # Only 1 API call per symbol if data is current
```

#### Multi-Exchange Portfolio Example

```python
from ml4t.data.providers import EODHDProvider
import polars as pl

provider = EODHDProvider(api_key="your_key")

# Build a global tech portfolio
symbols = {
    "AAPL.US": "Apple (US)",
    "MSFT.US": "Microsoft (US)",
    "VOD.LSE": "Vodafone (UK)",
    "SAP.FRA": "SAP (Germany)",
    "6758.TSE": "Sony (Japan)",
    "0700.HK": "Tencent (Hong Kong)",
}

all_data = []
for symbol, name in symbols.items():
    df = provider.fetch_ohlcv(
        symbol=symbol,
        start="2024-01-01",
        end="2024-12-31",
        frequency="daily"
    )
    df = df.with_columns(pl.lit(name).alias("company"))
    all_data.append(df)

# Combine all markets
global_portfolio = pl.concat(all_data)

# Calculate returns in local currency
returns = global_portfolio.group_by("symbol").agg(
    pl.col("close").pct_change().mean().alias("avg_daily_return")
)

print(returns)
```

#### When to Use EODHD

✅ **Good for**:
- Global equity strategies
- International portfolios
- Emerging markets research
- Multi-exchange analysis
- Cost-conscious production systems (€20/month unlimited!)

❌ **Not ideal for**:
- Real-time intraday trading (free tier)
- Ultra-high-frequency strategies
- When 500 calls/day free tier is insufficient and budget is <€20/month

---

### Finnhub (Professional Grade - Paid for Historical)

#### Why Finnhub?

- ✅ **70+ global exchanges**
- ✅ **Real-time quotes** on free tier (60/min)
- ✅ **WebSocket support** for live data
- ⚠️ **Historical OHLCV requires paid subscription**

**Best for**: Professional trading systems with real-time requirements

#### Quick Start

```python
from ml4t.data.providers import FinnhubProvider

# Get API key from: https://finnhub.io/register
provider = FinnhubProvider(api_key="your_finnhub_api_key")

# Real-time quote (FREE TIER)
quote = provider.fetch_quote("AAPL")
print(quote)
# {'c': 185.64, 'h': 188.44, 'l': 183.89, 'o': 187.15, ...}

# Historical OHLCV (PAID SUBSCRIPTION REQUIRED)
df = provider.fetch_ohlcv(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)
# Only works with paid plan ($59.99+/month)
```

#### Pricing

**Free Tier**:
- ✅ Real-time quotes (60 calls/min)
- ✅ 50 symbols via WebSocket
- ❌ Historical OHLCV (**paid only**)

**Paid Tier** ($59.99/month minimum):
- ✅ Historical daily/weekly/monthly OHLCV
- ✅ 70+ global exchanges
- ✅ Forex and crypto historical data
- ✅ Company fundamentals
- ✅ News and sentiment data

#### When to Use Finnhub

✅ **Good for**:
- Real-time quote monitoring (free tier)
- Professional trading systems (paid tier)
- News and sentiment analysis
- Multi-asset strategies

❌ **Not ideal for**:
- Budget-conscious projects (EODHD is 3x cheaper)
- Historical backtesting on free tier (not available)
- Research projects (too expensive vs alternatives)

---

## Multi-Asset Providers

### Twelve Data (Stocks + Forex + Crypto)

**Free Tier**: 800 API calls per day
**Coverage**: Stocks, forex, crypto, indices
**Best for**: Multi-asset portfolios

```python
from ml4t.data.providers import TwelveDataProvider

provider = TwelveDataProvider(api_key="your_key")

# Stocks
stocks = provider.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-12-31")

# Forex
forex = provider.fetch_ohlcv("EUR/USD", start="2024-01-01", end="2024-12-31")

# Crypto
crypto = provider.fetch_ohlcv("BTC/USD", start="2024-01-01", end="2024-12-31")
```

**When to use**: Need stocks + forex + crypto in single provider

---

### Polygon (Real-Time + Historical)

**Free Tier**: 5 API calls per minute
**Coverage**: Stocks, forex, crypto, options
**Best for**: Real-time data needs

```python
from ml4t.data.providers import PolygonProvider

provider = PolygonProvider(api_key="your_key")
df = provider.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-12-31")
```

**Pricing**: Free (limited) or $199-$399/month

**When to use**: Real-time stock/options/crypto trading

---

## Best Practices

### 1. Use Incremental Updates

**Don't re-fetch entire history every day**:

```python
# ❌ Bad: Wastes API quota
for symbol in symbols:
    df = provider.fetch_ohlcv(symbol, start="2020-01-01", end="2024-12-31")
    # 4+ years × 252 days = 1000+ records per symbol
    # 100 symbols = 100 API calls for redundant data

# ✅ Good: Only fetch new data
from ml4t.data.provider_updater import ProviderUpdater

updater = ProviderUpdater(provider="tiingo")

# First run: Fetch all history (once)
df = updater.update("AAPL", frequency="daily", lookback_days=365)

# Daily runs: Only fetch last week, merge with cache
df = updater.update("AAPL", frequency="daily", lookback_days=7)
# 100-1000x fewer API calls!
```

**See**: [Tutorial 03: Incremental Updates](../tutorials/03_incremental_updates.md)

### 2. Handle Market Hours & Holidays

Stocks don't trade 24/7:

```python
from datetime import datetime, timedelta

# ❌ Wrong: Assuming data exists for weekends
df = provider.fetch_ohlcv("AAPL", start="2024-11-02", end="2024-11-03")
# Nov 2-3 is a weekend - no data!

# ✅ Correct: Check for gaps
df = provider.fetch_ohlcv("AAPL", start="2024-11-01", end="2024-11-08")

# Verify trading days only
gaps = df.select("timestamp").diff().filter(
    pl.col("timestamp") > timedelta(days=3)  # More than weekend
)

if not gaps.is_empty():
    print(f"Found {len(gaps)} market holidays or missing data")
```

**US Market Holidays 2024**:
- New Year's Day, MLK Day, Presidents' Day, Good Friday
- Memorial Day, Juneteenth, Independence Day, Labor Day
- Thanksgiving, Christmas

### 3. Adjusted vs Unadjusted Prices

Always use **adjusted close** for backtesting:

```python
# Most providers return adjusted close by default (ML4T Data standard)
df = provider.fetch_ohlcv("AAPL", start="2020-01-01", end="2024-12-31")

# Check if close is adjusted
# Example: AAPL had 4:1 split on Aug 31, 2020
# Pre-split: ~$500/share
# Post-split (adjusted): ~$125/share

# If you accidentally use unadjusted:
# - Backtests will show phantom gains/losses at split dates
# - Returns calculations will be completely wrong
```

**Why adjusted close matters**:

```python
# Example: AAPL 4:1 split on 2020-08-31
# Unadjusted: Aug 30 close = $499, Aug 31 close = $129
#   → Appears like -74% loss! (Wrong)
# Adjusted: Aug 30 close = $124.75, Aug 31 close = $129
#   → Actual +3.4% gain (Correct)
```

### 4. Validate Data Quality

```python
from ml4t.data.validation import validate_ohlcv

df = provider.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-12-31")

# Validate OHLCV invariants
validate_ohlcv(df)  # Raises exception if invalid

# Check for:
# - high >= low ✓
# - high >= open ✓
# - high >= close ✓
# - low <= open ✓
# - low <= close ✓
# - volume >= 0 ✓
```

**See**: [Tutorial 04: Data Quality](../tutorials/04_data_quality.md)

### 5. Multi-Provider Fallback

```python
from ml4t.data.providers import TiingoProvider, IEXCloudProvider, AlphaVantageProvider

def fetch_stock_data_robust(symbol: str, start: str, end: str):
    """Fetch stock data with fallback providers."""

    providers = [
        ("tiingo", TiingoProvider),
        ("iex_cloud", IEXCloudProvider),
        ("alpha_vantage", AlphaVantageProvider),
    ]

    for name, ProviderClass in providers:
        try:
            provider = ProviderClass()
            return provider.fetch_ohlcv(symbol=symbol, start=start, end=end)
        except Exception as e:
            print(f"{name} failed: {e}")

    raise RuntimeError("All providers failed")

# Usage
df = fetch_stock_data_robust("AAPL", "2024-01-01", "2024-12-31")
```

**See**: [Tutorial 05: Multi-Provider Strategies](../tutorials/05_multi_provider.md)

---

## Common Gotchas

### 1. Symbol Format Varies by Provider

```python
# ❌ Wrong provider/symbol combo
eodhd = EODHDProvider()
eodhd.fetch_ohlcv(symbol="AAPL", ...)  # Error! Needs AAPL.US

tiingo = TiingoProvider()
tiingo.fetch_ohlcv(symbol="AAPL.US", ...)  # Error! Needs AAPL

# ✅ Correct formats
eodhd.fetch_ohlcv(symbol="AAPL.US", ...)    # SYMBOL.EXCHANGE
tiingo.fetch_ohlcv(symbol="AAPL", ...)       # SYMBOL only
iex.fetch_ohlcv(symbol="AAPL", ...)          # SYMBOL only
```

### 2. Free Tier Limitations

```python
# Alpha Vantage: Only 25 calls/day
# ❌ This will fail on day 1:
for symbol in sp500_symbols:  # 500 symbols
    df = provider.fetch_ohlcv(symbol, ...)
    # Hits limit at symbol #25

# ✅ Spread over 20 days or upgrade to paid tier
```

### 3. Market Hours Matter

```python
# Fetching "today's" data before market close
from datetime import datetime

now = datetime.now()  # 10:00 AM ET (market just opened)

# ❌ Won't have today's complete bar yet
df = provider.fetch_ohlcv("AAPL", start="2024-11-04", end="2024-11-04")
# Returns empty or yesterday's data

# ✅ Fetch yesterday's complete data
yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
df = provider.fetch_ohlcv("AAPL", start=yesterday, end=yesterday)
```

### 4. Corporate Actions

```python
# Stock splits can surprise you
df = provider.fetch_ohlcv("NVDA", start="2021-06-01", end="2021-08-01")

# NVDA had 4:1 split on July 20, 2021
# - Pre-split: ~$750/share
# - Post-split (adjusted): ~$187/share

# Always use adjusted close to avoid phantom losses
```

---

## Provider Selection Decision Tree

```
START: What markets do you need?

├─ US stocks only?
│  ├─ High volume (100+ symbols, daily updates)?
│  │  → Tiingo (1000/day free) ✅
│  │
│  ├─ Need fundamentals too?
│  │  → IEX Cloud (50K messages/month) ✅
│  │
│  └─ Low volume research (<10 symbols)?
│     → Alpha Vantage (25/day free) ✅
│
├─ Global stocks (60+ exchanges)?
│  ├─ Budget-conscious?
│  │  → EODHD (€19.99/month unlimited) ✅ BEST VALUE
│  │
│  └─ Professional real-time needs?
│     → Finnhub ($59.99/month) or Polygon ($199/month)
│
└─ Multi-asset (stocks + forex + crypto)?
   → Twelve Data (800/day free) ✅
```

---

## Related Documentation

**Getting Started**:
- [Quick Start Guide](../README.md#quick-start)
- [Tutorial 01: Understanding OHLCV Data](../tutorials/01_understanding_ohlcv.md)

**Provider Setup**:
- [Provider Selection Guide](../provider-selection-guide.md)
- [Creating a Custom Provider](../creating_a_provider.md)

**Best Practices**:
- [Tutorial 02: Rate Limiting Best Practices](../tutorials/02_rate_limiting.md)
- [Tutorial 03: Incremental Updates](../tutorials/03_incremental_updates.md)
- [Tutorial 04: Data Quality Validation](../tutorials/04_data_quality.md)
- [Tutorial 05: Multi-Provider Strategies](../tutorials/05_multi_provider.md)

**Other Asset Classes**:
- [Cryptocurrency Data Guide](./crypto.md) - Bitcoin, Ethereum, altcoins
- [Forex Data Guide](./forex.md) - Foreign exchange pairs
- [Futures Data Guide](./futures.md) - Futures and options

---

## Next Steps

1. **Get API keys** for your chosen provider(s)
2. **Try the Quick Start examples** above
3. **Implement incremental updates** (Tutorial 03) to minimize API calls
4. **Set up data quality validation** (Tutorial 04) in your pipeline
5. **Build multi-provider fallback** (Tutorial 05) for production resilience

**Questions or issues?** See [CONTRIBUTING.md](../../CONTRIBUTING.md) or open a GitHub issue.
