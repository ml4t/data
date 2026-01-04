# Cryptocurrency Data Guide

**Asset Class**: Cryptocurrencies (Bitcoin, Ethereum, altcoins)
**Available Providers**: CoinGecko, CryptoCompare
**Difficulty**: Beginner-friendly
**Recommended For**: Crypto trading strategies, portfolio analysis, academic research

---

## Overview

ML4T Data provides two excellent **free-tier providers** for cryptocurrency data, both supporting 10,000+ digital assets with no API key requirements for basic usage. Cryptocurrency data is generally easier to work with than traditional financial data due to:

- **24/7 Markets**: No market hours, weekends, or holidays
- **Global Access**: No regional restrictions like US-only or exchange-specific data
- **Generous Free Tiers**: Both providers offer substantial free quotas
- **Simple Symbol Format**: BTC, ETH, etc. (though CoinGecko uses internal IDs)

**Quick Comparison**:

| Provider | Free Tier | API Key | Best For | Symbol Format |
|----------|-----------|---------|----------|---------------|
| **CoinGecko** | 30 calls/min (Demo) | Required | Historical analysis, wide coverage | Coin IDs (bitcoin, ethereum) |
| **CryptoCompare** | 100K calls/month | Optional | Aggregated prices, exchange data | Pairs (BTC/USD, ETH/BTC) |

---

## CoinGecko (Recommended for Beginners)

### Why CoinGecko?

- ✅ **Easy setup** - Free Demo plan with API key (10K calls/month)
- ✅ **10,000+ cryptocurrencies** with market data
- ✅ **Reasonable rate limits** - 30 calls/minute is sufficient for most use cases
- ✅ **Aggregated data** across 500+ exchanges
- ✅ **Market metrics** - Volume, market cap, circulating supply

**Best for**: Historical analysis, portfolio tracking, academic research

### Quick Start

```python
from ml4t.data.providers import CoinGeckoProvider

# Requires API key for Demo plan (free)
# Get your key from: https://www.coingecko.com/en/api/pricing
provider = CoinGeckoProvider()  # Reads COINGECKO_API_KEY from environment

# Fetch Bitcoin daily data (last 30 days)
df = provider.fetch_ohlcv(
    symbol="BTC",  # Uses symbol_to_id mapping internally
    start="2024-01-01",
    end="2024-01-31",
    frequency="daily"
)

print(df.head())
```

**Output**:
```
shape: (31, 7)
┌─────────────────────┬──────────┬───────────┬───────────┬───────────┬──────────────┬──────────┐
│ timestamp           │ symbol   │ open      │ high      │ low       │ close        │ volume   │
│ ---                 │ ---      │ ---       │ ---       │ ---       │ ---          │ ---      │
│ datetime[μs]        │ str      │ f64       │ f64       │ f64       │ f64          │ f64      │
╞═════════════════════╪══════════╪═══════════╪═══════════╪═══════════╪══════════════╪══════════╡
│ 2024-01-01 00:00:00 │ BTC      │ 42265.18  │ 44923.45  │ 41824.36  │ 44167.21     │ 1.85e9   │
│ 2024-01-02 00:00:00 │ BTC      │ 44167.21  │ 45870.34  │ 43956.12  │ 45456.87     │ 2.12e9   │
└─────────────────────┴──────────┴───────────┴───────────┴───────────┴──────────────┴──────────┘
```

### Symbol Mapping

CoinGecko uses internal **coin IDs** (e.g., "bitcoin", "ethereum") rather than ticker symbols. ML4T Data handles this automatically for major coins:

```python
# These all work - ML4T Data maps symbols to IDs internally
provider.fetch_ohlcv(symbol="BTC", ...)   # → bitcoin
provider.fetch_ohlcv(symbol="ETH", ...)   # → ethereum
provider.fetch_ohlcv(symbol="DOGE", ...)  # → dogecoin

# Or use coin IDs directly
provider.fetch_ohlcv(symbol="bitcoin", ...)
```

**Pre-mapped symbols** (20+ popular coins):
- BTC → bitcoin
- ETH → ethereum
- USDT → tether
- BNB → binancecoin
- USDC → usd-coin
- XRP → ripple
- ADA → cardano
- DOGE → dogecoin
- SOL → solana
- And more...

**Finding coin IDs** for other cryptocurrencies:
```python
# Search CoinGecko's coin list (programmatically)
# Visit: https://www.coingecko.com/en/coins/
# Or use their API: https://api.coingecko.com/api/v3/coins/list
```

### Supported Frequencies

CoinGecko free tier supports **daily OHLCV** data only:

```python
# ✅ Supported
df = provider.fetch_ohlcv(symbol="BTC", frequency="daily")

# ❌ Not supported on free tier (requires CoinGecko Pro)
df = provider.fetch_ohlcv(symbol="BTC", frequency="hourly")  # Error!
```

### Rate Limits

**Demo Plan (Free)**: 30 calls per minute, 10,000 calls/month
**Public API (No Key)**: 5-15 calls per minute (unstable, not recommended)

The Demo plan is **sufficient** for most use cases:

```python
# Example: Fetch 10 coins, 90 days of data each
# = 10 API calls = well within rate limit

symbols = ["BTC", "ETH", "SOL", "ADA", "DOGE",
           "MATIC", "LINK", "UNI", "AVAX", "DOT"]

for symbol in symbols:
    df = provider.fetch_ohlcv(
        symbol=symbol,
        start="2024-01-01",
        end="2024-03-31",
        frequency="daily"
    )
    # Provider automatically rate-limits (no sleep needed)
```

**Pro Tier** (with API key): 10 calls/second

```python
provider = CoinGeckoProvider(
    api_key="your_coingecko_api_key",  # From environment or parameter
    use_pro=True  # Enable pro endpoints
)
```

### Complete Example: Multi-Coin Analysis

```python
from ml4t.data.providers import CoinGeckoProvider
from datetime import datetime, timedelta
import polars as pl

# Initialize provider (no API key needed)
provider = CoinGeckoProvider()

# Define portfolio
portfolio = ["BTC", "ETH", "SOL", "AVAX", "MATIC"]

# Fetch 90 days of data for each
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

all_data = []
for symbol in portfolio:
    df = provider.fetch_ohlcv(
        symbol=symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        frequency="daily"
    )
    all_data.append(df)

# Combine into single DataFrame
combined = pl.concat(all_data)

# Calculate daily returns
returns = combined.group_by("symbol").agg(
    pl.col("close").pct_change().alias("daily_return")
)

print(returns)
```

### Error Handling

Common errors and solutions:

```python
from ml4t.data.core.exceptions import SymbolNotFoundError, RateLimitError

try:
    df = provider.fetch_ohlcv(symbol="INVALID", start="2024-01-01", end="2024-01-31")
except SymbolNotFoundError as e:
    print(f"Coin not found: {e}")
    # Solution: Check symbol spelling or use coin ID directly

try:
    # Too many rapid calls
    for i in range(100):
        df = provider.fetch_ohlcv(symbol="BTC", ...)
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # ML4T Data automatically retries with backoff
```

### When to Use CoinGecko

✅ **Good for**:
- Beginners (zero setup)
- Historical backtesting (daily data)
- Portfolio analysis
- Academic research
- Multi-coin comparisons

❌ **Not ideal for**:
- Intraday trading strategies (no hourly/minute data on free tier)
- Exchange-specific data (aggregated only)
- Ultra-high-frequency analysis

---

## CryptoCompare (Advanced Features)

### Why CryptoCompare?

- ✅ **Aggregated prices** across multiple exchanges (CCCAGG index)
- ✅ **Exchange-specific data** - Choose Binance, Coinbase, Kraken, etc.
- ✅ **Multiple frequencies** - Minute, hourly, daily (free tier)
- ✅ **100,000 calls/month** free tier
- ✅ **5,000+ cryptocurrencies**

**Best for**: Aggregated price analysis, exchange comparisons, intraday strategies

### Quick Start

```python
from ml4t.data.providers import CryptoCompareProvider

# Optional: Set API key for higher rate limits
provider = CryptoCompareProvider(
    api_key="your_cryptocompare_api_key",  # Or from env: CRYPTOCOMPARE_API_KEY
    exchange="CCCAGG"  # Aggregate across exchanges (default)
)

# Fetch Bitcoin daily data
df = provider.fetch_ohlcv(
    symbol="BTC/USD",  # Note: Uses pair format
    start="2024-01-01",
    end="2024-01-31",
    frequency="daily"
)

print(df.head())
```

### Symbol Format

CryptoCompare uses **trading pairs** with slash notation:

```python
# ✅ Correct format
"BTC/USD"   # Bitcoin vs US Dollar
"ETH/BTC"   # Ethereum vs Bitcoin
"SOL/USDT"  # Solana vs Tether

# ❌ Incorrect
"BTC"       # Missing quote currency
"BTCUSD"    # Missing slash separator
```

**Common pairs**:
```python
# Fiat pairs
"BTC/USD", "ETH/USD", "SOL/USD"

# Stablecoin pairs
"BTC/USDT", "ETH/USDC", "BNB/BUSD"

# Crypto pairs
"ETH/BTC", "SOL/ETH", "ADA/BTC"
```

### Supported Frequencies

CryptoCompare free tier supports **minute, hourly, and daily** data:

```python
# Daily (up to 2000 days per call)
df = provider.fetch_ohlcv(symbol="BTC/USD", frequency="daily")

# Hourly (up to 2000 hours per call)
df = provider.fetch_ohlcv(symbol="BTC/USD", frequency="hourly")

# Minute (up to 2000 minutes per call)
df = provider.fetch_ohlcv(symbol="BTC/USD", frequency="1minute")
```

**Frequency limits**:
- `daily`: 2000 bars per API call (~5.5 years)
- `hourly`: 2000 bars per API call (~83 days)
- `1minute`: 2000 bars per API call (~33 hours)

### Exchange Selection

Fetch data from specific exchanges or use aggregated data:

```python
# Aggregated (default) - CryptoCompare's weighted index
provider_agg = CryptoCompareProvider(exchange="CCCAGG")

# Binance-specific
provider_binance = CryptoCompareProvider(exchange="Binance")

# Coinbase-specific
provider_coinbase = CryptoCompareProvider(exchange="Coinbase")

# Kraken-specific
provider_kraken = CryptoCompareProvider(exchange="Kraken")
```

**Popular exchanges supported**:
- Binance, Coinbase, Kraken, Gemini, Bitstamp
- Bitfinex, Huobi, OKEx, FTX (historical), and 100+ more

### Rate Limits

**Free Tier**: 100,000 calls per month (10 calls/minute recommended)

```python
# ML4T Data automatically rate-limits to ~10 calls/minute
# This stays well within monthly quota:
# 10 calls/min × 60 min × 24 hr × 30 days = 432,000 theoretical max
# But capped at 100K/month by API

# Example: Safe usage pattern
import time
from datetime import datetime, timedelta

symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

for symbol in symbols:
    df = provider.fetch_ohlcv(
        symbol=symbol,
        start="2024-01-01",
        end="2024-12-31",
        frequency="daily"
    )
    # No manual sleep needed - ML4T Data handles rate limiting
```

**With API Key**: Higher rate limits (varies by plan)

### Complete Example: Exchange Comparison

```python
from ml4t.data.providers import CryptoCompareProvider
import polars as pl

# Compare Bitcoin price across exchanges
exchanges = ["CCCAGG", "Binance", "Coinbase", "Kraken"]

all_data = []
for exchange in exchanges:
    provider = CryptoCompareProvider(exchange=exchange)

    df = provider.fetch_ohlcv(
        symbol="BTC/USD",
        start="2024-01-01",
        end="2024-01-31",
        frequency="daily"
    )

    # Add exchange column
    df = df.with_columns(pl.lit(exchange).alias("exchange"))
    all_data.append(df)

    provider.close()

# Combine and analyze
combined = pl.concat(all_data)

# Calculate price spread across exchanges
spread = combined.group_by("timestamp").agg([
    pl.col("close").max().alias("max_price"),
    pl.col("close").min().alias("min_price"),
    (pl.col("close").max() - pl.col("close").min()).alias("spread"),
    ((pl.col("close").max() - pl.col("close").min()) / pl.col("close").mean() * 100).alias("spread_pct")
])

print(spread.sort("timestamp"))
# Identify arbitrage opportunities!
```

### Error Handling

```python
from ml4t.data.core.exceptions import DataNotAvailableError, RateLimitError

try:
    df = provider.fetch_ohlcv(
        symbol="OBSCURECOIN/USD",
        start="2024-01-01",
        end="2024-01-31"
    )
except DataNotAvailableError as e:
    print(f"Pair not available: {e}")
    # Try different exchange or check coin listing

try:
    # Exceeding rate limits
    for i in range(20):
        df = provider.fetch_ohlcv(symbol="BTC/USD", ...)
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # Wait or reduce call frequency
```

### When to Use CryptoCompare

✅ **Good for**:
- Intraday strategies (minute/hourly data)
- Exchange-specific analysis
- Price aggregation studies
- Arbitrage detection
- Professional crypto research

❌ **Not ideal for**:
- Absolute beginners (requires API key understanding)
- Projects needing >100K calls/month without paid tier
- Coins not listed on major exchanges

---

## Provider Comparison

### Feature Comparison

| Feature | CoinGecko | CryptoCompare |
|---------|-----------|---------------|
| **API Key Required** | Yes (Demo plan) | No (optional for higher limits) |
| **Free Tier Limit** | 30 calls/min (10K/month) | 100K calls/month |
| **Cryptocurrencies** | 10,000+ | 5,000+ |
| **Daily Data** | ✅ Yes | ✅ Yes |
| **Hourly Data** | ❌ Pro only | ✅ Yes |
| **Minute Data** | ❌ Pro only | ✅ Yes |
| **Aggregated Prices** | ✅ Yes | ✅ Yes (CCCAGG) |
| **Exchange-Specific** | ❌ No | ✅ Yes |
| **Symbol Format** | Coin IDs | Trading pairs |
| **Market Metrics** | ✅ Yes (market cap, volume) | ⚠️ Limited |
| **Setup Difficulty** | 🟢 Easy | 🟡 Moderate |

### Performance Comparison

```python
# Benchmark: Fetch 1 year of daily BTC data

# CoinGecko
provider = CoinGeckoProvider()
df = provider.fetch_ohlcv(symbol="BTC", start="2023-01-01", end="2023-12-31")
# → 1 API call, ~2 seconds

# CryptoCompare
provider = CryptoCompareProvider()
df = provider.fetch_ohlcv(symbol="BTC/USD", start="2023-01-01", end="2023-12-31")
# → 1 API call, ~2 seconds

# Both providers are equally fast for daily data
```

### Cost Comparison

**Free Tier Value**:

| Provider | Free Limit | Est. Monthly Value | Paid Tier |
|----------|------------|--------------------|-----------|
| **CoinGecko** | 30 calls/min (10K/month) | $0 (requires signup) | ~$129/month (Pro) |
| **CryptoCompare** | 100K calls/month | $0 (very generous) | $49-$999/month |

**Which is more generous?**

```python
# Scenario 1: Daily batch job (fetch 100 coins daily)
# CoinGecko: 100 calls/day × 30 days = 3,000 calls/month ✅
# CryptoCompare: 100 calls/day × 30 days = 3,000 calls/month ✅
# Winner: Tie

# Scenario 2: Intraday strategy (hourly updates for 10 coins)
# CoinGecko: Not available on free tier ❌
# CryptoCompare: 10 calls/hour × 24 hrs × 30 days = 7,200 calls/month ✅
# Winner: CryptoCompare

# Scenario 3: High-frequency batch (1000 coins, daily)
# CoinGecko: 1000 calls/day = 34 minutes @30/min (within monthly 10K cap for ~10 days) ⚠️
# CryptoCompare: 1000 calls/day × 30 = 30,000 calls/month ✅
# Winner: CryptoCompare (for sustained high volume)
```

---

## Best Practices

### 1. Use Incremental Updates

Don't re-fetch entire history every time:

```python
from ml4t.data.provider_updater import ProviderUpdater

# First run: Fetch all history
updater = ProviderUpdater(provider="coingecko")
df = updater.update(
    symbol="BTC",
    frequency="daily",
    lookback_days=365  # Initial: 1 year of data
)

# Subsequent runs: Only fetch new data
df = updater.update(
    symbol="BTC",
    frequency="daily",
    lookback_days=7  # Only fetch last 7 days, merge with existing
)
# Automatically detects gaps and fills them!
```

**See**: [Tutorial 03: Incremental Updates](../tutorials/03_incremental_updates.md)

### 2. Validate Data Quality

Always validate OHLCV data after fetching:

```python
from ml4t.data.validation import validate_ohlcv

df = provider.fetch_ohlcv(symbol="BTC", ...)

# Validate invariants (high >= low, etc.)
validate_ohlcv(df)  # Raises exception if data is corrupt

# Check for gaps
gaps = df.select("timestamp").diff().filter(pl.col("timestamp") > timedelta(days=2))
if not gaps.is_empty():
    print(f"Warning: {len(gaps)} gaps found in data")
```

**See**: [Tutorial 04: Data Quality](../tutorials/04_data_quality.md)

### 3. Handle Rate Limits Gracefully

ML4T Data automatically rate-limits, but be aware:

```python
# ✅ Good: Sequential fetching (automatic rate limiting)
for symbol in ["BTC", "ETH", "SOL"]:
    df = provider.fetch_ohlcv(symbol=symbol, ...)

# ❌ Bad: Parallel fetching without coordination
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Will exceed rate limits!
    futures = [executor.submit(provider.fetch_ohlcv, symbol=s) for s in symbols]
```

**See**: [Tutorial 02: Rate Limiting](../tutorials/02_rate_limiting.md)

### 4. Use Multi-Provider Fallback

Never depend on a single provider:

```python
from ml4t.data.providers import CoinGeckoProvider, CryptoCompareProvider

def fetch_crypto_data_robust(symbol: str, start: str, end: str):
    """Fetch crypto data with fallback providers."""

    # Try CoinGecko first (no API key needed)
    try:
        provider = CoinGeckoProvider()
        return provider.fetch_ohlcv(symbol=symbol, start=start, end=end)
    except Exception as e:
        print(f"CoinGecko failed: {e}")

    # Fallback to CryptoCompare
    try:
        provider = CryptoCompareProvider()
        # Convert symbol format: BTC → BTC/USD
        pair = f"{symbol}/USD"
        return provider.fetch_ohlcv(symbol=pair, start=start, end=end)
    except Exception as e:
        print(f"CryptoCompare failed: {e}")
        raise RuntimeError("All providers failed")

# Usage
df = fetch_crypto_data_robust("BTC", "2024-01-01", "2024-01-31")
```

**See**: [Tutorial 05: Multi-Provider Strategies](../tutorials/05_multi_provider.md)

---

## Common Gotchas

### 1. Symbol Format Differences

```python
# ❌ Wrong: Using wrong format for provider
coingecko = CoinGeckoProvider()
coingecko.fetch_ohlcv(symbol="BTC/USD", ...)  # Error! Needs "BTC" or "bitcoin"

cryptocompare = CryptoCompareProvider()
cryptocompare.fetch_ohlcv(symbol="BTC", ...)  # Error! Needs "BTC/USD"

# ✅ Correct: Use provider-specific format
coingecko.fetch_ohlcv(symbol="BTC", ...)       # Symbol or coin ID
cryptocompare.fetch_ohlcv(symbol="BTC/USD", ...)  # Trading pair
```

### 2. Timezone Assumptions

All timestamps are UTC, but crypto markets are 24/7:

```python
# Crypto data doesn't have "market close" like stocks
df = provider.fetch_ohlcv(symbol="BTC", start="2024-01-01", end="2024-01-02")

# Check timezone
assert df["timestamp"][0].dt.hour() == 0  # Midnight UTC
```

### 3. Rate Limit Planning

```python
# ❌ Bad: Not accounting for rate limits
for i in range(1000):  # Will take 34+ minutes on CoinGecko Demo plan
    df = provider.fetch_ohlcv(symbol=f"COIN_{i}", ...)

# ✅ Good: Batch wisely or upgrade to paid tier
# CoinGecko Demo: 30 calls/min = 1000 calls in 34 minutes (hits monthly cap fast!)
# CryptoCompare free: 100K/month = plenty for most use cases
```

### 4. Aggregated vs Exchange Data

```python
# CoinGecko: Always aggregated across exchanges
coingecko_df = coingecko.fetch_ohlcv(symbol="BTC", ...)
# Price is weighted average across 500+ exchanges

# CryptoCompare: Choose aggregated or exchange-specific
agg_provider = CryptoCompareProvider(exchange="CCCAGG")  # Aggregated
binance_provider = CryptoCompareProvider(exchange="Binance")  # Binance only

# Prices may differ significantly!
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
- [Equities Data Guide](./equities.md) - US and global stocks
- [Forex Data Guide](./forex.md) - Foreign exchange pairs
- [Futures Data Guide](./futures.md) - Futures and options

---

## Next Steps

1. **Try the Quick Start examples** above with CoinGecko (no API key needed!)
2. **Read Tutorial 01** to understand OHLCV data fundamentals
3. **Implement incremental updates** (Tutorial 03) for production systems
4. **Set up data quality validation** (Tutorial 04) to catch corrupted data
5. **Build multi-provider fallback** (Tutorial 05) for resilience

**Questions or issues?** See [CONTRIBUTING.md](../../CONTRIBUTING.md) or open a GitHub issue.
