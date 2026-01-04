# Foreign Exchange (Forex) Data Guide

**Asset Class**: Currency Pairs (FX)
**Available Providers**: OANDA (primary), Twelve Data, Alpha Vantage, Polygon
**Difficulty**: Moderate (requires understanding of forex mechanics)
**Recommended For**: FX trading strategies, currency risk analysis, global macro research

---

## Overview

ML4T Data provides access to **professional-grade forex data** through multiple providers, with OANDA being the gold standard for FX market data. The forex market is the world's largest financial market with unique characteristics:

- **24/5 Markets**: Sunday 5 PM ET - Friday 5 PM ET (closed weekends)
- **High Liquidity**: $7.5 trillion daily volume
- **Global Trading**: Follows the sun (Tokyo → London → New York sessions)
- **Leverage & Spreads**: Tight bid-ask spreads, high leverage available
- **No Central Exchange**: OTC market, prices vary by broker/bank

**Quick Provider Comparison**:

| Provider | Free Tier | API Key | Pairs | Timeframes | Best For |
|----------|-----------|---------|-------|------------|----------|
| **OANDA** | Demo account (practice) | Required | 70+ | Second to monthly | Professional FX trading |
| **Twelve Data** | 800/day | Required | 50+ major pairs | Minute to monthly | Multi-asset portfolios |
| **Alpha Vantage** | 25/day | Required | Major pairs | Daily+ | Research (low volume) |
| **Polygon** | 5/min | Required | 30+ pairs | Minute to daily | Real-time FX + stocks |

---

## OANDA (Recommended for Forex)

### Why OANDA?

- ✅ **Professional-grade FX data** - Used by institutional traders
- ✅ **120 requests/second** - Highest rate limit in class
- ✅ **70+ currency pairs** - Majors, minors, exotics
- ✅ **19 timeframes** - From 5 seconds to 1 month
- ✅ **Practice account** - Free demo environment for testing
- ✅ **Spread data included** - Bid/ask spreads for realistic simulation

**Best for**: Forex trading strategies, high-frequency FX analysis, professional backtesting

### Quick Start

```python
from ml4t.data.providers import OandaProvider

# Get API key from: https://www.oanda.com/us-en/trading/api/
# Create a practice (demo) account first
provider = OandaProvider(
    api_key="your_oanda_api_key",  # Or set OANDA_API_KEY env var
    account_type="practice"  # Use 'live' for real trading account
)

# Fetch EUR/USD hourly data
df = provider.fetch_ohlcv(
    symbol="EUR_USD",  # Note: Underscore separator
    start="2024-01-01",
    end="2024-12-31",
    frequency="H1"  # 1-hour bars
)

print(df.head())
```

**Output**:
```
shape: (8760, 7)
┌─────────────────────┬──────────┬─────────┬─────────┬─────────┬─────────┬────────┐
│ timestamp           │ symbol   │ open    │ high    │ low     │ close   │ volume │
│ ---                 │ ---      │ ---     │ ---     │ ---     │ ---     │ ---    │
│ datetime[μs]        │ str      │ f64     │ f64     │ f64     │ f64     │ f64    │
╞═════════════════════╪══════════╪═════════╪═════════╪═════════╪═════════╪════════╡
│ 2024-01-01 00:00:00 │ EUR_USD  │ 1.1045  │ 1.1052  │ 1.1038  │ 1.1048  │ 125.0  │
│ 2024-01-01 01:00:00 │ EUR_USD  │ 1.1048  │ 1.1055  │ 1.1042  │ 1.1050  │ 198.0  │
└─────────────────────┴──────────┴─────────┴─────────┴─────────┴─────────┴────────┘
```

### Currency Pair Symbols

OANDA uses **underscore notation** for currency pairs:

```python
# Major Pairs (USD-based)
"EUR_USD"  # Euro / US Dollar
"GBP_USD"  # British Pound / US Dollar
"USD_JPY"  # US Dollar / Japanese Yen
"USD_CHF"  # US Dollar / Swiss Franc
"AUD_USD"  # Australian Dollar / US Dollar
"USD_CAD"  # US Dollar / Canadian Dollar
"NZD_USD"  # New Zealand Dollar / US Dollar

# Cross Pairs (No USD)
"EUR_GBP"  # Euro / British Pound
"EUR_JPY"  # Euro / Japanese Yen
"GBP_JPY"  # British Pound / Japanese Yen
"AUD_JPY"  # Australian Dollar / Japanese Yen

# Exotic Pairs
"USD_TRY"  # US Dollar / Turkish Lira
"USD_MXN"  # US Dollar / Mexican Peso
"USD_ZAR"  # US Dollar / South African Rand
"EUR_TRY"  # Euro / Turkish Lira
```

**Available Pairs** (70+ total):
- **Majors**: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
- **Minors**: EUR/GBP, EUR/JPY, GBP/JPY, AUD/JPY, EUR/CHF, GBP/CHF, etc.
- **Exotics**: USD/TRY, USD/MXN, USD/ZAR, EUR/TRY, USD/SGD, USD/THB, etc.

### Timeframes (Granularities)

OANDA offers **19 different timeframes** - more than any other provider:

```python
# Second-level (ultra-high-frequency)
"S5"   # 5 seconds
"S10"  # 10 seconds
"S15"  # 15 seconds
"S30"  # 30 seconds

# Minute-level (high-frequency)
"M1"   # 1 minute
"M2"   # 2 minutes
"M4"   # 4 minutes
"M5"   # 5 minutes
"M10"  # 10 minutes
"M15"  # 15 minutes
"M30"  # 30 minutes

# Hour-level (intraday)
"H1"   # 1 hour
"H2"   # 2 hours
"H3"   # 3 hours
"H4"   # 4 hours
"H6"   # 6 hours
"H8"   # 8 hours
"H12"  # 12 hours

# Day/Week/Month (swing/position)
"D"    # 1 day
"W"    # 1 week
"M"    # 1 month
```

**Example: Multi-timeframe Analysis**

```python
provider = OandaProvider(account_type="practice")

# Daily for trend
daily = provider.fetch_ohlcv("EUR_USD", start="2024-01-01", end="2024-12-31", frequency="D")

# 4-hour for entries
h4 = provider.fetch_ohlcv("EUR_USD", start="2024-11-01", end="2024-11-30", frequency="H4")

# 15-minute for timing
m15 = provider.fetch_ohlcv("EUR_USD", start="2024-11-04", end="2024-11-05", frequency="M15")
```

### Rate Limits

OANDA has the **most generous rate limits** among forex providers:

**Rate Limit**: 120 requests per second (7,200/minute, 432,000/hour)

```python
# Example: Fetch 50 pairs, 1 year hourly data
# 50 pairs × 1 API call each = 50 requests
# Completes in < 1 second (well within 120/sec limit)

pairs = ["EUR_USD", "GBP_USD", "USD_JPY", ...]  # 50 pairs

import time
start = time.time()

for pair in pairs:
    df = provider.fetch_ohlcv(
        symbol=pair,
        start="2024-01-01",
        end="2024-12-31",
        frequency="H1"
    )
    # No manual rate limiting needed!

elapsed = time.time() - start
print(f"Fetched 50 pairs in {elapsed:.2f} seconds")
# Output: Fetched 50 pairs in 1.45 seconds
```

**Comparison**:
| Provider | Rate Limit | Notes |
|----------|------------|-------|
| OANDA | 120/second | ⭐ Best for high-frequency |
| Twelve Data | ~13/second (800/minute) | Good |
| Alpha Vantage | 5/minute | Very restrictive |
| Polygon | 5/minute (free) | Restrictive |

### Practice vs Live Accounts

OANDA offers two account types:

**Practice Account** (Demo):
- ✅ **Free** - No cost, no risk
- ✅ **Full API access** - Same data as live
- ✅ **Unlimited testing** - Perfect for backtesting
- ✅ **Real market prices** - Live data feed
- ⚠️ **Virtual money** - Can't execute real trades

**Live Account** (Real Trading):
- ✅ **Real trading** - Execute actual FX trades
- ✅ **Same API** - Identical to practice
- ⚠️ **Real money at risk** - Trade carefully
- ⚠️ **Regulatory requirements** - KYC, minimum deposit

**Setup**:
```python
# Practice account (for backtesting)
practice_provider = OandaProvider(
    api_key="practice_api_key",
    account_type="practice"  # Demo environment
)

# Live account (for paper trading or real execution)
live_provider = OandaProvider(
    api_key="live_api_key",
    account_type="live"  # Production environment
)
```

### Spread Data (Bid/Ask)

OANDA includes **bid/ask spread data** for realistic backtesting:

```python
provider = OandaProvider(
    api_key="your_key",
    include_spread=True  # Default: True
)

df = provider.fetch_ohlcv("EUR_USD", start="2024-11-04", end="2024-11-04", frequency="M5")

# DataFrame includes spread information in volume column
# (Note: Full spread data available in raw API response)
print(df.select(["timestamp", "close", "volume"]))
```

**Why spreads matter**:
```python
# Example: EUR/USD spread is typically 0.1 pips (0.00001)
# If you buy at 1.10500 (ask) and immediately sell at 1.10499 (bid)
# You lose 1 pip = $10 on a standard lot (100,000 EUR)

# Without spread modeling:
# Backtest shows profit on quick scalps ❌ (unrealistic)

# With spread modeling:
# Backtest deducts spread cost ✅ (realistic)
```

### Complete Example: Multi-Pair Strategy

```python
from ml4t.data.providers import OandaProvider
import polars as pl
from datetime import datetime, timedelta

# Initialize provider
provider = OandaProvider(account_type="practice")

# Define currency basket
pairs = [
    "EUR_USD",  # Euro
    "GBP_USD",  # Pound
    "USD_JPY",  # Yen
    "USD_CHF",  # Franc
    "AUD_USD",  # Aussie
    "USD_CAD",  # Loonie
    "NZD_USD",  # Kiwi
]

# Fetch 6 months of hourly data for all pairs
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

all_data = []
for pair in pairs:
    df = provider.fetch_ohlcv(
        symbol=pair,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        frequency="H1"
    )
    all_data.append(df)

# Combine all pairs
combined = pl.concat(all_data)

# Calculate hourly returns for each pair
returns = combined.group_by("symbol").agg([
    pl.col("close").pct_change().alias("hourly_return"),
    pl.col("close").pct_change().std().alias("volatility")
])

print(returns.sort("volatility", descending=True))
# Identify which pairs are most volatile (tradeable)
```

### Error Handling

```python
from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    RateLimitError
)

try:
    df = provider.fetch_ohlcv("INVALID_PAIR", start="2024-01-01", end="2024-12-31")
except DataNotAvailableError as e:
    print(f"Pair not available: {e}")
    # Check OANDA's supported instruments list

try:
    # Practice account trying to access live data
    provider = OandaProvider(
        api_key="invalid_key",
        account_type="live"
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Verify API key and account type match
```

### When to Use OANDA

✅ **Good for**:
- Professional FX trading strategies
- High-frequency forex analysis (5-second to 1-minute bars)
- Multi-timeframe backtesting
- Large pair universes (50+ pairs)
- Realistic spread modeling
- Intraday FX strategies

❌ **Not ideal for**:
- Stock/equity data (use Tiingo, IEX Cloud instead)
- Cryptocurrency data (use CoinGecko, CryptoCompare)
- Beginner exploratory projects (may prefer simpler providers)

---

## Alternative Forex Providers

### Twelve Data (Multi-Asset with Forex)

**Best for**: Portfolios combining FX + stocks + crypto

```python
from ml4t.data.providers import TwelveDataProvider

# Get API key from: https://twelvedata.com/
provider = TwelveDataProvider(api_key="your_key")

# Fetch forex data (uses slash notation)
df = provider.fetch_ohlcv(
    symbol="EUR/USD",  # Note: Slash separator
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)
```

**Features**:
- ✅ 50+ major forex pairs
- ✅ 800 API calls/day free tier
- ✅ Stocks, forex, crypto in one provider
- ✅ Daily to 1-minute intervals
- ⚠️ Less granular than OANDA (no 5-second bars)

**When to use**: Need FX + equities + crypto data from single source

---

### Alpha Vantage (Low-Volume Research)

**Best for**: Research projects with minimal FX data needs

```python
from ml4t.data.providers import AlphaVantageProvider

provider = AlphaVantageProvider(api_key="your_key")

# Fetch daily FX data
df = provider.fetch_ohlcv(
    symbol="EUR/USD",
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)
```

**Features**:
- ✅ Major currency pairs
- ✅ Daily, weekly, monthly intervals
- ⚠️ **Only 25 API calls/day** (very restrictive)
- ⚠️ No intraday data on free tier

**When to use**: Very low-volume FX research (5-10 pairs, monthly updates)

---

### Polygon (Real-Time FX + Stocks)

**Best for**: Real-time FX monitoring + stock trading

```python
from ml4t.data.providers import PolygonProvider

provider = PolygonProvider(api_key="your_key")

# Fetch FX data
df = provider.fetch_ohlcv(
    symbol="C:EURUSD",  # Note: 'C:' prefix for currencies
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)
```

**Features**:
- ✅ 30+ major currency pairs
- ✅ Real-time + historical FX data
- ✅ Stocks, options, crypto also available
- ⚠️ Free tier: 5 API calls/minute (restrictive)
- ⚠️ Paid tier: $199-$399/month

**When to use**: Need real-time FX + equities from single provider, budget >$199/month

---

## Forex Market Characteristics

### 24-Hour Market Sessions

The forex market follows the sun across global financial centers:

```
Sydney Open:   5:00 PM ET Sunday   → 2:00 AM ET Monday
Tokyo Open:    7:00 PM ET          → 4:00 AM ET
London Open:   3:00 AM ET          → 12:00 PM ET
New York Open: 8:00 AM ET          → 5:00 PM ET
Market Close:  5:00 PM ET Friday
```

**Trading activity by session**:

```python
from datetime import datetime

df = provider.fetch_ohlcv("EUR_USD", start="2024-11-04", end="2024-11-08", frequency="H1")

# Add session labels
def get_session(hour: int) -> str:
    """Determine trading session based on UTC hour."""
    if 22 <= hour or hour < 5:
        return "Sydney/Tokyo"
    elif 5 <= hour < 12:
        return "Tokyo/London"
    elif 12 <= hour < 17:
        return "London/NY"
    elif 17 <= hour < 22:
        return "NY"

df = df.with_columns(
    pl.col("timestamp").dt.hour().map_elements(get_session).alias("session")
)

# Calculate average volatility by session
session_vol = df.group_by("session").agg(
    ((pl.col("high") - pl.col("low")) / pl.col("open") * 100).mean().alias("avg_range_pct")
)

print(session_vol.sort("avg_range_pct", descending=True))
# London/NY overlap typically shows highest volatility
```

### Weekends & Holidays

Forex markets are **closed weekends** but have unique holiday behavior:

```python
# Forex markets close: Friday 5:00 PM ET
# Forex markets open: Sunday 5:00 PM ET

# Check for weekend gaps
df = provider.fetch_ohlcv("EUR_USD", start="2024-11-01", end="2024-11-30", frequency="D")

# Identify weekend gaps (no data Saturday/Sunday)
gaps = df.select("timestamp").diff().filter(
    pl.col("timestamp") > timedelta(days=2)  # More than 1 trading day
)

print(f"Found {len(gaps)} weekend gaps")
# Expected: ~8-9 weekends in November
```

**Important holidays**:
- Christmas/New Year: Reduced liquidity
- Easter (Good Friday): Market closed
- US Thanksgiving: Early close (Friday 1 PM ET)

Unlike stock markets, most national holidays don't close FX markets (since it's global/OTC).

### Currency Pair Conventions

**Base vs Quote Currency**:

```python
# EUR/USD = 1.1050
# Base: EUR (1 Euro)
# Quote: USD (1.1050 Dollars)
# Meaning: 1 Euro = 1.1050 US Dollars

# GBP/JPY = 185.50
# Base: GBP (1 British Pound)
# Quote: JPY (185.50 Japanese Yen)
# Meaning: 1 Pound = 185.50 Yen
```

**Direct vs Indirect Quotes**:

```python
# US Dollar as QUOTE currency (direct from US perspective)
"EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD"  # Majors

# US Dollar as BASE currency (indirect from US perspective)
"USD/JPY", "USD/CHF", "USD/CAD"  # Also majors, but flipped

# Cross pairs (no USD)
"EUR/GBP", "EUR/JPY", "GBP/JPY"  # Calculated from majors
```

### Pips and Pipettes

Forex price movements measured in **pips** (percentage in points):

```python
# EUR/USD = 1.10500
# 1 pip = 0.0001 (4th decimal place)
# 1 pipette = 0.00001 (5th decimal place)

# GBP/USD moves from 1.26500 to 1.26550
# Change: 50 pips = 5 pipettes

# USD/JPY = 149.50 (only 2 decimal places for JPY pairs)
# 1 pip = 0.01 (2nd decimal place)
```

**Calculating pip value**:

```python
def calculate_pip_value(pair: str, lot_size: int = 100000) -> float:
    """Calculate pip value in USD for standard lot."""
    if "JPY" in pair:
        return (0.01 / 149.50) * lot_size if pair.startswith("USD") else 0.01 * lot_size / 149.50
    else:
        return 0.0001 * lot_size

# EUR/USD standard lot (100,000 EUR)
pip_value = calculate_pip_value("EUR/USD", lot_size=100000)
print(f"1 pip = ${pip_value:.2f}")
# Output: 1 pip = $10.00

# If EUR/USD moves 50 pips, profit/loss = $500 on standard lot
```

---

## Best Practices

### 1. Account for Spreads

Don't backtest on close prices alone:

```python
# ❌ Bad: Ignores transaction costs
df = provider.fetch_ohlcv("EUR_USD", frequency="M5")
entry_price = df["close"][0]  # Assumes zero spread

# ✅ Good: Model realistic spreads
typical_spread_pips = 0.1  # EUR/USD typically 0.1-0.2 pips
spread_cost = typical_spread_pips * 0.0001  # Convert pips to price

# Buy at ask (close + spread/2)
entry_price = df["close"][0] + spread_cost / 2

# Sell at bid (close - spread/2)
exit_price = df["close"][-1] - spread_cost / 2

# Account for spread in backtest PnL
```

### 2. Understand Rollover (Swap)

Positions held overnight incur **rollover interest**:

```python
# Forex trades settle in 2 business days (T+2)
# If you hold EUR/USD long overnight:
# - You pay USD interest rate
# - You receive EUR interest rate
# - Net: Rollover = EUR rate - USD rate

# Example (simplified):
# EUR interest rate: 4.0% annually
# USD interest rate: 5.25% annually
# Daily rollover: (4.0% - 5.25%) / 365 = -0.00342% per day

# On 100,000 EUR position:
# Daily cost = 100,000 * 1.1050 * -0.00342% = -$3.78 per day
```

**Include rollover in backtests** for multi-day positions.

### 3. Use Incremental Updates

Forex data accumulates quickly at intraday frequencies:

```python
from ml4t.data.provider_updater import ProviderUpdater

updater = ProviderUpdater(provider="oanda")

# First run: Fetch 1 year of hourly data
df = updater.update(
    symbol="EUR_USD",
    frequency="H1",
    lookback_days=365
)
# 365 days × 24 hours = 8,760 bars

# Daily runs: Only fetch last 24 hours
df = updater.update(
    symbol="EUR_USD",
    frequency="H1",
    lookback_days=1
)
# Only 24 bars fetched, merged with existing data
# 365x fewer API calls on subsequent runs!
```

**See**: [Tutorial 03: Incremental Updates](../tutorials/03_incremental_updates.md)

### 4. Validate FX Data Quality

```python
from ml4t.data.validation import validate_ohlcv

df = provider.fetch_ohlcv("EUR_USD", start="2024-01-01", end="2024-12-31")

# Validate standard OHLCV invariants
validate_ohlcv(df)

# FX-specific checks
assert df.filter(pl.col("volume") == 0).is_empty(), "No zero-volume bars"
assert df.filter(pl.col("high") == pl.col("low")).is_empty(), "No single-price bars"

# Check for weekend gaps (expected)
gaps = df.select("timestamp").diff().filter(pl.col("timestamp") > timedelta(days=2))
print(f"Weekend gaps: {len(gaps)}")  # Should be ~52 for 1 year of daily data
```

**See**: [Tutorial 04: Data Quality](../tutorials/04_data_quality.md)

### 5. Multi-Pair Correlation Analysis

Forex pairs are highly correlated - use this in strategies:

```python
import polars as pl

# Fetch major pairs
pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD"]
all_data = []

for pair in pairs:
    df = provider.fetch_ohlcv(pair, start="2024-01-01", end="2024-12-31", frequency="D")
    df = df.select(["timestamp", "close"]).rename({"close": pair})
    all_data.append(df)

# Join on timestamp
combined = all_data[0]
for df in all_data[1:]:
    combined = combined.join(df, on="timestamp", how="outer")

# Calculate correlation matrix
corr_matrix = combined.select(pairs).corr()
print(corr_matrix)

# Expected correlations:
# EUR/USD ↔ GBP/USD: Positive (both vs USD)
# EUR/USD ↔ USD/JPY: Negative (USD is opposite side)
# EUR/USD ↔ USD/CHF: Negative (CHF is safe-haven like JPY)
```

---

## Common Gotchas

### 1. Symbol Format Differences

```python
# ❌ Wrong: Using incorrect format for provider
oanda = OandaProvider()
oanda.fetch_ohlcv(symbol="EUR/USD", ...)  # Error! Needs EUR_USD

twelve_data = TwelveDataProvider()
twelve_data.fetch_ohlcv(symbol="EUR_USD", ...)  # Error! Needs EUR/USD

# ✅ Correct formats
oanda.fetch_ohlcv(symbol="EUR_USD", ...)      # Underscore
twelve_data.fetch_ohlcv(symbol="EUR/USD", ...)  # Slash
polygon.fetch_ohlcv(symbol="C:EURUSD", ...)    # Prefix + no separator
```

### 2. Timezone Confusion

```python
# OANDA returns timestamps in UTC
df = provider.fetch_ohlcv("EUR_USD", frequency="H1")

# If you're in New York (ET = UTC-5 or UTC-4):
# 12:00 UTC = 7:00 AM ET (or 8:00 AM ET during DST)

# Always work in UTC for forex data, convert for display only
import pytz

df = df.with_columns(
    pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("timestamp_et")
)
```

### 3. Intraday Data Volume

High-frequency FX data is **massive**:

```python
# 1 minute bars for EUR/USD, 1 year
# 365 days × 24 hours × 60 minutes = 525,600 bars

# At 7 columns × 8 bytes each = 56 bytes per row
# 525,600 rows × 56 bytes = ~29 MB for 1 pair, 1 year

# For 50 pairs: 1.45 GB!
# Use Parquet compression and incremental updates
```

### 4. Practice vs Live Data Discrepancy

```python
# Practice and live accounts may show slight price differences
practice = OandaProvider(account_type="practice")
live = OandaProvider(account_type="live")

practice_df = practice.fetch_ohlcv("EUR_USD", start="2024-11-04", end="2024-11-04")
live_df = live.fetch_ohlcv("EUR_USD", start="2024-11-04", end="2024-11-04")

# Differences are typically < 0.1 pip (negligible for backtesting)
# But ALWAYS backtest on same environment you'll trade in
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
- [Equities Data Guide](./equities.md) - US and global stocks
- [Futures Data Guide](./futures.md) - Futures and options

---

## Next Steps

1. **Sign up for OANDA practice account** (free): https://www.oanda.com/us-en/trading/api/
2. **Try the Quick Start examples** above with demo account
3. **Understand forex mechanics** - Pips, spreads, rollover, sessions
4. **Implement incremental updates** (Tutorial 03) for production systems
5. **Set up data quality validation** (Tutorial 04) for your FX pipeline
6. **Model spreads realistically** in backtests for accurate results

**Questions or issues?** See [CONTRIBUTING.md](../../CONTRIBUTING.md) or open a GitHub issue.
