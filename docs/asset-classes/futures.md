# Futures & Options Data Guide

**Asset Classes**: Futures contracts, Options on futures
**Available Providers**: Databento (primary), Polygon
**Difficulty**: Advanced (requires understanding of derivatives mechanics)
**Recommended For**: Institutional strategies, derivatives research, quantitative trading

---

## Overview

ML4T Data provides access to **institutional-grade futures and options data** through Databento, the gold standard for derivatives market data. Futures markets have unique complexities:

- **Expiration & Rolling**: Contracts expire monthly/quarterly, requiring continuous contract construction
- **Multiple Exchanges**: CME, CBOE, ICE, Eurex - each with different conventions
- **Session Times**: Futures trade nearly 24 hours with complex session breaks
- **Tick Data**: Ultra-high-frequency data (microsecond timestamps)
- **Margin Requirements**: Leveraged instruments with daily mark-to-market

**Quick Provider Comparison**:

| Provider | Coverage | Free Tier | Data Quality | Best For |
|----------|----------|-----------|--------------|----------|
| **Databento** | CME, CBOE, ICE + more | ❌ Paid only | ⭐⭐⭐⭐⭐ Institutional | Professional futures trading |
| **Polygon** | Limited futures | 5/min | ⭐⭐⭐ Good | Multi-asset portfolios |

**Important**: Futures data is **NOT free**. Databento requires paid subscription (~$30-50/month minimum for historical data).

---

## Databento (Recommended for Futures)

### Why Databento?

- ✅ **Institutional-grade quality** - Used by quant funds and prop traders
- ✅ **Normalized data** - Consistent schemas across all exchanges
- ✅ **Continuous contracts** - Automatic front-month rolling (.v.0 notation)
- ✅ **Tick-level precision** - Microsecond timestamps
- ✅ **CME, CBOE, ICE, Eurex** - Major derivatives exchanges worldwide
- ✅ **Multiple schemas** - OHLCV, trades, quotes, market-by-order

**Best for**: Quantitative futures strategies, high-frequency trading, institutional research

### Quick Start

```python
from ml4t.data.providers import DatabentoProvider

# Get API key from: https://databento.com/
# NOTE: Requires paid subscription ($30-50/month minimum)
provider = DatabentoProvider(
    api_key="your_databento_api_key",  # Or set DATABENTO_API_KEY env var
    dataset="GLBX.MDP3"  # CME Globex futures (default)
)

# Fetch ES (E-mini S&P 500) continuous front-month contract
df = provider.fetch_ohlcv(
    symbol="ES.v.0",  # .v.0 = continuous front month
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)

print(df.head())
```

**Output**:
```
shape: (252, 7)
┌─────────────────────┬──────────┬─────────┬─────────┬─────────┬─────────┬──────────┐
│ timestamp           │ symbol   │ open    │ high    │ low     │ close   │ volume   │
│ ---                 │ ---      │ ---     │ ---     │ ---     │ ---     │ ---      │
│ datetime[μs]        │ str      │ f64     │ f64     │ f64     │ f64     │ f64      │
╞═════════════════════╪══════════╪═════════╪═════════╪═════════╪═════════╪══════════╡
│ 2024-01-02 00:00:00 │ ES.v.0   │ 4742.50 │ 4793.75 │ 4742.50 │ 4783.75 │ 2.15e6   │
│ 2024-01-03 00:00:00 │ ES.v.0   │ 4783.25 │ 4800.50 │ 4745.00 │ 4746.75 │ 2.43e6   │
└─────────────────────┴──────────┴─────────┴─────────┴─────────┴─────────┴──────────┘
```

### Continuous Contracts (.v.0 Notation)

Futures contracts **expire** monthly or quarterly. To create continuous price series:

```python
# Continuous front-month contract (automatically rolls)
"ES.v.0"   # E-mini S&P 500 front month
"NQ.v.0"   # E-mini NASDAQ-100 front month
"CL.v.0"   # WTI Crude Oil front month
"GC.v.0"   # Gold front month

# Continuous 2nd month contract
"ES.v.1"   # E-mini S&P 500 2nd month
"CL.v.1"   # WTI Crude Oil 2nd month

# Continuous 3rd month contract
"ES.v.2"   # E-mini S&P 500 3rd month

# Specific contract month (NO automatic rolling)
"ESZ4"     # E-mini S&P 500 December 2024
"ESH5"     # E-mini S&P 500 March 2025
```

**Why continuous contracts matter**:

```python
# ❌ Bad: Using individual contract months
df_dec = provider.fetch_ohlcv("ESZ4", start="2024-10-01", end="2024-12-15")
df_mar = provider.fetch_ohlcv("ESH5", start="2024-12-16", end="2025-03-15")
# Problem: Price gap at rollover date, backtests break

# ✅ Good: Using continuous contract
df = provider.fetch_ohlcv("ES.v.0", start="2024-10-01", end="2025-03-15")
# Databento automatically handles rollover, seamless price series
```

### Supported Futures

**CME Globex** (dataset: `GLBX.MDP3`):

| Symbol | Contract | Tick Size | Contract Size |
|--------|----------|-----------|---------------|
| **ES** | E-mini S&P 500 | 0.25 pts = $12.50 | $50 × S&P 500 |
| **NQ** | E-mini NASDAQ-100 | 0.25 pts = $5.00 | $20 × NASDAQ-100 |
| **YM** | E-mini Dow Jones | 1.00 pt = $5.00 | $5 × DJIA |
| **RTY** | E-mini Russell 2000 | 0.10 pts = $5.00 | $50 × Russell 2000 |
| **CL** | WTI Crude Oil | $0.01 = $10.00 | 1,000 barrels |
| **GC** | Gold | $0.10 = $10.00 | 100 troy oz |
| **SI** | Silver | $0.005 = $25.00 | 5,000 troy oz |
| **ZB** | 30-Year T-Bond | 1/32 = $31.25 | $100,000 face value |
| **ZN** | 10-Year T-Note | 1/64 = $15.625 | $100,000 face value |
| **ZF** | 5-Year T-Note | 1/128 = $7.8125 | $100,000 face value |
| **6E** | Euro FX | $0.00005 = $6.25 | €125,000 |
| **6J** | Japanese Yen | $0.0000005 = $6.25 | ¥12,500,000 |
| **6B** | British Pound | $0.0001 = $6.25 | £62,500 |

**Equity Index Futures**:
```python
# Major indices
"ES.v.0"   # S&P 500
"NQ.v.0"   # NASDAQ-100
"YM.v.0"   # Dow Jones
"RTY.v.0"  # Russell 2000

# International
"MES.v.0"  # Micro E-mini S&P 500 (1/10th size)
"MNQ.v.0"  # Micro E-mini NASDAQ-100
```

**Commodity Futures**:
```python
# Energy
"CL.v.0"   # WTI Crude Oil
"NG.v.0"   # Natural Gas
"HO.v.0"   # Heating Oil
"RB.v.0"   # RBOB Gasoline

# Metals
"GC.v.0"   # Gold
"SI.v.0"   # Silver
"HG.v.0"   # Copper
"PL.v.0"   # Platinum

# Agriculture
"ZC.v.0"   # Corn
"ZS.v.0"   # Soybeans
"ZW.v.0"   # Wheat
"KC.v.0"   # Coffee
"CT.v.0"   # Cotton
```

**Interest Rate Futures**:
```python
"ZB.v.0"   # 30-Year T-Bond
"ZN.v.0"   # 10-Year T-Note
"ZF.v.0"   # 5-Year T-Note
"ZT.v.0"   # 2-Year T-Note
```

**Currency Futures**:
```python
"6E.v.0"   # Euro FX
"6J.v.0"   # Japanese Yen
"6B.v.0"   # British Pound
"6C.v.0"   # Canadian Dollar
"6A.v.0"   # Australian Dollar
```

### Datasets (Exchanges)

Databento provides data from multiple exchanges:

```python
# CME Globex (most futures)
provider = DatabentoProvider(dataset="GLBX.MDP3")

# CME (all CME markets)
provider = DatabentoProvider(dataset="CME.MDP3")

# CBOE Options (equity options)
provider = DatabentoProvider(dataset="OPRA.PILLAR")

# US Equities
provider = DatabentoProvider(dataset="XNAS.ITCH")  # NASDAQ
provider = DatabentoProvider(dataset="XNYS.TRADES")  # NYSE
```

**Available datasets**:
- `GLBX.MDP3` - CME Globex (futures, options on futures)
- `CME.MDP3` - CME (all markets)
- `XCME.MDP3` - CME Crypto
- `OPRA.PILLAR` - CBOE Equity Options
- `XNAS.ITCH` - NASDAQ Equities
- `XNYS.TRADES` - NYSE Equities
- `BATS.PITCH` - BATS Equities

### Schemas (Data Types)

Databento supports multiple data schemas beyond OHLCV:

```python
# OHLCV aggregations
"ohlcv-1m"  # 1-minute bars
"ohlcv-1h"  # 1-hour bars
"ohlcv-1d"  # 1-day bars

# Tick data
"trades"    # Individual trades (time & sales)
"quotes"    # Top of book (best bid/ask)
"mbo"       # Market by order (full order book)
"mbp-1"     # Market by price (1 level)
"mbp-10"    # Market by price (10 levels)
```

**Example: Fetching tick data**:

```python
provider = DatabentoProvider(dataset="GLBX.MDP3", default_schema="trades")

# Fetch trade-by-trade data for ES
trades = provider.fetch_ohlcv(
    symbol="ES.v.0",
    start="2024-11-04",
    end="2024-11-04",
    frequency="tick"  # Uses default_schema="trades"
)

# Result: Every single trade with microsecond timestamps
print(f"Total trades: {len(trades)}")
# Output: Total trades: 1,247,352 (for 1 day of ES)
```

### Pricing

**Databento pricing** (as of 2024):

**Historical Data API**:
- **Starter**: ~$30/month - Limited historical data
- **Professional**: ~$50-100/month - Extended history + more symbols
- **Enterprise**: Custom pricing - Unlimited access

**Live Data**:
- **Real-time feed**: $100-500/month depending on exchanges
- **WebSocket API**: Included with subscription

**Free Tier**: ❌ None - All Databento data is paid

**Cost Comparison**:
| Provider | Futures Data | Cost | Quality |
|----------|--------------|------|---------|
| Databento | CME, CBOE, ICE | $30-100/month | ⭐⭐⭐⭐⭐ Best |
| Polygon | Limited | $199/month | ⭐⭐⭐ Good |
| IQFeed | Extensive | $100+/month | ⭐⭐⭐⭐ Very Good |
| CQG | Professional | $500+/month | ⭐⭐⭐⭐⭐ Best |

**Winner**: Databento (best value for institutional quality)

### Session Times & Date Adjustments

Futures markets trade nearly 24 hours with session breaks:

```python
# CME Globex session times (US/Central)
# Sunday 5:00 PM - Friday 4:00 PM CT
# Daily break: 4:00 PM - 5:00 PM CT

# ES (E-mini S&P 500) specific:
# Sunday 5:00 PM CT - Monday 4:00 PM CT = "Monday" session
# But in UTC: Monday 00:00 - Tuesday 22:00

# Databento handles this with adjust_session_dates
provider = DatabentoProvider(
    dataset="GLBX.MDP3",
    adjust_session_dates=True,  # Align dates with session start
    session_start_hour_utc=22  # 22:00 UTC = 5:00 PM CT (CST)
)

# Now daily bars align with CME session dates (not calendar dates)
```

### Rate Limits

**Databento Historical API**: 100 requests per second

```python
# Very generous for historical data fetching
symbols = [f"ES.v.0", "NQ.v.0", "YM.v.0", ...]  # 50 futures contracts

for symbol in symbols:
    df = provider.fetch_ohlcv(symbol, start="2024-01-01", end="2024-12-31")
    # 50 symbols = 50 requests = completes in 0.5 seconds
```

**Live WebSocket**: Unlimited (within subscription limits)

### Complete Example: Multi-Contract Strategy

```python
from ml4t.data.providers import DatabentoProvider
import polars as pl

# Initialize provider
provider = DatabentoProvider(
    dataset="GLBX.MDP3",
    adjust_session_dates=True
)

# Build a diversified futures portfolio
contracts = {
    "ES.v.0": "E-mini S&P 500",
    "NQ.v.0": "E-mini NASDAQ-100",
    "CL.v.0": "WTI Crude Oil",
    "GC.v.0": "Gold",
    "ZN.v.0": "10-Year T-Note",
}

# Fetch 1 year of daily data
all_data = []
for symbol, name in contracts.items():
    df = provider.fetch_ohlcv(
        symbol=symbol,
        start="2023-01-01",
        end="2023-12-31",
        frequency="daily"
    )
    df = df.with_columns(pl.lit(name).alias("contract"))
    all_data.append(df)

# Combine all contracts
portfolio = pl.concat(all_data)

# Calculate daily returns
returns = portfolio.group_by("symbol").agg([
    pl.col("close").pct_change().mean().alias("avg_return"),
    pl.col("close").pct_change().std().alias("volatility"),
    (pl.col("close").pct_change().mean() / pl.col("close").pct_change().std()).alias("sharpe")
])

print(returns.sort("sharpe", descending=True))
```

### Error Handling

```python
from ml4t.data.core.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    NetworkError
)

try:
    df = provider.fetch_ohlcv("INVALID.v.0", start="2024-01-01", end="2024-12-31")
except DataNotAvailableError as e:
    print(f"Contract not available: {e}")
    # Check Databento's symbol list for valid contracts

try:
    # Using expired API key
    provider = DatabentoProvider(api_key="expired_key")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Renew subscription or check API key
```

### When to Use Databento

✅ **Good for**:
- Professional futures trading strategies
- High-frequency derivatives research
- Institutional-grade backtesting
- Tick-level analysis
- Multi-exchange strategies
- Options on futures

❌ **Not ideal for**:
- Budget-constrained projects (requires $30+/month)
- Beginners (steep learning curve)
- Stock-only strategies (use Tiingo instead)
- Cryptocurrency-only (use CoinGecko instead)

---

## Polygon (Multi-Asset Alternative)

### Why Polygon?

- ✅ **Futures + Stocks + Crypto + Options** - Multi-asset coverage
- ✅ **Real-time + historical** data
- ⚠️ **Free tier very limited** (5 API calls/minute)
- ⚠️ **Paid tier expensive** ($199-$399/month)

**Best for**: Multi-asset portfolios needing futures + equities

### Quick Start

```python
from ml4t.data.providers import PolygonProvider

# Get API key from: https://polygon.io/
provider = PolygonProvider(api_key="your_polygon_api_key")

# Fetch ES futures data
df = provider.fetch_ohlcv(
    symbol="ES",  # E-mini S&P 500 (generic symbol)
    start="2024-01-01",
    end="2024-12-31",
    frequency="daily"
)
```

**Pricing**:
- **Free**: 5 API calls per minute (very restrictive)
- **Starter**: $199/month - Stocks + Forex + Crypto
- **Developer**: $399/month - Adds Options + Futures
- **Advanced**: Custom pricing

**When to use**: Need futures + stocks + options from single provider, budget >$199/month

---

## Futures Market Characteristics

### Contract Expiration & Rolling

Futures contracts **expire** on specific dates:

```python
# ES (E-mini S&P 500) expiration schedule:
# March (H), June (M), September (U), December (Z)
# 3rd Friday of expiration month

# Example expiration dates for 2024:
"ESH4"  # March 2024 - expires March 15, 2024
"ESM4"  # June 2024 - expires June 21, 2024
"ESU4"  # September 2024 - expires September 20, 2024
"ESZ4"  # December 2024 - expires December 20, 2024

# After expiration, contract becomes worthless
# Must "roll" to next contract to maintain position
```

**Roll dates** (when to switch contracts):

```python
# Typically roll 5-10 days before expiration
# ESZ4 (Dec 2024) expires Dec 20
# Roll to ESH5 (Mar 2025) around Dec 10-15

# Using continuous contracts (.v.0) automates this:
df = provider.fetch_ohlcv("ES.v.0", start="2024-01-01", end="2025-12-31")
# Databento automatically switches ESH4 → ESM4 → ESU4 → ESZ4 → ESH5
```

### Contract Months & Codes

**Month codes** (universal):

```python
# Jan=F, Feb=G, Mar=H, Apr=J, May=K, Jun=M,
# Jul=N, Aug=Q, Sep=U, Oct=V, Nov=X, Dec=Z

# Year: Last digit (2024 = 4, 2025 = 5)

# Examples:
"ESH5"  # ES March 2025
"CLZ4"  # CL December 2024
"GCM5"  # GC June 2025
```

**Not all contracts trade all months**:

```python
# ES (S&P 500): March, June, Sep, Dec (quarterly)
"ESH", "ESM", "ESU", "ESZ"

# CL (Crude Oil): All 12 months
"CLF", "CLG", "CLH", "CLJ", "CLK", "CLM", "CLN", "CLQ", "CLU", "CLV", "CLX", "CLZ"

# GC (Gold): Feb, Apr, Jun, Aug, Oct, Dec (bi-monthly)
"GCG", "GCJ", "GCM", "GCQ", "GCV", "GCZ"
```

### Tick Size & Value

Each futures contract has specific **tick size** and **value per tick**:

```python
# ES (E-mini S&P 500)
# Tick size: 0.25 points
# Tick value: $12.50 (0.25 × $50 multiplier)
# If ES moves from 4750.00 to 4750.25 → $12.50 profit/loss per contract

# CL (Crude Oil)
# Tick size: $0.01 per barrel
# Tick value: $10.00 ($0.01 × 1000 barrels)
# If CL moves from $75.00 to $75.01 → $10.00 profit/loss per contract

# GC (Gold)
# Tick size: $0.10 per troy ounce
# Tick value: $10.00 ($0.10 × 100 oz)
# If GC moves from $2000.00 to $2000.10 → $10.00 profit/loss per contract
```

### Margin Requirements

Futures are **leveraged** - you don't pay full contract value upfront:

```python
# Example: ES (E-mini S&P 500) @ 4750
# Notional value: 4750 × $50 = $237,500
# Initial margin: ~$12,000 (5% of notional)
# Maintenance margin: ~$11,000

# Leverage: $237,500 / $12,000 = ~20:1

# If ES moves 1% (47.5 points):
# Change in contract value: 47.5 × $50 = $2,375
# Return on margin: $2,375 / $12,000 = 19.8% (on 1% price move!)
```

**Risk warning**: High leverage magnifies both gains and losses.

---

## Best Practices

### 1. Use Continuous Contracts for Backtesting

```python
# ❌ Bad: Using individual contracts
df1 = provider.fetch_ohlcv("ESZ4", start="2024-10-01", end="2024-12-20")
df2 = provider.fetch_ohlcv("ESH5", start="2024-12-21", end="2025-03-20")
# Problem: Price gap at rollover, discontinuous series

# ✅ Good: Using continuous contract
df = provider.fetch_ohlcv("ES.v.0", start="2024-10-01", end="2025-03-20")
# Databento handles rollover seamlessly
```

### 2. Account for Roll Costs

```python
# When rolling futures, there's a "roll cost" (calendar spread)
# Example: ESZ4 @ 4750, ESH5 @ 4755
# Roll cost: 5 points = $250 per contract

# Include roll costs in backtests:
roll_cost_points = 5
roll_cost_dollars = roll_cost_points * 50  # ES multiplier
annual_rolls = 4  # Quarterly contracts
annual_roll_cost = roll_cost_dollars * annual_rolls * num_contracts
```

### 3. Understand Session Times

```python
# Futures trade nearly 24/5, but have session breaks
# ES (CME Globex):
# Sunday 5:00 PM CT - Friday 4:00 PM CT
# Daily break: 4:00 PM - 5:00 PM CT

# Use adjust_session_dates for proper date alignment
provider = DatabentoProvider(
    adjust_session_dates=True,
    session_start_hour_utc=22  # 5:00 PM CT = 22:00 UTC (CST)
)
```

### 4. Validate Futures Data

```python
from ml4t.data.validation import validate_ohlcv

df = provider.fetch_ohlcv("ES.v.0", start="2024-01-01", end="2024-12-31")

# Standard OHLCV validation
validate_ohlcv(df)

# Futures-specific checks
assert df.filter(pl.col("volume") == 0).is_empty(), "No zero-volume bars"

# Check for contract roll gaps (expected at quarterly roll dates)
gaps = df.select("timestamp").diff().filter(pl.col("timestamp") > timedelta(days=3))
print(f"Contract roll gaps: {len(gaps)}")  # Should be 3-4 per year
```

### 5. Use Incremental Updates

```python
from ml4t.data.provider_updater import ProviderUpdater

updater = ProviderUpdater(provider="databento")

# First run: Fetch 1 year of data
df = updater.update("ES.v.0", frequency="daily", lookback_days=365)

# Daily runs: Only fetch last 1-2 days
df = updater.update("ES.v.0", frequency="daily", lookback_days=2)
# Saves API quota and reduces costs
```

---

## Common Gotchas

### 1. Continuous vs Specific Contracts

```python
# ❌ Wrong: Using .v.0 for live trading
# ES.v.0 is synthetic, can't actually trade it
live_order = "Buy 1 ES.v.0 @ market"  # Invalid!

# ✅ Correct: Use specific contract for live trading
live_order = "Buy 1 ESH5 @ market"  # Valid (March 2025 contract)

# Rule: Use .v.0 for backtesting, specific contracts for live trading
```

### 2. Tick Size Violations

```python
# Each contract has minimum tick size
# ES: 0.25 points
# Can't place limit order at 4750.10 (invalid tick)

# ❌ Wrong
limit_price = 4750.10  # Not a valid ES tick

# ✅ Correct
limit_price = 4750.25  # Valid ES tick (multiple of 0.25)
```

### 3. Expiration Surprises

```python
# Futures contracts expire - don't get caught!

# Example: ESZ4 expires Dec 20, 2024
# If you hold ESZ4 on Dec 21, you may be:
# 1. Physically settled (rare for financial futures)
# 2. Cash settled (common for index futures)
# 3. Assigned a large loss if on wrong side

# Always roll before expiration (5-10 days prior)
```

### 4. Margin Calls

```python
# Futures are marked-to-market daily
# If position moves against you, broker may require additional margin

# Example: Long 1 ES @ 4750, margin = $12,000
# ES drops to 4700 (-50 points = -$2,500)
# Your margin account: $12,000 - $2,500 = $9,500
# Maintenance margin: $11,000
# → Margin call! Must deposit $1,500 or close position
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
- [Forex Data Guide](./forex.md) - Foreign exchange pairs

---

## Next Steps

1. **Sign up for Databento** ($30-50/month): https://databento.com/
2. **Understand futures mechanics** - Expiration, rolling, margin, tick sizes
3. **Start with simple strategies** - Single contract (ES or NQ) daily data
4. **Use continuous contracts** (.v.0) for backtesting
5. **Account for roll costs** and session times in strategies
6. **Validate data quality** thoroughly before trading with real money

**Important**: Futures trading carries significant risk due to leverage. Always backtest thoroughly and start with paper trading.

**Questions or issues?** See [CONTRIBUTING.md](../../CONTRIBUTING.md) or open a GitHub issue.
