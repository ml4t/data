# Understanding OHLCV Data

**Target Audience**: Beginners to quantitative finance
**Time to Read**: 10 minutes
**Prerequisites**: Basic Python knowledge

## What is OHLCV?

OHLCV stands for:
- **O**pen - First traded price in the period
- **H**igh - Highest traded price in the period
- **L**ow - Lowest traded price in the period
- **C**lose - Last traded price in the period
- **V**olume - Total quantity traded in the period

This is the most common format for historical market data and is used across stocks, crypto, forex, and futures markets.

## Why OHLCV?

OHLCV data compresses thousands of individual trades into a single summary bar. For example:
- **Daily OHLCV**: One row per day summarizing all trades that day
- **Hourly OHLCV**: One row per hour
- **1-minute OHLCV**: One row per minute

This compression makes historical analysis tractable. Storing every trade for Apple stock over 10 years would require billions of rows; OHLCV gives you ~2,500 rows (daily bars) that capture the essential price movement.

## Example OHLCV Data

Here's what Bitcoin OHLCV data looks like:

```python
from ml4t.data.providers import CoinGeckoProvider

provider = CoinGeckoProvider()
btc = provider.fetch_ohlcv("bitcoin", "2024-01-01", "2024-01-05")
print(btc)
```

Output:
```
shape: (5, 7)
┌─────────────────────┬────────┬──────────┬──────────┬──────────┬──────────┬─────────────┐
│ timestamp           │ symbol │ open     │ high     │ low      │ close    │ volume      │
│ ---                 │ ---    │ ---      │ ---      │ ---      │ ---      │ ---         │
│ datetime[μs]        │ str    │ f64      │ f64      │ f64      │ f64      │ f64         │
╞═════════════════════╪════════╪══════════╪══════════╪══════════╪══════════╪═════════════╡
│ 2024-01-01 00:00:00 │ BTC    │ 42258.31 │ 44821.04 │ 42104.50 │ 44156.51 │ 23847539882 │
│ 2024-01-02 00:00:00 │ BTC    │ 44156.51 │ 45899.83 │ 43979.70 │ 45561.30 │ 30127839201 │
│ 2024-01-03 00:00:00 │ BTC    │ 45561.30 │ 46367.27 │ 43986.14 │ 44663.77 │ 31890472847 │
│ 2024-01-04 00:00:00 │ BTC    │ 44663.77 │ 45540.62 │ 43476.52 │ 44141.86 │ 27364928173 │
│ 2024-01-05 00:00:00 │ BTC    │ 44141.86 │ 45848.35 │ 43668.28 │ 43994.92 │ 25718365291 │
└─────────────────────┴────────┴──────────┴──────────┴──────────┴──────────┴─────────────┘
```

## Interpreting OHLCV Bars

### The Anatomy of a Bar

Let's examine January 1, 2024:

```
timestamp:  2024-01-01 00:00:00
open:       $42,258.31  ← Bitcoin started the day at this price
high:       $44,821.04  ← Reached this peak during the day
low:        $42,104.50  ← Dropped to this low during the day
close:      $44,156.51  ← Ended the day at this price
volume:     23,847,539,882  ← ~$23.8B worth traded
```

**Key Insights**:
1. Price went UP: Open ($42,258) < Close ($44,156) → Bullish day
2. Price range: $2,716 spread between high and low → Volatile day
3. Close near high: Close ($44,156) ≈ High ($44,821) → Strong buying pressure

### Visualizing OHLCV: Candlestick Charts

OHLCV data is typically visualized as **candlesticks**:

```
    High ──┬──
          │  │
    Open ─┤  │  Green candle (Close > Open)
          │  │
   Close ─┤  │
          │  │
     Low ──┴──


    High ──┬──
          │  │
   Close ─┤  │  Red candle (Close < Open)
          │  │
    Open ─┤  │
          │  │
     Low ──┴──
```

**Green (Bullish)**: Price went up (Close > Open)
**Red (Bearish)**: Price went down (Close < Open)

The "body" is the rectangle between Open and Close. The "wicks" (lines above/below) show the High and Low.

## OHLCV Invariants (Data Quality Checks)

Valid OHLCV data must satisfy these **invariants**:

```python
# CRITICAL INVARIANTS - Always true for valid data
assert (df["high"] >= df["low"]).all()      # High always ≥ Low
assert (df["high"] >= df["open"]).all()     # High always ≥ Open
assert (df["high"] >= df["close"]).all()    # High always ≥ Close
assert (df["low"] <= df["open"]).all()      # Low always ≤ Open
assert (df["low"] <= df["close"]).all()     # Low always ≤ Close
assert (df["volume"] >= 0).all()            # Volume always non-negative
```

ML4T Data automatically validates these invariants for all provider data. If a provider returns invalid data, you'll get a clear error:

```python
DataValidationError: OHLCV invariant violated: High < Low at 2024-01-15
```

## Common OHLCV Patterns

### 1. Doji (Indecision)

```
Open ≈ Close, with wicks above and below
```
- Buyers and sellers fought to a draw
- Often signals potential reversal

### 2. Hammer (Bullish Reversal)

```
Long lower wick, small body near high
```
- Price dropped significantly but recovered
- Buyers rejected lower prices

### 3. Shooting Star (Bearish Reversal)

```
Long upper wick, small body near low
```
- Price rallied but sellers pushed it back down
- Often appears at tops

### 4. Engulfing Pattern

```
Large candle completely covers previous candle
```
- **Bullish Engulfing**: Green candle engulfs red → Reversal up
- **Bearish Engulfing**: Red candle engulfs green → Reversal down

## Frequencies: Daily, Hourly, Minute

OHLCV data comes in different **frequencies**:

```python
from ml4t.data.providers import TiingoProvider

provider = TiingoProvider(api_key="your_key")

# Daily bars (most common for backtesting)
daily = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31", frequency="daily")
# → 252 trading days ≈ 252 rows

# Weekly bars (for longer-term analysis)
weekly = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31", frequency="weekly")
# → 52 weeks ≈ 52 rows

# Monthly bars (for macro analysis)
monthly = provider.fetch_ohlcv("AAPL", "2020-01-01", "2024-12-31", frequency="monthly")
# → 60 months = 60 rows
```

**Rule of Thumb**:
- **Daily**: Most common for backtesting (years of data, manageable size)
- **Hourly/Minute**: Intraday strategies (short-term, large datasets)
- **Weekly/Monthly**: Long-term macro strategies

## Adjusted vs. Unadjusted Prices

### The Problem: Corporate Actions

Stock prices need adjustment for:
1. **Stock Splits**: 2-for-1 split → Price halves, shares double
2. **Dividends**: $1 dividend → Price drops $1 on ex-dividend date

Without adjustment, you'd see artificial "crashes" that weren't real price moves.

### The Solution: Adjusted Close

Most providers return **adjusted close** prices:

```python
# Raw (unadjusted) close: What actually traded
raw_close = 100.00

# Adjusted close: Accounts for splits/dividends
# If there was a 2-for-1 split, historical prices are halved
adjusted_close = 50.00  # Retroactively adjusted
```

**ML4T Data Standard**: The `close` column is always the **adjusted close** (if available from provider). Some providers also return `unadjusted_close` or raw prices.

### Example: Apple Stock Split (August 2020)

Apple did a 4-for-1 stock split on August 31, 2020:

```python
# Before split (August 30, 2020)
unadjusted_close = 499.23  # Actually traded at ~$500
adjusted_close   = 124.81  # Retroactively divided by 4

# After split (August 31, 2020)
unadjusted_close = 129.04  # New shares at ~$129
adjusted_close   = 129.04  # Same as unadjusted (no future splits)
```

**Use adjusted prices for backtesting** to avoid false signals.

## Working with OHLCV in ML4T Data

### Standard Schema

Every ML4T Data provider returns the same schema:

```python
{
    "timestamp": pl.Datetime,   # UTC datetime
    "symbol": pl.String,        # Uppercase symbol
    "open": pl.Float64,         # Opening price
    "high": pl.Float64,         # High price
    "low": pl.Float64,          # Low price
    "close": pl.Float64,        # Closing price (adjusted)
    "volume": pl.Float64,       # Trading volume
}
```

This consistency lets you swap providers without changing your code:

```python
# Works with any provider!
def analyze_volatility(df):
    df = df.with_columns(
        ((df["high"] - df["low"]) / df["close"]).alias("intraday_range")
    )
    return df.select(["timestamp", "close", "intraday_range"])

# Use with CoinGecko
btc = CoinGeckoProvider().fetch_ohlcv("bitcoin", start, end)
analyze_volatility(btc)

# Use with Tiingo
aapl = TiingoProvider(api_key="key").fetch_ohlcv("AAPL", start, end)
analyze_volatility(aapl)  # Same function!
```

## Calculating Common Metrics

### 1. Daily Returns

```python
# Simple returns (%)
df = df.with_columns(
    ((df["close"] - df["close"].shift(1)) / df["close"].shift(1) * 100)
    .alias("return_pct")
)

# Log returns (for compounding)
df = df.with_columns(
    (df["close"].log() - df["close"].shift(1).log()).alias("log_return")
)
```

### 2. Intraday Volatility

```python
# Daily range as % of close
df = df.with_columns(
    ((df["high"] - df["low"]) / df["close"] * 100).alias("range_pct")
)

# True Range (for ATR calculation)
df = df.with_columns(
    pl.max([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ]).alias("true_range")
)
```

### 3. Volume Analysis

```python
# Abnormal volume (vs. 20-day average)
df = df.with_columns(
    df["volume"].rolling_mean(window_size=20).alias("avg_volume_20d")
)
df = df.with_columns(
    (df["volume"] / df["avg_volume_20d"]).alias("volume_ratio")
)

# High volume days (>2x average)
high_volume_days = df.filter(df["volume_ratio"] > 2.0)
```

## Common Pitfalls

### 1. Look-Ahead Bias

**WRONG**:
```python
# DON'T USE FUTURE DATA!
df = df.with_columns(
    df["close"].shift(-1).alias("next_close")  # ❌ Uses tomorrow's price!
)
```

**RIGHT**:
```python
# Only use past/current data
df = df.with_columns(
    df["close"].shift(1).alias("prev_close")  # ✅ Uses yesterday's price
)
```

### 2. Survivor Bias

Only analyzing stocks that exist today misses stocks that went bankrupt or were delisted. Good data providers include delisted stocks.

### 3. Ignoring Splits/Dividends

Always use adjusted prices for backtesting unless you have a specific reason not to.

### 4. Misaligned Frequencies

Don't mix daily and intraday data without careful timestamp alignment.

## Next Steps

Now that you understand OHLCV data:

1. **Explore Providers**: [Provider Selection Guide](../provider-selection-guide.md)
2. **Learn Rate Limiting**: [Tutorial 02: Rate Limiting Best Practices](02_rate_limiting.md)
3. **Master Updates**: [Tutorial 03: Incremental Updates](03_incremental_updates.md)
4. **Build Strategies**: Start with simple moving average crossovers

## Further Reading

- **Books**:
  - *Advances in Financial Machine Learning* by Marcos López de Prado
  - *Quantitative Trading* by Ernest Chan
- **ML4T Data Docs**:
  - [Creating a Provider](../creating_a_provider.md)
  - [Data Validation](../../src/ml4t-data/validation/)
  - [Storage Layer](../user-guide/storage.md)

---

**Next Tutorial**: [02: Rate Limiting Best Practices](02_rate_limiting.md)
