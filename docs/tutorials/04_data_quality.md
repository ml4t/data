# Data Quality Validation

**Target Audience**: Quant researchers and production engineers
**Time to Read**: 15 minutes
**Prerequisites**: [Understanding OHLCV](01_understanding_ohlcv.md)

## Why Data Quality Matters

**Bad data = Bad strategies = Lost money**

A single bad data point can:
- Generate false trading signals
- Corrupt backtest results
- Cause production trading errors
- Lead to incorrect risk calculations

Example of how bad data causes problems:

```python
# Bad data: High < Low (impossible!)
timestamp  | symbol | open  | high  | low   | close
2024-01-15 | AAPL   | 185.0 | 183.0 | 187.0 | 186.0
                            ↑ ERROR! High should be >= Low

# This corrupt data could trigger:
# - False "flash crash" signals
# - Invalid volatility calculations
# - Broken technical indicators (RSI, Bollinger Bands)
# - Position sizing errors
```

**ML4T Data's approach**: Validate everything, fail loudly, never silently corrupt your data.

## OHLCV Invariants

These rules MUST be true for valid OHLCV data:

```python
# The 6 OHLCV Invariants
assert (df["high"] >= df["low"]).all()      # 1. High >= Low
assert (df["high"] >= df["open"]).all()     # 2. High >= Open
assert (df["high"] >= df["close"]).all()    # 3. High >= Close
assert (df["low"] <= df["open"]).all()      # 4. Low <= Open
assert (df["low"] <= df["close"]).all()     # 5. Low <= Close
assert (df["volume"] >= 0).all()            # 6. Volume >= 0
```

**ML4T Data validates these automatically for all provider data.**

### Example: Catching Bad Data

```python
from ml4t.data.providers import TiingoProvider

provider = TiingoProvider(api_key="key")

try:
    # ML4T Data validates on fetch
    data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    print("✅ Data passed validation")

except DataValidationError as e:
    # Bad data caught immediately
    print(f"❌ Invalid data: {e}")
    # Example error:
    # "OHLCV invariant violated at 2024-01-15: High (183.0) < Low (187.0)"
```

## Schema Validation

Beyond OHLCV invariants, ML4T Data validates:

### 1. Column Presence

```python
# Required columns
REQUIRED_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

# ML4T Data checks all providers return these columns
if not all(col in df.columns for col in REQUIRED_COLUMNS):
    raise DataValidationError("Missing required columns")
```

### 2. Data Types

```python
# Expected types
EXPECTED_TYPES = {
    "timestamp": pl.Datetime,
    "symbol": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# ML4T Data enforces correct types
if df["timestamp"].dtype != pl.Datetime:
    raise DataValidationError("timestamp must be Datetime type")
```

### 3. Null/NaN Checks

```python
# No nulls allowed in critical columns
critical_cols = ["timestamp", "symbol", "close"]

for col in critical_cols:
    if df[col].null_count() > 0:
        raise DataValidationError(f"Null values found in {col}")

# NaN checks (different from null)
if df["close"].is_nan().any():
    raise DataValidationError("NaN values found in close prices")
```

## Anomaly Detection

ML4T Data includes anomaly detection for suspicious (but technically valid) data:

### 1. Price Spikes

```python
# Detect abnormal price changes
df = df.with_columns(
    ((df["close"] - df["close"].shift(1)) / df["close"].shift(1) * 100)
    .alias("pct_change")
)

# Flag extreme moves (>20% in one day for large-cap stocks)
anomalies = df.filter(df["pct_change"].abs() > 20)

if len(anomalies) > 0:
    print(f"⚠️  Warning: {len(anomalies)} extreme price moves detected")
    print(anomalies.select(["timestamp", "symbol", "close", "pct_change"]))
```

### 2. Zero Volume

```python
# Zero volume on trading days is suspicious
zero_volume = df.filter(df["volume"] == 0)

if len(zero_volume) > 0:
    print(f"⚠️  Warning: {len(zero_volume)} days with zero volume")
    # Could indicate:
    # - Trading halts
    # - Data provider issues
    # - Delisted stocks
```

### 3. Constant Prices

```python
# Prices don't change for multiple days (suspicious)
df = df.with_columns(
    (df["close"] == df["close"].shift(1)).alias("price_unchanged")
)

consecutive_unchanged = df.filter(
    df["price_unchanged"] & df["price_unchanged"].shift(1)
)

if len(consecutive_unchanged) > 0:
    print(f"⚠️  Warning: {len(consecutive_unchanged)} periods with no price change")
```

### 4. Timestamp Gaps

```python
from datetime import timedelta

# Expected 1-day intervals (for daily data)
df = df.with_columns(
    (df["timestamp"] - df["timestamp"].shift(1)).alias("time_gap")
)

# Flag gaps > 7 days (weekends/holidays ok, long gaps suspicious)
large_gaps = df.filter(df["time_gap"] > timedelta(days=7))

if len(large_gaps) > 0:
    print(f"⚠️  Warning: {len(large_gaps)} large time gaps detected")
    print(large_gaps.select(["timestamp", "time_gap"]))
```

## Using ML4T Data's Validation Module

```python
from ml4t.data.validation import OHLCVValidator, ValidationConfig

# Configure validation rules
config = ValidationConfig(
    check_ohlcv_invariants=True,
    check_nulls=True,
    check_price_spikes=True,
    max_daily_change_pct=30.0,  # Flag >30% moves
    check_zero_volume=True,
    check_timestamp_gaps=True,
)

# Create validator
validator = OHLCVValidator(config)

# Validate data
result = validator.validate(df)

if result.is_valid:
    print("✅ Data passed all validation checks")
else:
    print(f"❌ Validation failed: {len(result.errors)} errors")
    for error in result.errors:
        print(f"  - {error}")

    # Optionally raise exception
    if result.severity == "critical":
        raise DataValidationError("Critical validation failures")
```

## Handling Bad Data

### Option 1: Reject and Alert

```python
try:
    data = provider.fetch_ohlcv("AAPL", start, end)
    validator.validate(data)
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
    # Send alert to monitoring system
    alert_system.send(f"Bad data from {provider.name()}: {e}")
    # Don't use bad data
    return None
```

### Option 2: Clean and Continue

```python
def clean_ohlcv_data(df):
    """Clean data by removing invalid rows."""
    original_len = len(df)

    # Remove rows with invariant violations
    df = df.filter(df["high"] >= df["low"])
    df = df.filter(df["high"] >= df["open"])
    df = df.filter(df["high"] >= df["close"])
    df = df.filter(df["low"] <= df["open"])
    df = df.filter(df["low"] <= df["close"])

    # Remove null values
    df = df.drop_nulls(subset=["timestamp", "close"])

    # Remove NaN values
    df = df.filter(~df["close"].is_nan())

    cleaned_len = len(df)
    dropped = original_len - cleaned_len

    if dropped > 0:
        logger.warning(f"Dropped {dropped} invalid rows ({dropped/original_len*100:.1f}%)")

    return df

# Use cleaned data (with caution!)
data = provider.fetch_ohlcv("AAPL", start, end)
data = clean_ohlcv_data(data)
```

### Option 3: Fill Forward (Last Resort)

```python
def fill_missing_data(df):
    """Fill missing values with forward fill (use sparingly)."""
    # Fill nulls with previous valid value
    df = df.with_columns([
        pl.col("open").fill_null(strategy="forward"),
        pl.col("high").fill_null(strategy="forward"),
        pl.col("low").fill_null(strategy="forward"),
        pl.col("close").fill_null(strategy="forward"),
        pl.col("volume").fill_null(0),  # Volume = 0 for missing days
    ])

    logger.warning("Applied forward fill to missing data")
    return df
```

## Provider-Specific Validation

Different providers have different data quality issues:

### Yahoo Finance

```python
# Known issues:
# - Adjusted close can be null
# - Volume sometimes zero
# - Splits not always adjusted correctly

def validate_yahoo_data(df):
    # Check for null adjusted close
    if "adj_close" in df.columns:
        null_adj = df.filter(df["adj_close"].is_null())
        if len(null_adj) > 0:
            logger.warning(f"Yahoo: {len(null_adj)} rows with null adj_close")

    # Verify split adjustments
    if "split_factor" in df.columns:
        splits = df.filter(df["split_factor"] != 1.0)
        if len(splits) > 0:
            logger.info(f"Yahoo: {len(splits)} stock splits detected")
```

### CoinGecko

```python
# Known issues:
# - Volume can be zero for low-liquidity coins
# - Prices sometimes lag by 1-2 minutes
# - Historical data can be revised

def validate_coingecko_data(df):
    # Check for zero volume (common for small coins)
    zero_vol_pct = (df["volume"] == 0).sum() / len(df) * 100
    if zero_vol_pct > 10:
        logger.warning(f"CoinGecko: {zero_vol_pct:.1f}% of data has zero volume")

    # Check for price lags (timestamp issues)
    current_time = datetime.now()
    latest_timestamp = df["timestamp"].max()
    lag_hours = (current_time - latest_timestamp).total_seconds() / 3600
    if lag_hours > 2:
        logger.warning(f"CoinGecko: Data lagged by {lag_hours:.1f} hours")
```

## Cross-Provider Validation

Compare data from multiple providers to catch issues:

```python
def cross_validate_providers(symbol, start, end):
    """Compare data from multiple providers."""
    # Fetch from 2 providers
    tiingo_data = TiingoProvider(api_key="key1").fetch_ohlcv(symbol, start, end)
    iex_data = IEXCloudProvider(api_key="key2").fetch_ohlcv(symbol, start, end)

    # Merge on timestamp
    merged = tiingo_data.join(iex_data, on="timestamp", suffix="_iex")

    # Compare close prices
    merged = merged.with_columns(
        ((merged["close"] - merged["close_iex"]).abs() / merged["close"] * 100)
        .alias("price_diff_pct")
    )

    # Flag large discrepancies (>1%)
    discrepancies = merged.filter(merged["price_diff_pct"] > 1.0)

    if len(discrepancies) > 0:
        logger.warning(f"Price discrepancies between providers:")
        print(discrepancies.select([
            "timestamp",
            "close",
            "close_iex",
            "price_diff_pct"
        ]))
        return False

    logger.info(f"✅ Providers agree on {symbol} prices")
    return True
```

## Production Validation Checklist

Use this checklist for production pipelines:

```python
class ProductionValidator:
    """Comprehensive validation for production data."""

    def validate_for_production(self, df, symbol, provider_name):
        """Run all validation checks."""
        errors = []
        warnings = []

        # 1. Schema validation
        if not self._validate_schema(df):
            errors.append("Schema validation failed")

        # 2. OHLCV invariants
        if not self._validate_invariants(df):
            errors.append("OHLCV invariant violations")

        # 3. Null checks
        null_count = df.null_count().sum()
        if null_count > 0:
            warnings.append(f"{null_count} null values found")

        # 4. Price spike detection
        spikes = self._detect_price_spikes(df)
        if len(spikes) > 0:
            warnings.append(f"{len(spikes)} price spikes detected")

        # 5. Volume checks
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > len(df) * 0.1:  # >10% zero volume
            warnings.append(f"{zero_vol} days with zero volume ({zero_vol/len(df)*100:.1f}%)")

        # 6. Timestamp continuity
        gaps = self._detect_timestamp_gaps(df)
        if len(gaps) > 0:
            warnings.append(f"{len(gaps)} timestamp gaps detected")

        # 7. Data freshness
        latest = df["timestamp"].max()
        age_hours = (datetime.now() - latest).total_seconds() / 3600
        if age_hours > 48:
            warnings.append(f"Data is {age_hours:.1f} hours old")

        # Log results
        if errors:
            logger.error(f"{symbol} validation FAILED: {errors}")
            return False

        if warnings:
            logger.warning(f"{symbol} validation warnings: {warnings}")

        logger.info(f"✅ {symbol} passed production validation")
        return True
```

## Monitoring Data Quality Over Time

Track data quality metrics:

```python
class DataQualityMonitor:
    """Monitor data quality trends."""

    def __init__(self):
        self.metrics = []

    def record_validation(self, symbol, provider, result):
        """Record validation results."""
        metrics = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "provider": provider,
            "valid": result.is_valid,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
        }
        self.metrics.append(metrics)

    def get_provider_quality_score(self, provider, days=30):
        """Calculate quality score for provider."""
        recent = [m for m in self.metrics
                  if m["provider"] == provider
                  and (datetime.now() - m["timestamp"]).days <= days]

        if not recent:
            return None

        valid_count = sum(1 for m in recent if m["valid"])
        total_count = len(recent)
        quality_score = valid_count / total_count * 100

        return {
            "provider": provider,
            "quality_score": quality_score,
            "total_validations": total_count,
            "valid": valid_count,
            "invalid": total_count - valid_count,
        }
```

## Summary

**Key Takeaways**:
1. ✅ **ML4T Data validates automatically** - OHLCV invariants checked on every fetch
2. ✅ **Multiple validation layers** - Schema, nulls, anomalies, provider-specific
3. ✅ **Fail loudly** - Never silently accept bad data
4. ✅ **Cross-provider validation** - Compare multiple sources
5. ✅ **Monitor quality** - Track data quality metrics over time

**Best Practices**:
- Always validate before using data in production
- Log validation results for debugging
- Alert on validation failures
- Compare providers when possible
- Monitor data freshness

**Next Steps**:
- [Tutorial 05: Multi-Provider Strategies](05_multi_provider.md)
- [Data Validation API Reference](../../src/ml4t-data/validation/)

---

**Previous Tutorial**: [03: Incremental Updates](03_incremental_updates.md)
**Next Tutorial**: [05: Multi-Provider Strategies](05_multi_provider.md)
