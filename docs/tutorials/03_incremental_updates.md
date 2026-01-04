# Incremental Updates: The Smart Way to Update Data

**Target Audience**: Developers building production data pipelines
**Time to Read**: 20 minutes
**Prerequisites**: [Understanding OHLCV](01_understanding_ohlcv.md), [Rate Limiting](02_rate_limiting.md)

## The Problem: Naive Data Updates

Let's say you want to maintain a database of stock prices for 100 symbols. The naive approach:

```python
# ❌ NAIVE APPROACH - Don't do this!
for symbol in symbols:
    # Fetch entire history every day
    data = provider.fetch_ohlcv(symbol, "2020-01-01", "2024-12-31")
    storage.write(data, symbol, provider_name)

# Problems:
# 1. Wastes 1000+ API calls per symbol (refetching old data)
# 2. Slow (download gigabytes of data you already have)
# 3. Expensive (hits rate limits quickly)
# 4. Fragile (fails if provider rate-limits you)
```

**The math**:
- 100 symbols
- 1000 days of history each
- = 100,000 OHLCV records fetched
- = 100,000+ API calls per day! 🔥

Most free tiers allow 500-1000 calls/day. You'd burn through your quota in seconds.

## The Solution: Incremental Updates

**Incremental updates** only fetch NEW data since your last update:

```python
# ✅ SMART APPROACH - Incremental updates
from ml4t.data.providers import TiingoProvider, TiingoUpdater
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.backend import StorageConfig

# Setup storage and updater
storage = HiveStorage(StorageConfig(base_path="./data"))
provider = TiingoProvider(api_key="your_key")
updater = TiingoUpdater(provider, storage)

# Update symbols incrementally
for symbol in symbols:
    result = updater.update_symbol(symbol, incremental=True)
    print(f"{symbol}: {result['records_added']} new records")

# Benefits:
# 1. Only fetches 1 day of new data per symbol (100 API calls total)
# 2. Fast (downloads only what's new)
# 3. Efficient (respects rate limits)
# 4. Robust (handles gaps and missing data automatically)
```

**The math**:
- 100 symbols
- 1 day of new data each
- = 100 OHLCV records fetched
- = 100 API calls per day ✅
- **1000x reduction in API calls!**

## How Incremental Updates Work

### Step 1: Check What You Have

```python
# Updater checks storage for existing data
metadata = storage.get_metadata(symbol, provider_name)

if metadata:
    last_date = metadata['end_date']  # e.g., "2024-01-14"
    print(f"Last update: {last_date}")
else:
    print("No existing data - will fetch default history")
```

### Step 2: Calculate Gap

```python
# Calculate what data is missing
today = datetime.now().strftime("%Y-%m-%d")

if last_date:
    start_date = last_date  # Start from last known date
    gap_days = (today - last_date).days
    print(f"Gap: {gap_days} days")
else:
    start_date = default_start  # e.g., 90 days ago for crypto
    print("First fetch - getting default history")

end_date = today
```

### Step 3: Fetch Only Gap

```python
# Only fetch the missing data
if gap_days > 0:
    new_data = provider.fetch_ohlcv(symbol, start_date, end_date)
    print(f"Fetched {len(new_data)} new records")
else:
    print("Already up to date!")
    return
```

### Step 4: Merge and Store

```python
# Append new data to existing data
# ML4T Data handles deduplication automatically
storage.write(new_data, symbol, provider_name)
print("Stored successfully")
```

## The ProviderUpdater Pattern

Every ML4T Data provider has a corresponding `ProviderUpdater` class:

| Provider | Updater Class |
|----------|---------------|
| TiingoProvider | TiingoUpdater |
| CoinGeckoProvider | CoinGeckoUpdater |
| IEXCloudProvider | IEXCloudUpdater |
| EODHDProvider | EODHDUpdater |
| AlphaVantageProvider | AlphaVantageUpdater |
| FinnhubProvider | FinnhubUpdater |

All updaters share the same interface:

```python
class ProviderUpdater:
    def __init__(self, provider, storage):
        self.provider = provider
        self.storage = storage

    def update_symbol(
        self,
        symbol: str,
        start_time: str | None = None,
        end_time: str | None = None,
        frequency: str = "daily",
        incremental: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """Update symbol data incrementally or forcefully."""
        ...
```

## Basic Usage

### First Update (Bootstrap)

```python
from ml4t.data.providers import TiingoProvider, TiingoUpdater
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.backend import StorageConfig

# Setup
storage = HiveStorage(StorageConfig(base_path="./data"))
provider = TiingoProvider(api_key="your_key")
updater = TiingoUpdater(provider, storage)

# First update: Downloads default history (30 days for stocks)
result = updater.update_symbol("AAPL", incremental=True)

print(result)
# {
#     'success': True,
#     'symbol': 'AAPL',
#     'records_fetched': 30,
#     'records_added': 30,
#     'start_date': '2024-12-15',
#     'end_date': '2025-01-14',
#     'message': 'Successfully updated'
# }
```

### Daily Updates

```python
# Second update (next day): Only fetches NEW data
result = updater.update_symbol("AAPL", incremental=True)

print(result)
# {
#     'success': True,
#     'symbol': 'AAPL',
#     'records_fetched': 1,
#     'records_added': 1,
#     'start_date': '2025-01-15',
#     'end_date': '2025-01-15',
#     'message': 'Added 1 new record'
# }
```

### Already Up to Date

```python
# Same day, run again: Skips fetch
result = updater.update_symbol("AAPL", incremental=True)

print(result)
# {
#     'success': True,
#     'symbol': 'AAPL',
#     'skip_reason': 'already_up_to_date',
#     'last_date': '2025-01-15',
#     'message': 'Data already current'
# }
```

## Advanced Usage

### Custom Date Ranges

```python
# Override default range for bootstrap
result = updater.update_symbol(
    "AAPL",
    start_time="2020-01-01",  # Get 5 years of history
    end_time="2024-12-31",
    incremental=False,  # Force fetch (ignore existing data)
)
```

### Dry Run Mode

```python
# Preview what would be fetched without actually storing
result = updater.update_symbol("AAPL", incremental=True, dry_run=True)

print(result)
# {
#     'success': True,
#     'symbol': 'AAPL',
#     'records_fetched': 5,
#     'dry_run': True,
#     'message': 'Would add 5 new records (dry run - not stored)'
# }
```

### Multiple Frequencies

```python
# Update daily, weekly, and monthly data
for frequency in ["daily", "weekly", "monthly"]:
    result = updater.update_symbol(
        "AAPL",
        frequency=frequency,
        incremental=True
    )
    print(f"{frequency}: {result['records_added']} records")
```

## Handling Gaps

### The Gap Problem

What if you miss a few days of updates?

```
Stored data:  [Jan 1] [Jan 2] [Jan 3] ... [Jan 10]
                                              ↑ Last update
Today:        Jan 15
                                         Gap!
Missing:      [Jan 11] [Jan 12] [Jan 13] [Jan 14]
```

**Solution**: Incremental updates automatically detect and fill gaps!

```python
# Even after 5 days without updates, incremental mode works
result = updater.update_symbol("AAPL", incremental=True)

print(result)
# {
#     'success': True,
#     'records_fetched': 5,  # Fetched Jan 11-15
#     'records_added': 5,
#     'gap_detected': True,
#     'gap_days': 5,
#     'message': 'Filled 5-day gap'
# }
```

### Handling Provider Outages

If a provider is down, updater gracefully handles it:

```python
try:
    result = updater.update_symbol("AAPL", incremental=True)
except NetworkError as e:
    logger.error(f"Provider unavailable: {e}")
    # Try again later - incremental mode will catch up
```

## Batch Updates

### Update Multiple Symbols

```python
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

for symbol in symbols:
    try:
        result = updater.update_symbol(symbol, incremental=True)
        logger.info(f"{symbol}: +{result['records_added']} records")
    except Exception as e:
        logger.error(f"{symbol}: Failed - {e}")
        continue  # Skip to next symbol
```

### Parallel Updates (Careful!)

```python
from concurrent.futures import ThreadPoolExecutor
import threading

# Use thread-safe storage
storage = HiveStorage(StorageConfig(base_path="./data"))

def update_symbol_safe(symbol):
    """Thread-safe update function."""
    # Each thread gets its own updater instance
    thread_updater = TiingoUpdater(provider, storage)
    return thread_updater.update_symbol(symbol, incremental=True)

# Update in parallel (max 5 concurrent to respect rate limits)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(update_symbol_safe, s): s for s in symbols}

    for future in futures:
        symbol = futures[future]
        try:
            result = future.result()
            print(f"{symbol}: +{result['records_added']}")
        except Exception as e:
            print(f"{symbol}: ERROR - {e}")
```

**⚠️ Warning**: Parallel updates can hit rate limits faster. Use conservatively.

## Production Patterns

### Pattern 1: Scheduled Daily Updates

```python
import schedule
import time
from datetime import datetime

def daily_update_job():
    """Run at market close + 30 minutes (16:30 ET)."""
    logger.info("Starting daily update...")

    updated = 0
    failed = 0

    for symbol in portfolio_symbols:
        try:
            result = updater.update_symbol(symbol, incremental=True)
            if result['success']:
                updated += 1
                logger.info(f"{symbol}: +{result['records_added']} records")
        except Exception as e:
            failed += 1
            logger.error(f"{symbol}: {e}")

    logger.info(f"Update complete: {updated} success, {failed} failed")

# Schedule for 16:30 ET (21:30 UTC) every weekday
schedule.every().monday.at("21:30").do(daily_update_job)
schedule.every().tuesday.at("21:30").do(daily_update_job)
schedule.every().wednesday.at("21:30").do(daily_update_job)
schedule.every().thursday.at("21:30").do(daily_update_job)
schedule.every().friday.at("21:30").do(daily_update_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Pattern 2: Catch-Up After Downtime

```python
def catch_up_all_symbols():
    """Catch up all symbols after downtime."""
    logger.info("Starting catch-up update...")

    for symbol in all_symbols:
        result = updater.update_symbol(symbol, incremental=True)

        if result.get('gap_detected'):
            logger.warning(
                f"{symbol}: Filled {result['gap_days']}-day gap "
                f"({result['records_added']} records)"
            )
        elif result.get('skip_reason') == 'already_up_to_date':
            logger.info(f"{symbol}: Already current")
        else:
            logger.info(f"{symbol}: +{result['records_added']} records")

catch_up_all_symbols()
```

### Pattern 3: Prioritized Updates with Rate Limit Handling

```python
from ml4t.data.core.exceptions import RateLimitError

def prioritized_update(symbols_priority_order):
    """Update high-priority symbols first, stop at rate limit."""
    updated_symbols = []

    for symbol in symbols_priority_order:
        try:
            result = updater.update_symbol(symbol, incremental=True)
            updated_symbols.append(symbol)
            logger.info(f"{symbol}: Updated ({len(updated_symbols)}/{len(symbols_priority_order)})")

        except RateLimitError as e:
            logger.warning(f"Rate limit reached at symbol {symbol}")
            logger.info(f"Successfully updated {len(updated_symbols)} symbols")
            logger.info(f"Remaining symbols will be updated in next run")
            break

        except Exception as e:
            logger.error(f"{symbol}: Failed - {e}")
            continue

    return updated_symbols

# Run with priority order (high market cap first)
symbols_by_market_cap = ["AAPL", "MSFT", "GOOGL", ...]
updated = prioritized_update(symbols_by_market_cap)
```

## Monitoring and Debugging

### Check Update Status

```python
# Check when symbol was last updated
metadata = storage.get_metadata("AAPL", "tiingo")

print(f"Last update: {metadata['end_date']}")
print(f"Total records: {metadata['record_count']}")
print(f"Date range: {metadata['start_date']} to {metadata['end_date']}")
```

### Validate Data Freshness

```python
from datetime import datetime, timedelta

def check_data_freshness(symbol, max_age_days=2):
    """Alert if data is stale."""
    metadata = storage.get_metadata(symbol, "tiingo")

    if not metadata:
        logger.warning(f"{symbol}: No data found!")
        return False

    last_date = datetime.strptime(metadata['end_date'], "%Y-%m-%d")
    age_days = (datetime.now() - last_date).days

    if age_days > max_age_days:
        logger.warning(f"{symbol}: Data is {age_days} days old!")
        return False

    logger.info(f"{symbol}: Fresh (updated {age_days} days ago)")
    return True

# Check all symbols
stale_symbols = []
for symbol in portfolio_symbols:
    if not check_data_freshness(symbol, max_age_days=2):
        stale_symbols.append(symbol)

if stale_symbols:
    logger.error(f"Stale data for {len(stale_symbols)} symbols: {stale_symbols}")
```

## Common Pitfalls

### Pitfall 1: Not Using Incremental Updates

```python
# ❌ BAD: Refetching everything
data = provider.fetch_ohlcv("AAPL", "2020-01-01", "2024-12-31")
# Wastes 1000+ API calls every time

# ✅ GOOD: Incremental updates
result = updater.update_symbol("AAPL", incremental=True)
# Only 1 API call per day
```

### Pitfall 2: Ignoring Return Values

```python
# ❌ BAD: Not checking result
updater.update_symbol("AAPL", incremental=True)
# Did it work? Did it skip? Did it fill a gap? Who knows!

# ✅ GOOD: Check result and log
result = updater.update_symbol("AAPL", incremental=True)
if result['success']:
    logger.info(f"Updated: +{result['records_added']} records")
else:
    logger.error(f"Failed: {result.get('error')}")
```

### Pitfall 3: Running Too Frequently

```python
# ❌ BAD: Updating every minute
while True:
    updater.update_symbol("AAPL", incremental=True)
    time.sleep(60)
# Most markets update once per day! Wastes API calls.

# ✅ GOOD: Update once per day after market close
schedule.every().day.at("16:30").do(update_all_symbols)
```

### Pitfall 4: Not Handling Provider-Specific Limits

```python
# ❌ BAD: Updating 1000 symbols with Alpha Vantage (25/day limit)
for symbol in symbols[:1000]:  # Won't work!
    updater.update_symbol(symbol, incremental=True)

# ✅ GOOD: Respect provider limits
daily_limit = 25  # Alpha Vantage free tier
for symbol in symbols[:daily_limit]:
    updater.update_symbol(symbol, incremental=True)
```

## Performance Optimization

### Benchmark: Naive vs. Incremental

```python
import time

# Naive approach
start = time.time()
for symbol in symbols[:100]:
    data = provider.fetch_ohlcv(symbol, "2020-01-01", "2024-12-31")
    storage.write(data, symbol, "provider")
naive_time = time.time() - start
print(f"Naive: {naive_time:.1f}s")
# Naive: 342.7s (5.7 minutes)

# Incremental approach
start = time.time()
for symbol in symbols[:100]:
    updater.update_symbol(symbol, incremental=True)
incremental_time = time.time() - start
print(f"Incremental: {incremental_time:.1f}s")
# Incremental: 34.2s (34 seconds)

print(f"Speedup: {naive_time / incremental_time:.1f}x faster")
# Speedup: 10.0x faster
```

## Summary

**Key Takeaways**:
1. ✅ **Always use incremental updates** - 100-1000x more efficient
2. ✅ **ProviderUpdater pattern** - Consistent across all providers
3. ✅ **Automatic gap filling** - Handles missing days gracefully
4. ✅ **Dry run mode** - Test before storing
5. ✅ **Check return values** - Monitor success and debug failures

**When to Use Incremental**:
- Daily/weekly portfolio updates
- Production data pipelines
- Any scenario where you update the same symbols repeatedly

**When to Force Full Fetch**:
- First-time bootstrap
- Data quality issues (provider had bad data, now fixed)
- Switching providers
- Backfilling historical data

**Next Steps**:
- [Tutorial 04: Data Quality Validation](04_data_quality.md)
- [Tutorial 05: Multi-Provider Strategies](05_multi_provider.md)

---

**Previous Tutorial**: [02: Rate Limiting Best Practices](02_rate_limiting.md)
**Next Tutorial**: [04: Data Quality Validation](04_data_quality.md)
