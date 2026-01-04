# Rate Limiting Best Practices

**Target Audience**: Developers building production systems
**Time to Read**: 15 minutes
**Prerequisites**: [Understanding OHLCV Data](01_understanding_ohlcv.md)

## Why Rate Limiting Matters

API providers limit how often you can call their endpoints to:
1. **Prevent abuse** - Stop malicious actors from overwhelming servers
2. **Ensure fairness** - Give all users equal access
3. **Control costs** - Server capacity costs money

**Violating rate limits** can result in:
- ⚠️ Temporary IP bans (1 hour to 24 hours)
- 🚫 Permanent API key revocation
- 💸 Unexpected charges (some providers charge overage fees)

## Understanding Rate Limit Types

### 1. Requests Per Minute (RPM)

Most common limit type:

```python
# Alpha Vantage: 5 calls per minute
provider = AlphaVantageProvider(api_key="key")

# First 5 calls succeed immediately
for i in range(5):
    data = provider.fetch_ohlcv("AAPL", start, end)  # ✅ Success

# 6th call blocks until 1 minute passes
data = provider.fetch_ohlcv("MSFT", start, end)  # ⏳ Waits ~60 seconds
```

### 2. Requests Per Day (RPD)

Daily quota that resets at midnight (UTC):

```python
# Tiingo: 1000 calls per day
provider = TiingoProvider(api_key="key")

# After 1000 calls today:
data = provider.fetch_ohlcv("AAPL", start, end)
# ❌ RateLimitError: Daily limit exceeded. Resets at 2024-01-15 00:00:00 UTC
```

### 3. Message Credits

Some providers (like IEX Cloud) use **credits** instead of call counts:

```python
# IEX Cloud: 50,000 message credits/month
# 1 OHLCV record = ~1 credit
# Fetching 30 days of AAPL = ~30 credits

provider = IEXCloudProvider(api_key="key")
data = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-30")  # Uses ~30 credits

# Check usage
usage = provider.get_usage_stats()
print(f"Used: {usage['credits_used']}/50000")
```

### 4. Concurrent Connections

Maximum simultaneous requests:

```python
# Some providers limit to 1-5 concurrent connections
# Exceeded limit → 429 Too Many Requests error

# BAD: Parallel requests without limiting
with ThreadPoolExecutor(max_workers=20) as executor:  # ❌ Too many!
    futures = [executor.submit(fetch, symbol) for symbol in symbols]
```

## ML4T Data's Built-In Rate Limiting

Every ML4T Data provider has **automatic rate limiting** built-in. You don't need to implement it yourself!

### How It Works

```python
from ml4t.data.providers import TiingoProvider

# Rate limiter configured based on provider's limits
provider = TiingoProvider(api_key="key")
# Tiingo allows 1000 calls/day → ML4T Data automatically enforces this

# Make requests normally - ML4T Data handles rate limiting
for symbol in ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]:
    data = provider.fetch_ohlcv(symbol, "2024-01-01", "2024-01-31")
    # ✅ ML4T Data automatically paces requests to respect rate limits
```

**Behind the scenes**:
1. ML4T Data tracks calls across **all instances** (global rate limiter)
2. Automatically **blocks** when limit reached
3. **Resumes** when rate limit window resets
4. **Logs** rate limit events for debugging

### Global Rate Limiting

ML4T Data uses **global rate limiters** shared across all provider instances:

```python
# Even with multiple instances, rate limits are shared
provider1 = TiingoProvider(api_key="key")
provider2 = TiingoProvider(api_key="key")  # Same API key

# First instance makes 500 calls
for i in range(500):
    provider1.fetch_ohlcv(f"SYMBOL{i}", start, end)

# Second instance respects same limit (500 remaining out of 1000)
for i in range(600):  # Tries to make 600 calls
    provider2.fetch_ohlcv(f"OTHER{i}", start, end)
    # After 500 calls, ML4T Data blocks until daily reset
```

**Why global?** Prevents accidental limit violations when running multiple scripts/processes.

## Rate Limit Configuration

### Default Limits

Each provider has sensible defaults:

```python
# Providers have DEFAULT_RATE_LIMIT class variable
TiingoProvider.DEFAULT_RATE_LIMIT  # (1000, 86400.0) = 1000 per day
AlphaVantageProvider.DEFAULT_RATE_LIMIT  # (5, 60.0) = 5 per minute
CoinGeckoProvider.DEFAULT_RATE_LIMIT  # (50, 60.0) = 50 per minute
```

### Custom Limits (Advanced)

Override limits for testing or paid tiers:

```python
# Example: You upgraded Tiingo to paid tier (20,000/hour)
provider = TiingoProvider(
    api_key="key",
    rate_limit=(20000, 3600.0)  # (max_calls, period_seconds)
)

# Or set very conservative limit for testing
provider = AlphaVantageProvider(
    api_key="key",
    rate_limit=(3, 60.0)  # Only 3/min instead of 5/min (safer)
)
```

## Best Practices

### 1. Use Incremental Updates

**Don't refetch all data every time!**

```python
# ❌ BAD: Refetch entire history daily
for symbol in symbols:
    data = provider.fetch_ohlcv(symbol, "2020-01-01", "2024-12-31")
    # 1000+ days × 100 symbols = 100,000+ API calls!

# ✅ GOOD: Incremental updates
from ml4t.data.providers import TiingoUpdater
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.backend import StorageConfig

storage = HiveStorage(StorageConfig(base_path="./data"))
updater = TiingoUpdater(provider, storage)

for symbol in symbols:
    updater.update_symbol(symbol, incremental=True)
    # Only fetches NEW data since last update
    # 1 day × 100 symbols = 100 API calls (1000x reduction!)
```

### 2. Batch Requests When Possible

Some providers support batch requests:

```python
# ❌ BAD: Individual requests
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    quote = provider.fetch_quote(symbol)  # 3 API calls

# ✅ GOOD: Batch request (if provider supports it)
quotes = provider.fetch_quote(["AAPL", "MSFT", "GOOGL"])  # 1 API call
```

**Providers with batch support**:
- **Twelve Data**: `fetch_quote(["AAPL", "MSFT", ...])` (up to 120 symbols)
- **IEX Cloud**: Batch endpoints available

### 3. Cache Aggressively

Don't fetch the same data multiple times in one session:

```python
from functools import lru_cache
from datetime import datetime

@lru_cache(maxsize=1000)
def get_ohlcv_cached(symbol, start, end):
    """Cache OHLCV data for the session."""
    return provider.fetch_ohlcv(symbol, start, end)

# First call: Fetches from API
data1 = get_ohlcv_cached("AAPL", "2024-01-01", "2024-12-31")  # API call

# Second call: Returns cached result
data2 = get_ohlcv_cached("AAPL", "2024-01-01", "2024-12-31")  # Instant, no API call
```

### 4. Schedule During Off-Peak Hours

Many rate limits reset at specific times:

```python
# Daily limits often reset at midnight UTC
# Run data updates at 00:05 UTC to get fresh quota

import schedule
import time

def daily_update():
    """Run at 00:05 UTC every day."""
    for symbol in symbols:
        updater.update_symbol(symbol, incremental=True)

schedule.every().day.at("00:05").do(daily_update)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 5. Prioritize Important Symbols

If you hit rate limits, fetch high-priority symbols first:

```python
# Prioritize by market cap or portfolio weight
symbols_priority = [
    "AAPL",  # High priority
    "MSFT",
    "GOOGL",
    # ... 50 more ...
    "SMALLCAP123",  # Low priority
]

# Fetch in order - high-priority symbols get fetched first
for symbol in symbols_priority:
    try:
        updater.update_symbol(symbol, incremental=True)
    except RateLimitError:
        logger.warning(f"Rate limit hit at symbol: {symbol}")
        break  # Stop here, resume tomorrow
```

## Monitoring Rate Limit Usage

### 1. Check Provider Dashboard

Most providers have usage dashboards:
- **Tiingo**: https://api.tiingo.com/account/usage
- **IEX Cloud**: https://iexcloud.io/console/usage
- **Alpha Vantage**: https://www.alphavantage.co/support/#support

### 2. Log Rate Limit Events

ML4T Data automatically logs rate limit events:

```python
import logging
logging.basicConfig(level=logging.INFO)

# ML4T Data will log:
# INFO: Rate limit: 950/1000 calls used (95%)
# WARNING: Rate limit: 990/1000 calls used (99%)
# WARNING: Rate limit exceeded, blocking until reset at 2024-01-15 00:00:00 UTC
```

### 3. Implement Custom Tracking

```python
import time

class RateLimitTracker:
    def __init__(self, provider, max_calls_per_day):
        self.provider = provider
        self.max_calls = max_calls_per_day
        self.calls_made = 0
        self.reset_time = time.time() + 86400

    def fetch_with_tracking(self, symbol, start, end):
        if time.time() > self.reset_time:
            self.calls_made = 0
            self.reset_time = time.time() + 86400

        if self.calls_made >= self.max_calls:
            raise RateLimitError(f"Daily limit reached: {self.calls_made}/{self.max_calls}")

        data = self.provider.fetch_ohlcv(symbol, start, end)
        self.calls_made += 1
        return data

tracker = RateLimitTracker(provider, max_calls_per_day=1000)
```

## Handling Rate Limit Errors

### Graceful Degradation

```python
from ml4t.data.core.exceptions import RateLimitError
import time

def fetch_with_retry(provider, symbol, start, end, max_retries=3):
    """Retry with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return provider.fetch_ohlcv(symbol, start, end)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 60  # 1min, 2min, 4min
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise  # Give up after max_retries
```

### Circuit Breaker Pattern

ML4T Data has built-in circuit breakers:

```python
# After 5 consecutive failures, circuit breaker opens
# Provider stops making requests for 5 minutes
# After 5 minutes, tries one request (half-open state)
# If successful, circuit closes and normal operation resumes

try:
    data = provider.fetch_ohlcv("AAPL", start, end)
except CircuitBreakerOpenError:
    logger.error("Circuit breaker open - provider unavailable")
    # Fall back to cached data or secondary provider
```

## Provider-Specific Tips

### Alpha Vantage (5/min, 25/day)

**Extremely restrictive!** Only use for research or low-volume updates.

```python
# With 25/day limit, you can update 25 symbols daily
# Or 5 symbols × 5 frequencies (daily, weekly, monthly, etc.)

# Strategy: Update top 25 portfolio holdings daily
# Rotate through full universe over multiple days
symbols_day_1 = ["AAPL", "MSFT", "GOOGL", ...]  # Top 25
symbols_day_2 = ["NFLX", "TSLA", "NVDA", ...]   # Next 25
# etc.
```

### Tiingo (1000/day)

**Generous free tier** - Good for daily portfolio updates.

```python
# 1000 calls/day allows:
# - 1000 symbols × daily updates
# - 333 symbols × 3 frequencies (daily, weekly, monthly)
# - 200 symbols × 5 data types (OHLCV, news, fundamentals, etc.)

# Strategy: Update full portfolio daily + fetch news for top holdings
for symbol in portfolio_symbols:  # 200 symbols
    updater.update_symbol(symbol, incremental=True)  # 200 calls

for symbol in top_holdings[:100]:  # 100 symbols
    news = provider.fetch_news(symbol)  # 100 calls

# Total: 300/1000 calls used (30%)
```

### IEX Cloud (50K messages/month)

**Message-based pricing** - Count records, not requests.

```python
# 50,000 messages = ~50,000 OHLCV records
# Fetching 30 days × 100 symbols = 3,000 records = 3,000 messages

# Strategy: Monthly updates instead of daily
# Or focus on smaller symbol universe

# Conservative approach:
# 50K messages ÷ 30 days = ~1,666 messages/day
# = 1,666 OHLCV records/day
# = 55 symbols × 30-day history per day
```

### EODHD (500/day free, unlimited paid)

**Most generous free tier** for global stocks.

```python
# 500 calls/day allows:
# - 500 symbols × daily updates (global exchanges!)
# - 166 symbols × 3 frequencies
# - 100 symbols × 5 exchanges (if multi-exchange strategy)

# Strategy: Use free tier for testing, upgrade for production
# €19.99/month for unlimited is excellent value
```

## Troubleshooting

### "RateLimitError: Too many requests"

**Cause**: Exceeded provider's rate limit
**Solution**:
1. Check current usage in provider dashboard
2. Implement incremental updates (reduce API calls 100x)
3. Space out requests over multiple days
4. Consider upgrading to paid tier

### "HTTP 429: Too Many Requests"

**Cause**: Provider's server rejected request due to rate limiting
**Solution**:
- ML4T Data should handle this automatically
- If you see this, report as bug (ML4T Data rate limiter should prevent it)

### "Blocked for 24 hours"

**Cause**: Severely exceeded rate limits (usually manual curl/script)
**Solution**:
- Wait for ban to expire
- Use ML4T Data's built-in rate limiting going forward
- Contact provider support if ban doesn't lift

## Summary

**Key Takeaways**:
1. ✅ ML4T Data handles rate limiting automatically - just use the providers normally
2. ✅ Use incremental updates to reduce API calls by 100-1000x
3. ✅ Global rate limiters prevent accidental violations across processes
4. ✅ Monitor usage and prioritize important symbols
5. ✅ Free tiers are generous for research, but production needs paid plans

**Next Steps**:
- [Tutorial 03: Incremental Updates](03_incremental_updates.md) - Learn to minimize API calls
- [Tutorial 04: Data Quality Validation](04_data_quality.md) - Ensure data integrity
- [Provider Selection Guide](../provider-selection-guide.md) - Choose the right rate limits

---

**Previous Tutorial**: [01: Understanding OHLCV Data](01_understanding_ohlcv.md)
**Next Tutorial**: [03: Incremental Updates](03_incremental_updates.md)
