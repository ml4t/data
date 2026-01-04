# Yahoo Finance Bulk Download Analysis

**Date**: 2025-11-27
**Status**: Implementation Complete ✅

## Executive Summary

Testing revealed the current ml4t-data YahooFinanceProvider is **22x slower** than batch download approaches for bulk data acquisition. This document analyzes the options and recommends enhancements.

## Test Results (90 symbols, 5 years of data)

| Method | Time | ms/symbol | Success Rate | Notes |
|--------|------|-----------|--------------|-------|
| **batch_all** | 3.1s | 35ms | 100% | Single `yf.download()` with all symbols |
| **chunked_25** | 3.7s | 41ms | 100% | 25 symbols/chunk, 0.5s delay |
| **chunked_50** | 3.8s | 42ms | 100% | 50 symbols/chunk, 1.0s delay |
| **ml4t_provider** | 67.5s | 750ms | 97.8% | Current single-symbol approach |

**Key Finding**: Batch downloads are dramatically faster because:
1. Single HTTP connection for multiple symbols
2. Yahoo Finance API optimized for multi-ticker requests
3. `yf.download()` uses internal threading (`threads=True`)

## Current Implementations Compared

### ml4t-data YahooFinanceProvider (Current)

```python
# Single-symbol download per call
def fetch_ohlcv(self, symbol, start, end, frequency):
    df = yf.download(symbol, start=start, end=end, ...)
    return self._convert_to_polars(df, symbol)
```

**Pros**:
- Simple, per-symbol interface
- OHLC validation built-in
- Polars output for downstream speed
- Integrates with incremental update system

**Cons**:
- 22x slower for bulk downloads
- Inefficient for initial data acquisition
- Generates many HTTP requests

### ml3t Notebook Approach

```python
# Chunked batch downloads
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND)),
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),
)

for chunk in chunks(symbols, 100):
    prices = yf.download(chunk, start="2000-01-01", end=today, ...)
```

**Pros**:
- Fast bulk downloads
- Rate limiting prevents blocks
- Caching reduces duplicate requests

**Cons**:
- CachedLimiterSession doesn't work with `yf.download()` (only Ticker methods)
- Returns pandas, not Polars
- No OHLC validation
- No incremental update integration

## Yahoo Finance Rate Limiting (Nov 2024)

Recent changes from [GitHub Issue #2128](https://github.com/ranaroussi/yfinance/issues/2128):

- **Threshold**: ~950 symbols before rate limiting kicks in
- **Recommendation**: 2 requests/second for sequential, or chunked batches
- **yfinance 0.2.58+**: TLS fingerprinting fixes help avoid blocks
- **CachedLimiterSession**: Useful for `.info()` metadata calls, NOT for `yf.download()`

## Recommendations

### 1. Add Batch Download Method to YahooFinanceProvider

```python
def fetch_batch_ohlcv(
    self,
    symbols: list[str],
    start: str,
    end: str,
    frequency: str = "daily",
    chunk_size: int = 50,
    delay_seconds: float = 1.0,
) -> pl.DataFrame:
    """
    Fetch OHLCV data for multiple symbols efficiently.

    Uses chunked batch downloads for speed while maintaining
    reliability through rate limiting.
    """
    all_data = []

    for chunk in chunks(symbols, chunk_size):
        df_pandas = yf.download(
            chunk,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=True,
        )

        # Convert to long format with symbol column
        df_polars = self._convert_batch_to_polars(df_pandas)
        all_data.append(df_polars)

        time.sleep(delay_seconds)

    return pl.concat(all_data)
```

### 2. Keep Single-Symbol Method for Updates

The current `fetch_ohlcv()` method is appropriate for:
- Incremental daily updates
- Small symbol counts (<10)
- Integration with UpdateManager

### 3. Add Initial Data Acquisition Script

Create `scripts/download_yahoo_universe.py`:
```python
# For book examples and initial setup
provider = YahooFinanceProvider()
df = provider.fetch_batch_ohlcv(
    symbols=sp500_symbols,
    start="2000-01-01",
    end=today,
    chunk_size=50,
)
# Save to Hive storage
storage.write(df)
```

### 4. Document Both Approaches in Book

**Notebook (Educational)**:
- Explain batch vs sequential trade-offs
- Show chunking pattern
- Discuss rate limiting

**Library (Production)**:
- Use `fetch_batch_ohlcv()` for initial acquisition
- Use `fetch_ohlcv()` + UpdateManager for daily updates

## Implementation Priority

1. **High**: Add `fetch_batch_ohlcv()` method to YahooFinanceProvider
2. **High**: Create example notebook showing bulk download patterns
3. **Medium**: Add CLI command `ml4t-data download --provider yahoo --symbols-file sp500.txt`
4. **Low**: CachedLimiterSession for metadata (`.info`) downloads

## Sources

- [yfinance Rate Limiting Issue #2128](https://github.com/ranaroussi/yfinance/issues/2128)
- [yfinance Rate Limiting Issue #2125](https://github.com/ranaroussi/yfinance/issues/2125)
- [Sling Academy - Rate Limiting Best Practices](https://www.slingacademy.com/article/rate-limiting-and-api-best-practices-for-yfinance/)
- [yfinance Smarter Scraping docs](https://github.com/ranaroussi/yfinance)

## Implementation Results (2025-11-27)

### fetch_batch_ohlcv() Method Added

Added to `src/ml4t/data/providers/yahoo.py`:

```python
def fetch_batch_ohlcv(
    self,
    symbols: list[str],
    start: str,
    end: str,
    frequency: str = "daily",
    chunk_size: int = 50,
    delay_seconds: float = 1.0,
) -> pl.DataFrame:
    """Fetch OHLCV data for multiple symbols efficiently."""
```

### Verified Test Results

| Test | Result |
|------|--------|
| Basic functionality (5 symbols) | ✅ PASSED |
| Speedup (20 symbols) | ✅ 6.5x faster |
| Scale test (100 symbols) | ✅ 100% success rate |
| Data quality | ✅ No violations |

**Performance**: 100 symbols in 4.1s (41ms/symbol) vs 285ms/symbol with single-symbol method

### Usage

```python
from ml4t.data.providers.yahoo import YahooFinanceProvider

provider = YahooFinanceProvider()

# Bulk download (fast)
df = provider.fetch_batch_ohlcv(
    symbols=["AAPL", "MSFT", "GOOGL", ...],
    start="2020-01-01",
    end="2024-01-01",
    chunk_size=50,
    delay_seconds=1.0,
)

# Single symbol (for updates)
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-15", "daily")
```

## Test Scripts

Created during this analysis:
- `scripts/test_bulk_download_approaches.py` - Initial comparison
- `scripts/test_bulk_scale.py` - 50 symbol scale test
- `scripts/test_bulk_final.py` - Comprehensive 90 symbol test
- `scripts/test_batch_ohlcv.py` - Implementation verification
