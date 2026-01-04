# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bulk Market Data Download with Yahoo Finance
#
# This notebook demonstrates efficient bulk data acquisition for quantitative finance research.
# We'll cover:
#
# 1. **Why bulk downloads matter** - Performance comparison
# 2. **The batch download approach** - How it works under the hood
# 3. **Rate limiting considerations** - Avoiding blocks from Yahoo Finance
# 4. **Data quality validation** - Ensuring clean data for backtesting
# 5. **Integration with ml4t-data** - Using the library's batch download feature
#
# > **Book Reference**: This notebook accompanies Chapter 4 of *Machine Learning for Trading*,
# > which covers data acquisition and storage strategies for quantitative research.

# %% [markdown]
# ## The Challenge: Downloading Data at Scale
#
# When building ML trading strategies, you need historical data for:
# - **Universe of stocks** (S&P 500 = 500 symbols)
# - **Long history** (10-20 years for robust backtesting)
# - **Multiple frequencies** (daily, hourly, minute)
#
# A naive approach downloads one symbol at a time:
#
# ```python
# # Slow approach - DON'T DO THIS
# for symbol in sp500_symbols:
#     data = yf.download(symbol, start="2010-01-01", end="2024-01-01")
#     # Takes ~300ms per symbol = 2.5 minutes for S&P 500
# ```
#
# This is **inefficient** because:
# 1. Each call creates a new HTTP connection
# 2. Yahoo Finance API is optimized for multi-ticker requests
# 3. You're not leveraging parallelism

# %% [markdown]
# ## Setup

# %%
import time
from datetime import datetime

import polars as pl
import yfinance as yf

# ml4t-data library
from ml4t.data.providers.yahoo import YahooFinanceProvider

# %%
# Test symbols - a mix of sectors for realistic testing
TEST_SYMBOLS = [
    # Tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    # Finance
    "JPM",
    "V",
    "MA",
    "BAC",
    "GS",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "ABT",
    # Consumer
    "WMT",
    "PG",
    "KO",
    "HD",
]

START_DATE = "2020-01-01"
END_DATE = "2025-11-27"

print(f"Test universe: {len(TEST_SYMBOLS)} symbols")
print(f"Date range: {START_DATE} to {END_DATE}")

# %% [markdown]
# ## Part 1: Why Batch Downloads Are Faster
#
# Let's compare the two approaches with actual timing.

# %% [markdown]
# ### Approach A: Single-Symbol Downloads (Slow)


# %%
def download_single_symbol(symbols: list[str], start: str, end: str) -> tuple[list, float]:
    """Download symbols one at a time - the slow way."""
    results = []
    start_time = time.perf_counter()

    for symbol in symbols:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if not df.empty:
            df["symbol"] = symbol
            results.append(df)

    elapsed = time.perf_counter() - start_time
    return results, elapsed


# %%
# Time the single-symbol approach
print("Downloading one symbol at a time...")
single_results, single_time = download_single_symbol(TEST_SYMBOLS, START_DATE, END_DATE)

print("\nSingle-symbol approach:")
print(f"  Time: {single_time:.2f}s")
print(f"  Per symbol: {single_time / len(TEST_SYMBOLS) * 1000:.0f}ms")
print(f"  Symbols downloaded: {len(single_results)}")

# %% [markdown]
# ### Approach B: Batch Download (Fast)


# %%
def download_batch(symbols: list[str], start: str, end: str) -> tuple[any, float]:
    """Download all symbols in one request - the fast way."""
    start_time = time.perf_counter()

    df = yf.download(
        symbols,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=True,  # Enable multi-threading
    )

    elapsed = time.perf_counter() - start_time
    return df, elapsed


# %%
# Time the batch approach
print("Downloading all symbols in one batch...")
batch_result, batch_time = download_batch(TEST_SYMBOLS, START_DATE, END_DATE)

print("\nBatch approach:")
print(f"  Time: {batch_time:.2f}s")
print(f"  Per symbol: {batch_time / len(TEST_SYMBOLS) * 1000:.0f}ms")
print(f"  DataFrame shape: {batch_result.shape}")

# %%
# Calculate speedup
speedup = single_time / batch_time
print(f"\n{'=' * 50}")
print(f"SPEEDUP: {speedup:.1f}x faster with batch download!")
print(f"{'=' * 50}")

# %% [markdown]
# ### Why Is Batch Faster?
#
# The batch approach is faster because:
#
# 1. **Single HTTP Connection**: One connection handles all symbols vs. N connections
# 2. **Yahoo API Optimization**: The API is designed for multi-ticker requests
# 3. **Internal Threading**: `yf.download()` parallelizes requests internally
# 4. **Reduced Overhead**: Less time spent on connection setup/teardown
#
# **Rule of Thumb**: For >10 symbols, always use batch downloads.

# %% [markdown]
# ## Part 2: Rate Limiting - Avoiding Blocks
#
# Yahoo Finance implements rate limiting to prevent abuse. If you download too aggressively,
# you'll get HTTP 429 (Too Many Requests) errors.
#
# ### What Triggers Rate Limiting?
#
# Based on community experience (as of late 2024):
# - **~950 symbols** in quick succession triggers blocks
# - **Aggressive sequential calls** (no delays) can trigger blocks
# - **Metadata calls** (`.info`) are more rate-limited than price downloads
#
# ### The Solution: Chunked Batches
#
# For large universes (>100 symbols), split into chunks with delays:


# %%
def download_chunked(
    symbols: list[str], start: str, end: str, chunk_size: int = 50, delay_seconds: float = 1.0
) -> tuple[list, float, int]:
    """
    Download in chunks with delays - safe for large universes.

    Args:
        symbols: List of ticker symbols
        start: Start date
        end: End date
        chunk_size: Symbols per batch (recommended: 50)
        delay_seconds: Delay between batches (recommended: 1.0)

    Returns:
        List of DataFrames, total time, successful count
    """

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    results = []
    successful = 0
    start_time = time.perf_counter()

    n_chunks = (len(symbols) + chunk_size - 1) // chunk_size

    for i, chunk in enumerate(chunks(symbols, chunk_size), 1):
        print(f"  Chunk {i}/{n_chunks}: {len(chunk)} symbols...", end=" ")

        df = yf.download(
            chunk,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=True,
        )

        if not df.empty:
            results.append(df)
            if df.columns.nlevels > 1:
                successful += len(df.columns.get_level_values(1).unique())
            else:
                successful += 1
            print("OK")
        else:
            print("EMPTY")

        # Delay between chunks (except after last)
        if i < n_chunks:
            time.sleep(delay_seconds)

    elapsed = time.perf_counter() - start_time
    return results, elapsed, successful


# %%
# Demonstrate chunked approach
print("Downloading with chunked batches (chunk_size=10)...")
chunked_results, chunked_time, chunked_count = download_chunked(
    TEST_SYMBOLS, START_DATE, END_DATE, chunk_size=10, delay_seconds=0.5
)

print("\nChunked approach:")
print(f"  Time: {chunked_time:.2f}s")
print(f"  Successful: {chunked_count}/{len(TEST_SYMBOLS)}")
print(f"  Success rate: {chunked_count / len(TEST_SYMBOLS) * 100:.1f}%")

# %% [markdown]
# ### Recommended Settings
#
# | Universe Size | chunk_size | delay_seconds | Notes |
# |--------------|------------|---------------|-------|
# | < 50 | All at once | 0 | No chunking needed |
# | 50-200 | 50 | 0.5 | Conservative |
# | 200-500 | 50 | 1.0 | Safe for S&P 500 |
# | 500+ | 50 | 2.0 | Very conservative |

# %% [markdown]
# ## Part 3: Using ml4t-data for Bulk Downloads
#
# The `ml4t-data` library provides a clean interface that handles:
# - Chunked batching automatically
# - Conversion to Polars (faster downstream processing)
# - OHLC data validation
# - Standardized schema across providers

# %%
# Initialize the provider
provider = YahooFinanceProvider()

# %%
# Batch download with ml4t-data
print("Downloading with ml4t-data fetch_batch_ohlcv()...")

start_time = time.perf_counter()
df = provider.fetch_batch_ohlcv(
    symbols=TEST_SYMBOLS,
    start=START_DATE,
    end=END_DATE,
    chunk_size=50,
    delay_seconds=0.5,
)
ml4t_time = time.perf_counter() - start_time

print("\nml4t-data approach:")
print(f"  Time: {ml4t_time:.2f}s")
print(f"  Shape: {df.shape}")
print(f"  Symbols: {df['symbol'].n_unique()}")

# %%
# Examine the output
print("\nDataFrame schema:")
print(df.schema)

print("\nSample data:")
print(df.head(10))

# %% [markdown]
# ### Key Advantages of ml4t-data
#
# 1. **Polars Output**: 10-100x faster for downstream operations
# 2. **Long Format**: One row per symbol per timestamp (easier for ML)
# 3. **Validated Schema**: Consistent columns across all providers
# 4. **Logging**: Structured logging for debugging

# %% [markdown]
# ## Part 4: Data Quality Validation
#
# Before using data for backtesting, validate it:
#
# 1. **No missing values** in critical columns
# 2. **OHLC invariants** are satisfied (high >= low, etc.)
# 3. **No duplicates** (same symbol + timestamp)
# 4. **Reasonable values** (no negative prices, etc.)


# %%
def validate_ohlcv(df: pl.DataFrame) -> dict:
    """
    Validate OHLCV data quality.

    Returns dict with validation results.
    """
    results = {}

    # 1. Check for nulls
    null_counts = df.null_count()
    results["null_counts"] = {col: null_counts[col][0] for col in df.columns}

    # 2. Check OHLC invariants
    # high >= low
    # high >= open, high >= close
    # low <= open, low <= close
    violations = df.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
    )
    results["ohlc_violations"] = len(violations)

    # 3. Check for duplicates
    duplicates = df.filter(pl.struct(["symbol", "timestamp"]).is_duplicated())
    results["duplicates"] = len(duplicates)

    # 4. Check for negative values
    negative_prices = df.filter(
        (pl.col("open") < 0) | (pl.col("high") < 0) | (pl.col("low") < 0) | (pl.col("close") < 0)
    )
    results["negative_prices"] = len(negative_prices)

    # 5. Summary stats
    results["total_rows"] = len(df)
    results["unique_symbols"] = df["symbol"].n_unique()
    results["date_range"] = (df["timestamp"].min(), df["timestamp"].max())

    return results


# %%
# Validate our downloaded data
validation = validate_ohlcv(df)

print("Data Quality Report")
print("=" * 50)
print(f"Total rows: {validation['total_rows']:,}")
print(f"Unique symbols: {validation['unique_symbols']}")
print(f"Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
print()
print("Null counts:")
for col, count in validation["null_counts"].items():
    status = "✅" if count == 0 else "⚠️"
    print(f"  {col}: {count} {status}")
print()
print(
    f"OHLC violations: {validation['ohlc_violations']} {'✅' if validation['ohlc_violations'] == 0 else '❌'}"
)
print(f"Duplicates: {validation['duplicates']} {'✅' if validation['duplicates'] == 0 else '❌'}")
print(
    f"Negative prices: {validation['negative_prices']} {'✅' if validation['negative_prices'] == 0 else '❌'}"
)

# %% [markdown]
# ## Part 5: Complete Workflow Example
#
# Here's a complete workflow for downloading S&P 500 data:

# %%
# Example: Download S&P 500 constituent data
# (Using a subset for this example)

SP500_SAMPLE = [
    # Top 50 by market cap (as of 2024)
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "LLY",
    "V",
    "UNH",
    "JPM",
    "XOM",
    "JNJ",
    "MA",
    "PG",
    "AVGO",
    "HD",
    "CVX",
    "MRK",
    "COST",
    "ABBV",
    "PEP",
    "KO",
    "ADBE",
    "WMT",
    "CRM",
    "MCD",
    "CSCO",
    "BAC",
    "NFLX",
    "PFE",
    "TMO",
    "ACN",
    "LIN",
    "ABT",
    "DHR",
    "ORCL",
    "CMCSA",
    "NKE",
    "TXN",
    "PM",
    "VZ",
    "NEE",
    "INTC",
    "WFC",
    "DIS",
    "AMD",
    "RTX",
    "UPS",
]


# %%
def download_sp500_sample():
    """
    Complete workflow for downloading market data.

    This demonstrates the recommended approach for production use.
    """
    print("=" * 60)
    print("S&P 500 Sample Download Workflow")
    print("=" * 60)

    # Step 1: Initialize provider
    print("\n1. Initializing provider...")
    provider = YahooFinanceProvider()

    # Step 2: Download data
    print("\n2. Downloading data (this may take a moment)...")
    start_time = time.perf_counter()

    df = provider.fetch_batch_ohlcv(
        symbols=SP500_SAMPLE,
        start="2020-01-01",
        end=datetime.now().strftime("%Y-%m-%d"),
        frequency="daily",
        chunk_size=50,  # Safe batch size
        delay_seconds=1.0,  # Conservative delay
    )

    download_time = time.perf_counter() - start_time
    print(f"   Downloaded in {download_time:.1f}s")

    # Step 3: Validate data
    print("\n3. Validating data quality...")
    validation = validate_ohlcv(df)

    issues = []
    if any(v > 0 for v in validation["null_counts"].values()):
        issues.append("has null values")
    if validation["ohlc_violations"] > 0:
        issues.append("has OHLC violations")
    if validation["duplicates"] > 0:
        issues.append("has duplicates")

    if issues:
        print(f"   ⚠️ Warning: Data {', '.join(issues)}")
    else:
        print("   ✅ All quality checks passed")

    # Step 4: Summary
    print("\n4. Summary:")
    print(f"   Symbols: {df['symbol'].n_unique()}")
    print(f"   Rows: {len(df):,}")
    print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"   Memory: {df.estimated_size('mb'):.1f} MB")

    return df


# %%
# Run the complete workflow
df_sp500 = download_sp500_sample()

# %%
# Show per-symbol statistics
symbol_stats = (
    df_sp500.group_by("symbol")
    .agg(
        [
            pl.len().alias("rows"),
            pl.col("timestamp").min().alias("first_date"),
            pl.col("timestamp").max().alias("last_date"),
            pl.col("close").mean().alias("avg_price"),
            pl.col("volume").mean().alias("avg_volume"),
        ]
    )
    .sort("symbol")
)

print("\nPer-symbol statistics:")
print(symbol_stats.head(10))

# %% [markdown]
# ## Summary
#
# ### Key Takeaways
#
# 1. **Use batch downloads** for >10 symbols (5-20x faster)
# 2. **Chunk large universes** (50 symbols/chunk with 1s delay)
# 3. **Always validate** data quality before backtesting
# 4. **Use ml4t-data** for production workflows (handles chunking, validation, Polars output)
#
# ### Performance Reference
#
# | Approach | Time (20 symbols) | Time (100 symbols) |
# |----------|-------------------|-------------------|
# | Single-symbol | ~3s | ~15s |
# | Batch | ~0.5s | ~2s |
# | ml4t-data batch | ~0.5s | ~4s (with delays) |
#
# ### When to Use Each Method
#
# | Use Case | Method |
# |----------|--------|
# | Initial data acquisition | `fetch_batch_ohlcv()` |
# | Daily incremental updates | `fetch_ohlcv()` + UpdateManager |
# | Quick analysis in notebook | Native `yf.download()` |
# | Production pipelines | ml4t-data with Hive storage |

# %% [markdown]
# ## Next Steps
#
# - **Storage**: See `storage_exploration.py` for Hive-partitioned Parquet storage
# - **Updates**: See `automation_updates.py` for incremental update workflows
# - **Quality**: See `data_quality_validation.py` for advanced anomaly detection
# - **Providers**: See `provider_comparison.py` for multi-provider strategies
