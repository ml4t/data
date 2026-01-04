"""Multi-Asset Data Loading - Quick Start Examples.

This demonstrates the new batch_load() API for efficient multi-symbol data loading.
"""

from ml4t.data.data_manager import DataManager


def example_basic_batch_load():
    """Example 1: Basic batch loading of multiple symbols."""
    print("\n=== Example 1: Basic Batch Load ===")

    manager = DataManager()

    # Load multiple symbols in one call
    df = manager.batch_load(
        symbols=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
        start="2024-01-01",
        end="2024-01-31",
        provider="yahoo",
    )

    print(f"Loaded {len(df)} rows for {df['symbol'].n_unique()} symbols")
    print("\nFirst few rows:")
    print(df.head())

    print(f"\nSymbols: {df['symbol'].unique().sort().to_list()}")
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")


def example_graceful_failure():
    """Example 2: Graceful handling of partial failures."""
    print("\n=== Example 2: Graceful Partial Failure ===")

    manager = DataManager()

    # Some symbols may fail - by default, returns partial results
    df = manager.batch_load(
        symbols=["AAPL", "INVALID_SYMBOL", "MSFT"],
        start="2024-01-01",
        end="2024-01-31",
        fail_on_partial=False,  # Default: graceful degradation
        provider="yahoo",
    )

    print(f"Successfully loaded: {df['symbol'].unique().to_list()}")
    print("(INVALID_SYMBOL was skipped gracefully)")


def example_parallel_loading():
    """Example 3: Fast parallel loading with configurable workers."""
    print("\n=== Example 3: Parallel Loading ===")

    import time

    manager = DataManager()

    # Load many symbols in parallel for speed
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "WMT"]

    start_time = time.time()
    df = manager.batch_load(
        symbols=symbols,
        start="2024-01-01",
        end="2024-01-31",
        max_workers=8,  # Parallel fetching
        provider="yahoo",
    )
    elapsed = time.time() - start_time

    print(f"Loaded {len(symbols)} symbols in {elapsed:.2f} seconds")
    print(f"Average: {elapsed / len(symbols):.2f} sec/symbol")
    print(f"Total rows: {len(df):,}")


def example_cross_sectional_analysis():
    """Example 4: Using stacked format for cross-sectional analysis."""
    print("\n=== Example 4: Cross-Sectional Analysis ===")

    import polars as pl

    manager = DataManager()

    # Load tech stocks
    df = manager.batch_load(
        symbols=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
        start="2024-01-01",
        end="2024-01-31",
        provider="yahoo",
    )

    # Calculate returns by symbol
    df = df.with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))

    # Get cross-sectional stats for each date
    daily_stats = (
        df.group_by("timestamp")
        .agg(
            [
                pl.col("returns").mean().alias("mean_return"),
                pl.col("returns").std().alias("std_return"),
                pl.col("volume").sum().alias("total_volume"),
            ]
        )
        .sort("timestamp")
    )

    print("\nDaily cross-sectional statistics:")
    print(daily_stats.head())


def example_sector_analysis():
    """Example 5: Sector-based analysis with grouped operations."""
    print("\n=== Example 5: Sector Analysis ===")

    import polars as pl

    manager = DataManager()

    # Load stocks from different sectors
    tech_stocks = ["AAPL", "MSFT", "GOOG"]
    finance_stocks = ["JPM", "BAC", "GS"]

    # Load all symbols
    all_symbols = tech_stocks + finance_stocks
    df = manager.batch_load(
        symbols=all_symbols, start="2024-01-01", end="2024-01-31", provider="yahoo"
    )

    # Add sector labels
    df = df.with_columns(
        pl.when(pl.col("symbol").is_in(tech_stocks))
        .then(pl.lit("Tech"))
        .otherwise(pl.lit("Finance"))
        .alias("sector")
    )

    # Calculate sector performance
    sector_perf = (
        df.group_by(["timestamp", "sector"])
        .agg(
            [
                pl.col("close").mean().alias("avg_close"),
                pl.col("volume").sum().alias("total_volume"),
            ]
        )
        .sort(["timestamp", "sector"])
    )

    print("\nSector performance:")
    print(sector_perf.head(10))


def example_universe_loading():
    """Example 6: Loading pre-defined universes (future feature)."""
    print("\n=== Example 6: Universe Loading (Coming Soon) ===")

    print("Future API (TASK-003):")
    print("""
    # Load entire S&P 500
    df = manager.batch_load_universe(
        universe='SP500',
        start='2024-01-01',
        end='2024-01-31'
    )

    # Or custom universe
    df = manager.batch_load_universe(
        universe=['SPY', 'QQQ', 'IWM'],  # ETFs
        start='2024-01-01',
        end='2024-01-31'
    )
    """)


if __name__ == "__main__":
    # Run examples
    example_basic_batch_load()
    example_graceful_failure()
    example_parallel_loading()
    example_cross_sectional_analysis()
    example_sector_analysis()
    example_universe_loading()

    print("\n" + "=" * 60)
    print("Multi-asset loading examples completed!")
    print("=" * 60)
