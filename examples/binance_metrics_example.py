"""Binance Futures Metrics Example

Demonstrates how to fetch Open Interest and Long/Short ratio data
from Binance Public Data (data.binance.vision).

This data is valuable for:
- Sentiment analysis (long/short ratios)
- Position crowding detection (open interest)
- Market regime identification

Available metrics (5-minute intervals):
- open_interest: Sum of open interest (contracts)
- open_interest_value: Sum of open interest (USD)
- toptrader_long_short_ratio: Top trader long/short ratio
- account_long_short_ratio: Account-level long/short ratio
- taker_volume_ratio: Taker buy/sell volume ratio

Data available since: 2021-12-01

Requirements:
    - ml4t-data installed: pip install ml4t-data
    - No API key needed (public S3 bucket)
    - Internet connection

Usage:
    python examples/binance_metrics_example.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.providers import BinancePublicProvider


def main():
    """Run the Binance metrics example."""
    print("=" * 80)
    print("  Binance Futures Metrics Example")
    print("  Open Interest & Long/Short Ratios from data.binance.vision")
    print("=" * 80)
    print()

    # Use fixed historical dates for reliable results
    # (Binance Public Data has a 1-2 day upload lag)
    start_date = "2024-11-01"
    end_date = "2024-11-07"

    print(f"Fetching BTC futures metrics from {start_date} to {end_date}...")
    print()

    # Initialize provider for futures market
    provider = BinancePublicProvider(market="futures")

    # Fetch metrics data (5-minute intervals)
    metrics = provider.fetch_metrics("BTCUSDT", start_date, end_date)

    print(f"Fetched {len(metrics):,} rows ({len(metrics) * 5 / 60:.1f} hours of data)")
    print()

    # Show schema
    print("=" * 80)
    print("  Data Schema")
    print("=" * 80)
    print()
    for col, dtype in zip(metrics.columns, metrics.dtypes):
        print(f"  {col:35} {dtype}")
    print()

    # Show sample data
    print("=" * 80)
    print("  Sample Data (first 10 rows)")
    print("=" * 80)
    print()
    print(
        metrics.select(
            "timestamp",
            "open_interest_value",
            "toptrader_long_short_ratio",
            "account_long_short_ratio",
        ).head(10)
    )
    print()

    # Basic statistics
    print("=" * 80)
    print("  Open Interest Statistics")
    print("=" * 80)
    print()
    oi_usd = metrics["open_interest_value"]
    print(f"  Current OI:  ${oi_usd[-1]:,.0f}")
    print(f"  Max OI:      ${oi_usd.max():,.0f}")
    print(f"  Min OI:      ${oi_usd.min():,.0f}")
    print(f"  Mean OI:     ${oi_usd.mean():,.0f}")
    print()

    # Long/Short ratio analysis
    print("=" * 80)
    print("  Long/Short Ratio Analysis")
    print("=" * 80)
    print()
    ls_ratio = metrics["toptrader_long_short_ratio"]
    print(f"  Current L/S:  {ls_ratio[-1]:.3f}")
    print(f"  Max L/S:      {ls_ratio.max():.3f} (most long)")
    print(f"  Min L/S:      {ls_ratio.min():.3f} (most short)")
    print(f"  Mean L/S:     {ls_ratio.mean():.3f}")
    print()

    # Interpret current sentiment
    current_ls = ls_ratio[-1]
    if current_ls > 1.5:
        sentiment = "Heavily Long"
    elif current_ls > 1.1:
        sentiment = "Moderately Long"
    elif current_ls > 0.9:
        sentiment = "Neutral"
    elif current_ls > 0.6:
        sentiment = "Moderately Short"
    else:
        sentiment = "Heavily Short"

    print(f"  Interpretation: Top traders are {sentiment}")
    print()

    # Resample to hourly for plotting
    print("=" * 80)
    print("  Hourly Aggregation Example")
    print("=" * 80)
    print()

    hourly = metrics.group_by_dynamic("timestamp", every="1h").agg(
        [
            pl.col("open_interest_value").mean().alias("oi_usd"),
            pl.col("toptrader_long_short_ratio").mean().alias("ls_ratio"),
            pl.col("account_long_short_ratio").mean().alias("account_ls"),
        ]
    )

    print(f"  Aggregated {len(metrics):,} rows -> {len(hourly):,} hourly bars")
    print()
    print(hourly.tail(10))
    print()

    # Use cases
    print("=" * 80)
    print("  Use Cases for This Data")
    print("=" * 80)
    print()
    print("  1. Sentiment Analysis:")
    print("     - Track L/S ratio changes over time")
    print("     - Identify crowded positions")
    print("     - Contrarian signals at extremes")
    print()
    print("  2. Open Interest Analysis:")
    print("     - Track position buildup/unwind")
    print("     - Identify market conviction")
    print("     - Spot potential liquidation cascades")
    print()
    print("  3. Feature Engineering:")
    print("     - Create OI momentum features")
    print("     - L/S ratio z-scores")
    print("     - Cross-asset OI correlations")
    print()

    # Available symbols
    print("=" * 80)
    print("  Available Symbols")
    print("=" * 80)
    print()
    print("  Metrics data is available for all USD-M perpetual futures:")
    print()
    symbols = provider.get_available_symbols()[:10]
    print(f"  {', '.join(symbols)}, ...")
    print()

    provider.close()

    print("=" * 80)
    print("  Done! Happy trading!")
    print("=" * 80)
    print()


# Need polars for group_by_dynamic
import polars as pl

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as err:
        print(f"\n\nError: {err}")
        print()
        print("Common issues:")
        print("  - Check internet connection")
        print("  - Metrics data available since 2021-12-01")
        print("  - Only USD-M futures have metrics data (not spot)")
        print()
        sys.exit(1)
