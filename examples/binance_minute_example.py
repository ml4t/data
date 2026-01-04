"""Binance Minute Data Example

Demonstrates how to fetch high-frequency minute OHLCV data from
Binance Public Data (data.binance.vision).

Key features:
- No API key required (public S3 bucket)
- No rate limits or geo-restrictions
- Bulk historical data (years of minute bars)
- Both spot and futures markets

Available history:
- BTC Spot 1m: from 2017-08-17 (~4.3M bars, ~207MB compressed)
- BTC Futures 1m: from 2019-12-31 (~3.1M bars, ~148MB)
- ETH Spot 1m: from 2017-08-17

Requirements:
    - ml4t-data installed: pip install ml4t-data
    - No API key needed (public S3 bucket)
    - Internet connection

Usage:
    python examples/binance_minute_example.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

from ml4t.data.providers import BinancePublicProvider


def main():
    """Run the Binance minute data example."""
    print("=" * 80)
    print("  Binance Minute Data Example")
    print("  High-frequency OHLCV from data.binance.vision")
    print("=" * 80)
    print()

    # Demo: fetch 3 days of minute data (~4,320 bars)
    # Using fixed historical dates for reliable results
    # (Binance Public Data has a 1-2 day upload lag)
    start_date = "2024-11-01"
    end_date = "2024-11-03"

    # =========================================================================
    # SPOT MARKET MINUTE DATA
    # =========================================================================
    print("=" * 80)
    print("  Part 1: Spot Market Minute Data")
    print("=" * 80)
    print()

    print(f"Fetching BTC spot 1-minute data from {start_date} to {end_date}...")

    # Initialize provider for spot market (default)
    spot_provider = BinancePublicProvider(market="spot")

    # Fetch 1-minute OHLCV
    spot_1m = spot_provider.fetch_ohlcv("BTCUSDT", start_date, end_date, frequency="minute")

    print(f"Fetched {len(spot_1m):,} minute bars")
    print()

    if spot_1m.is_empty():
        print("  No data available - Binance Public Data may have a lag.")
        print("  Try dates at least 3-5 days in the past.")
        print()
    else:
        # Show sample
        print("Sample data (first 10 rows):")
        print()
        print(spot_1m.head(10))
        print()

        # Calculate intraday statistics
        print("Intraday Statistics:")
        print()
        print(f"  Total Volume:     {spot_1m['volume'].sum():,.0f} BTC")
        print(f"  Avg Volume/min:   {spot_1m['volume'].mean():,.2f} BTC")
        print(f"  Max Volume/min:   {spot_1m['volume'].max():,.2f} BTC")
        print()

        # Calculate returns
        returns = spot_1m["close"].pct_change().drop_nulls()
        print(f"  Avg Return/min:   {returns.mean() * 100:.4f}%")
        print(f"  Volatility/min:   {returns.std() * 100:.4f}%")
        print(f"  Max Up Move:      {returns.max() * 100:.2f}%")
        print(f"  Max Down Move:    {returns.min() * 100:.2f}%")
        print()

    # =========================================================================
    # FUTURES MARKET MINUTE DATA
    # =========================================================================
    print("=" * 80)
    print("  Part 2: Futures Market Minute Data")
    print("=" * 80)
    print()

    print(f"Fetching BTC futures 1-minute data from {start_date} to {end_date}...")

    # Initialize provider for futures market
    futures_provider = BinancePublicProvider(market="futures")

    # Fetch 1-minute OHLCV
    futures_1m = futures_provider.fetch_ohlcv("BTCUSDT", start_date, end_date, frequency="minute")

    print(f"Fetched {len(futures_1m):,} minute bars")
    print()

    # Compare spot vs futures (if data available)
    if not spot_1m.is_empty() and not futures_1m.is_empty():
        print("Spot vs Futures Comparison (last close):")
        print()
        spot_close = spot_1m["close"][-1]
        futures_close = futures_1m["close"][-1]
        basis = futures_close - spot_close
        basis_pct = (basis / spot_close) * 100

        print(f"  Spot Close:       ${spot_close:,.2f}")
        print(f"  Futures Close:    ${futures_close:,.2f}")
        print(f"  Basis:            ${basis:,.2f} ({basis_pct:+.4f}%)")
        print()
    else:
        print("  (Skipping comparison - data not available)")
        print()

    # =========================================================================
    # DIFFERENT TIMEFRAMES
    # =========================================================================
    print("=" * 80)
    print("  Part 3: Available Timeframes")
    print("=" * 80)
    print()

    print("Available timeframes from Binance Public Data:")
    print()
    print("  Minute:    1m, 3m, 5m, 15m, 30m")
    print("  Hourly:    1h, 2h, 4h, 6h, 8h, 12h")
    print("  Daily+:    1d, 3d, 1w, 1mo")
    print()

    # Fetch 5-minute data example
    print("Fetching 5-minute data...")
    spot_5m = spot_provider.fetch_ohlcv("BTCUSDT", start_date, end_date, frequency="5minute")
    print(f"  5m bars: {len(spot_5m):,} (vs {len(spot_1m):,} 1m bars)")
    print()

    # Fetch hourly data example
    print("Fetching hourly data...")
    spot_1h = spot_provider.fetch_ohlcv("BTCUSDT", start_date, end_date, frequency="hourly")
    print(f"  1h bars: {len(spot_1h):,} (vs {len(spot_1m):,} 1m bars)")
    print()

    # =========================================================================
    # RESAMPLING MINUTE DATA
    # =========================================================================
    print("=" * 80)
    print("  Part 4: Resampling Minute Data")
    print("=" * 80)
    print()

    if not spot_1m.is_empty():
        print("Resampling 1-minute to 15-minute bars using Polars...")
        print()

        # Resample to 15-minute bars
        resampled = spot_1m.group_by_dynamic("timestamp", every="15m").agg(
            [
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            ]
        )

        print(f"  Original:   {len(spot_1m):,} 1-minute bars")
        print(f"  Resampled:  {len(resampled):,} 15-minute bars")
        print()
        print("Sample 15-minute bars:")
        print()
        print(resampled.head(10))
        print()
    else:
        print("  (Skipping resampling - no minute data available)")
        print()

    # =========================================================================
    # DATA SIZE ESTIMATES
    # =========================================================================
    print("=" * 80)
    print("  Part 5: Historical Data Availability")
    print("=" * 80)
    print()
    print("  BTC/USDT Spot:")
    print("    - Available from: 2017-08-17")
    print("    - ~4.3 million 1-minute bars")
    print("    - ~207 MB compressed (~500 MB uncompressed)")
    print()
    print("  BTC/USDT Futures:")
    print("    - Available from: 2019-12-31")
    print("    - ~3.1 million 1-minute bars")
    print("    - ~148 MB compressed")
    print()
    print("  Tip: Use monthly files for bulk downloads (>60 days)")
    print("       The provider does this automatically!")
    print()

    # =========================================================================
    # USE CASES
    # =========================================================================
    print("=" * 80)
    print("  Use Cases for Minute Data")
    print("=" * 80)
    print()
    print("  1. Intraday Strategy Development:")
    print("     - Momentum, mean-reversion at minute scale")
    print("     - Market microstructure analysis")
    print()
    print("  2. Feature Engineering:")
    print("     - High-frequency volatility estimators")
    print("     - Volume profile analysis")
    print("     - Tick imbalance / volume imbalance bars")
    print()
    print("  3. Execution Analysis:")
    print("     - Slippage estimation")
    print("     - Optimal execution timing")
    print("     - VWAP/TWAP benchmarks")
    print()
    print("  4. Risk Management:")
    print("     - Intraday drawdown monitoring")
    print("     - High-frequency VaR/ES")
    print()

    # Clean up
    spot_provider.close()
    futures_provider.close()

    print("=" * 80)
    print("  Done! Happy trading!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as err:
        print(f"\n\nError: {err}")
        import traceback

        traceback.print_exc()
        print()
        print("Common issues:")
        print("  - Check internet connection")
        print("  - Try a smaller date range first")
        print("  - Minute data files can be large (~10MB/day)")
        print()
        sys.exit(1)
