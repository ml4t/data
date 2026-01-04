#!/usr/bin/env python3
"""Educational example: Corporate actions and price adjustments.

This example demonstrates:
1. How stock splits and dividends affect historical prices
2. How to apply corporate action adjustments to unadjusted data
3. How to validate adjustment logic against known-good data (Quandl)

For ML4T Book - Chapter on Data Quality and Preprocessing
"""

from pathlib import Path

import numpy as np
import polars as pl

from ml4t.data.adjustments import apply_corporate_actions


def load_quandl_example(ticker: str = "AAPL") -> pl.DataFrame:
    """Load Quandl WIKI data for demonstration."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"

    if not quandl_path.exists():
        raise FileNotFoundError(
            f"Quandl data not found at {quandl_path}\n"
            "This example requires the Quandl WIKI dataset for validation."
        )

    df = pl.read_parquet(quandl_path)
    return df.filter(pl.col("ticker") == ticker).sort("date")


def demonstrate_corporate_actions():
    """Educational demonstration of corporate actions adjustments."""

    print("=" * 80)
    print("CORPORATE ACTIONS DEMONSTRATION - AAPL")
    print("=" * 80)
    print()

    # Load AAPL data from Quandl
    aapl = load_quandl_example("AAPL")

    print("📊 Data Overview:")
    print(f"   Date range: {aapl['date'].min()} to {aapl['date'].max()}")
    print(f"   Total records: {len(aapl):,}")
    print()

    # Show corporate actions
    splits = aapl.filter(pl.col("split_ratio") != 1.0)
    dividends = aapl.filter(pl.col("ex-dividend") > 0)

    print(f"🔄 Stock Splits ({len(splits)} events):")
    print(splits.select(["date", "close", "split_ratio", "adj_close"]))
    print()

    print(f"💰 Dividends (showing first 5 of {len(dividends)} events):")
    print(dividends.select(["date", "close", "ex-dividend", "adj_close"]).head(5))
    print()

    # ========================================================================
    # DEMONSTRATION 1: Apply our adjustment logic
    # ========================================================================

    print("=" * 80)
    print("STEP 1: Apply Our Adjustment Logic")
    print("=" * 80)
    print()

    # Apply corporate actions using our utilities
    adjusted = apply_corporate_actions(
        aapl,
        split_col="split_ratio",
        dividend_col="ex-dividend",
        price_cols=["open", "high", "low", "close"],
        volume_col="volume",
    )

    print("✅ Applied corporate actions adjustments:")
    print("   - Split adjustments (4 events)")
    print("   - Dividend adjustments (54 events)")
    print()

    # Show example: unadjusted vs adjusted
    example_dates = [
        "1980-12-12",  # First date
        "1987-06-16",  # First split
        "2014-06-09",  # 7:1 split
        "2018-03-27",  # Last date
    ]

    print("📈 Example: Unadjusted vs Adjusted Prices")
    print("-" * 80)

    for date_str in example_dates:
        row = adjusted.filter(pl.col("date") == pl.lit(date_str).str.to_datetime()).row(
            0, named=True
        )
        split_ratio = row.get("split_ratio", 1.0)

        print(f"  {date_str}:")
        print(f"    Unadjusted close: ${row['close']:>8.2f}")
        print(f"    Adjusted close:   ${row['adj_close']:>8.2f}")
        if split_ratio != 1.0:
            print(f"    Split: {split_ratio}:1")
        print()

    # ========================================================================
    # DEMONSTRATION 2: Validate against Quandl's pre-calculated results
    # ========================================================================

    print("=" * 80)
    print("STEP 2: Validate Against Quandl (Known-Good Data)")
    print("=" * 80)
    print()

    # Compare our adjusted prices to Quandl's
    our_adj_close = adjusted["adj_close"].to_numpy()
    quandl_adj_close = aapl["adj_close"].to_numpy()

    # Calculate differences
    absolute_diff = np.abs(our_adj_close - quandl_adj_close)
    relative_diff = absolute_diff / quandl_adj_close * 100

    max_abs_diff = absolute_diff.max()
    max_rel_diff = relative_diff.max()
    mean_rel_diff = relative_diff.mean()

    print("📊 Validation Results:")
    print(f"   Total comparisons: {len(our_adj_close):,}")
    print(f"   Max absolute difference: ${max_abs_diff:.6f}")
    print(f"   Max relative difference: {max_rel_diff:.4f}%")
    print(f"   Mean relative difference: {mean_rel_diff:.4f}%")
    print()

    # Check if close enough (allow tiny floating point errors)
    # With 9400 iterative calculations, rounding errors can accumulate to ~0.03%
    tolerance = 5e-4  # 0.05% tolerance (accounts for rounding in iterative calculation)
    is_valid = np.allclose(our_adj_close, quandl_adj_close, rtol=tolerance)

    if is_valid:
        print("✅ VALIDATION PASSED!")
        print(
            f"   Our adjustment logic matches Quandl's results within {tolerance * 100:.4f}% tolerance"
        )
    else:
        print("❌ VALIDATION FAILED!")
        print("   Our logic differs from Quandl's calculations")

        # Show problematic rows
        problematic = np.where(relative_diff > tolerance * 100)[0]
        if len(problematic) > 0:
            print(f"\n   Problematic records: {len(problematic)}")
            print("\n   First 5 differences:")
            for idx in problematic[:5]:
                date = adjusted["date"][int(idx)]
                ours = our_adj_close[idx]
                theirs = quandl_adj_close[idx]
                diff = relative_diff[idx]
                print(f"     {date}: ${ours:.6f} vs ${theirs:.6f} ({diff:.4f}% diff)")

    print()

    # ========================================================================
    # DEMONSTRATION 3: Educational insights
    # ========================================================================

    print("=" * 80)
    print("KEY INSIGHTS FOR ML4T READERS")
    print("=" * 80)
    print()

    print("1. STOCK SPLITS:")
    print("   - 2:1 split: 1 share @ $100 → 2 shares @ $50 each")
    print("   - Historical prices divided by split ratio for continuity")
    print(f"   - AAPL cumulative split factor: {splits['split_ratio'].product():.1f}x")
    print()

    print("2. DIVIDEND ADJUSTMENTS:")
    print("   - Ex-dividend: price typically drops by dividend amount")
    print("   - Adjustment shows 'total return' (price + reinvested dividends)")
    print("   - Important for long-term backtesting (decades of data)")
    print()

    print("3. DATA SOURCE DIFFERENCES:")
    print("   - Quandl 'unadjusted' = truly unadjusted")
    print("   - Yahoo 'unadjusted' = split-adjusted (but not dividend-adjusted!)")
    print("   - Must handle when chaining data sources")
    print()

    print("4. VALIDATION MATTERS:")
    print("   - Always validate against known-good data when possible")
    print("   - Tiny errors compound over 30+ years of history")
    print("   - Quandl provides excellent validation baseline (ended 2018)")
    print()

    return adjusted


if __name__ == "__main__":
    try:
        result = demonstrate_corporate_actions()
        print("=" * 80)
        print(f"✅ Demo complete. Adjusted data shape: {result.shape}")
        print("=" * 80)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nThis example requires the Quandl WIKI dataset.")
        print("Expected location: ~/ml3t/data/equities/quandl/wiki_prices.parquet")
