#!/usr/bin/env python3
"""Diagnostic trace to understand why we get 0.025% error instead of exact match.

This script traces through the calculation step-by-step to identify where
exactness breaks down.
"""

from decimal import getcontext
from pathlib import Path

import numpy as np
import polars as pl

# Set high precision for Decimal
getcontext().prec = 50


def load_aapl() -> pl.DataFrame:
    """Load AAPL data from Quandl."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    return df.filter(pl.col("ticker") == "AAPL").sort("date")


def trace_exact_calculation():
    """Trace through calculation to find where exactness breaks."""
    print("=" * 80)
    print("DIAGNOSTIC TRACE: Why 0.025% Error Instead of Exact Match?")
    print("=" * 80)
    print()

    aapl = load_aapl()
    n = len(aapl)

    print(f"Dataset: {n} records from {aapl['date'].min()} to {aapl['date'].max()}")
    print()

    # Check: Does Quandl's adj_close have rounding?
    print("1. HYPOTHESIS: Quandl's adj_close is rounded to cents")
    print("-" * 80)

    # Check if all adj_close values are exact cents (2 decimal places)
    adj_close = aapl["adj_close"].to_numpy()
    rounded_to_cents = np.round(adj_close, 2)
    exact_cents = np.all(adj_close == rounded_to_cents)

    if exact_cents:
        print("✓ ALL adj_close values are rounded to exactly 2 decimal places")
        print("  This means Quandl stores results as cents, not exact float values")
    else:
        print("✗ Some adj_close values have more than 2 decimal places")
        # Show examples
        has_more_decimals = adj_close != rounded_to_cents
        if np.any(has_more_decimals):
            examples = aapl.filter(pl.Series(has_more_decimals)).head(5)
            print(f"  Examples with >2 decimals:\n{examples.select(['date', 'adj_close'])}")
    print()

    # Check: What happens if we round our results to cents?
    print("2. TEST: Does rounding our results to cents eliminate error?")
    print("-" * 80)

    from ml4t.data.adjustments import apply_corporate_actions

    adjusted = apply_corporate_actions(
        aapl,
        split_col="split_ratio",
        dividend_col="ex-dividend",
        price_cols=["close"],
    )

    our_adj_close = adjusted["adj_close"].to_numpy()
    quandl_adj_close = aapl["adj_close"].to_numpy()

    # Before rounding
    diff_before = np.abs(our_adj_close - quandl_adj_close)
    max_error_before = (diff_before / quandl_adj_close * 100).max()

    # After rounding to cents
    our_adj_close_rounded = np.round(our_adj_close, 2)
    diff_after = np.abs(our_adj_close_rounded - quandl_adj_close)
    max_error_after = (diff_after / quandl_adj_close * 100).max()

    print(f"Before rounding: {max_error_before:.6f}% max error")
    print(f"After rounding:  {max_error_after:.6f}% max error")

    if max_error_after < 1e-10:
        print("✓ FOUND IT! Rounding to cents eliminates the error")
        print("  Conclusion: Quandl rounds intermediate results during calculation")
    else:
        print("✗ Rounding doesn't fully explain the error")
    print()

    # Check: Trace through a specific split event
    print("3. DETAILED TRACE: June 1987 2:1 Split")
    print("-" * 80)

    # Find the 1987-06-16 split
    split_date = "1987-06-16"
    split_row = aapl.filter(pl.col("date") == pl.lit(split_date).str.to_datetime()).row(
        0, named=True
    )

    # Get day before split
    (
        aapl.filter(pl.col("date") == pl.lit(split_date).str.to_datetime())
        .select(pl.col("date").rank("dense").over(pl.lit(1)).alias("idx"))
        .item()
    )

    before_split = (
        aapl.filter(pl.col("date") < pl.lit(split_date).str.to_datetime())
        .tail(1)
        .row(0, named=True)
    )

    print(f"Day before split ({before_split['date']}):")
    print(f"  Close: ${before_split['close']:.2f}")
    print(f"  Adj Close: ${before_split['adj_close']:.2f}")
    print()
    print(f"Split day ({split_row['date']}):")
    print(f"  Close: ${split_row['close']:.2f} (should be ~half of previous)")
    print(f"  Split ratio: {split_row['split_ratio']}")
    print(f"  Adj Close: ${split_row['adj_close']:.2f}")
    print()

    # What does our formula produce?
    # Working backwards: adj_before = adj_split * (close_before / split_ratio - div) / close_split
    close_before = before_split["close"]
    close_split = split_row["close"]
    split_ratio = split_row["split_ratio"]
    div_split = split_row["ex-dividend"]
    adj_split = split_row["adj_close"]

    # Our calculation
    adjustment_factor = (close_before / split_ratio - div_split) / close_split
    our_adj_before = adj_split * adjustment_factor
    quandl_adj_before = before_split["adj_close"]

    print("Our formula:")
    print(
        f"  adjustment_factor = ({close_before:.2f} / {split_ratio} - {div_split:.2f}) / {close_split:.2f}"
    )
    print(f"  adjustment_factor = {adjustment_factor:.10f}")
    print(f"  adj_before = {adj_split:.2f} × {adjustment_factor:.10f} = ${our_adj_before:.6f}")
    print()
    print(f"Quandl's adj_close: ${quandl_adj_before:.6f}")
    print(f"Difference: ${abs(our_adj_before - quandl_adj_before):.6f}")
    print(
        f"Relative error: {abs(our_adj_before - quandl_adj_before) / quandl_adj_before * 100:.6f}%"
    )
    print()

    # Check: Is Quandl rounding at each step?
    print("4. HYPOTHESIS: Quandl rounds adj_close to cents at EACH iteration")
    print("-" * 80)

    # Simulate with rounding at each step
    close_vals = aapl["close"].to_numpy()
    split_vals = aapl["split_ratio"].to_numpy()
    div_vals = aapl["ex-dividend"].to_numpy()

    # Our approach: no intermediate rounding
    adj_no_round = close_vals.copy()
    for i in range(n - 2, -1, -1):
        factor = (close_vals[i] / split_vals[i + 1] - div_vals[i + 1]) / close_vals[i + 1]
        adj_no_round[i] = adj_no_round[i + 1] * factor

    # With intermediate rounding to cents
    adj_with_round = close_vals.copy()
    for i in range(n - 2, -1, -1):
        factor = (close_vals[i] / split_vals[i + 1] - div_vals[i + 1]) / close_vals[i + 1]
        adj_with_round[i] = np.round(adj_with_round[i + 1] * factor, 2)  # Round to cents

    # Compare both to Quandl
    error_no_round = np.abs(adj_no_round - quandl_adj_close) / quandl_adj_close * 100
    error_with_round = np.abs(adj_with_round - quandl_adj_close) / quandl_adj_close * 100

    print(f"No intermediate rounding:   max error = {error_no_round.max():.6f}%")
    print(f"With rounding to cents:     max error = {error_with_round.max():.6f}%")
    print()

    if error_with_round.max() < error_no_round.max():
        print("✓ Intermediate rounding REDUCES error")
        print("  Conclusion: Quandl likely rounds to cents at each iteration")
    else:
        print("✗ Intermediate rounding INCREASES error")
        print("  Conclusion: Quandl does NOT round at each iteration")
    print()

    # Check: Show where maximum error occurs
    print("5. LOCATION OF MAXIMUM ERROR")
    print("-" * 80)

    max_error_idx = int(error_no_round.argmax())
    max_error_row = aapl[max_error_idx]

    print(f"Max error at index {max_error_idx}:")
    print(f"  Date: {max_error_row['date'][0]}")
    print(f"  Close: ${max_error_row['close'][0]:.2f}")
    print(f"  Our adj_close: ${adj_no_round[max_error_idx]:.6f}")
    print(f"  Quandl adj_close: ${quandl_adj_close[max_error_idx]:.6f}")
    print(f"  Error: {error_no_round[max_error_idx]:.6f}%")
    print()

    # How many days from end?
    days_from_end = n - max_error_idx - 1
    print(f"This is {days_from_end:,} days before the most recent date")
    print(f"Cumulative effect of {days_from_end:,} iterations")
    print()

    return {
        "exact_cents": exact_cents,
        "error_before_rounding": max_error_before,
        "error_after_rounding": max_error_after,
        "error_no_round": error_no_round.max(),
        "error_with_round": error_with_round.max(),
    }


if __name__ == "__main__":
    results = trace_exact_calculation()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if results["exact_cents"]:
        print("KEY FINDING: Quandl stores adj_close as dollars and cents (2 decimals)")

    if results["error_after_rounding"] < 1e-10:
        print("KEY FINDING: Rounding our results to cents matches Quandl exactly")
        print()
        print("CONCLUSION:")
        print("  The 0.025% 'error' is actually just a display/comparison artifact.")
        print("  Our formula is CORRECT. Quandl rounds to cents for practical use.")
        print("  If we round our adj_close to cents, we get exact match.")
    else:
        print("PROBLEM: Even after rounding, there's still error")
        print("  Need to investigate Quandl's exact calculation methodology")
