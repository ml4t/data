#!/usr/bin/env python3
"""Final diagnosis of the 0.02486% error."""

from pathlib import Path

import numpy as np
import polars as pl


def load_aapl() -> pl.DataFrame:
    """Load AAPL data from Quandl."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    return df.filter(pl.col("ticker") == "AAPL").sort("date")


def final_diagnosis():
    """Final diagnosis."""
    print("=" * 80)
    print("FINAL DIAGNOSIS: The 0.02486% Error")
    print("=" * 80)
    print()

    aapl = load_aapl()
    n = len(aapl)

    # Key finding: Error drops to 0 after Aug 10, 2017 (index 9243)
    # Aug 10, 2017 is the LAST dividend in the dataset
    # Dataset ends March 27, 2018 (index 9399)

    print("Key Facts:")
    print(f"  Total records: {n}")
    print(f"  First date: {aapl['date'][0]}")
    print(f"  Last date: {aapl['date'][n - 1]}")
    print("  Last dividend: 2017-08-10 (index 9243)")
    print()

    # Get the last dividend and surrounding dates
    aug10_2017_idx = (
        aapl.filter(pl.col("date") == pl.lit("2017-08-10").str.to_datetime())
        .select(pl.int_range(0, pl.len()))
        .to_numpy()[0][0]
    )

    aug9_idx = aug10_2017_idx - 1
    aug11_idx = aug10_2017_idx + 1

    print("Last dividend (August 10, 2017):")
    row_aug10 = aapl[aug10_2017_idx]
    print(f"  Close: ${row_aug10['close'][0]:.2f}")
    print(f"  Dividend: ${row_aug10['ex-dividend'][0]:.2f}")
    print(f"  Quandl adj_close: ${row_aug10['adj_close'][0]:.2f}")
    print()

    print("Day before (August 9, 2017):")
    row_aug9 = aapl[aug9_idx]
    print(f"  Close: ${row_aug9['close'][0]:.2f}")
    print(f"  Quandl adj_close: ${row_aug9['adj_close'][0]:.6f}")
    print()

    print("Day after (August 11, 2017):")
    row_aug11 = aapl[aug11_idx]
    print(f"  Close: ${row_aug11['close'][0]:.2f}")
    print(f"  Quandl adj_close: ${row_aug11['adj_close'][0]:.6f}")
    print()

    # Now calculate using our formula
    close_vals = aapl["close"].to_numpy()
    split_vals = aapl["split_ratio"].to_numpy()
    div_vals = aapl["ex-dividend"].to_numpy()

    # Calculate our adj_close
    our_adj = close_vals.copy()
    for i in range(n - 2, -1, -1):
        factor = (close_vals[i] / split_vals[i + 1] - div_vals[i + 1]) / close_vals[i + 1]
        our_adj[i] = our_adj[i + 1] * factor

    print("=" * 80)
    print("OUR CALCULATION AT THE CRITICAL DATES")
    print("=" * 80)
    print()

    print("Working backwards from last date (March 27, 2018):")
    print(f"  Index {n - 1}: ${close_vals[n - 1]:.2f} (adj = close, no more events)")
    print()

    # Show calculation for Aug 11 (first date after last dividend)
    print("August 11, 2017 (first date after last dividend):")
    print(f"  Index: {aug11_idx}")

    # How many steps from aug11 to end?
    steps_to_end = n - 1 - aug11_idx
    print(f"  Steps to end of dataset: {steps_to_end}")

    # Calculate manually
    print(f"  Our adj[{aug11_idx}] = ${our_adj[aug11_idx]:.6f}")
    print(f"  Quandl adj[{aug11_idx}] = ${row_aug11['adj_close'][0]:.6f}")
    print(
        f"  Error: {abs(our_adj[aug11_idx] - row_aug11['adj_close'][0]) / row_aug11['adj_close'][0] * 100:.10f}%"
    )
    print()

    # Show calculation for Aug 10 (dividend date)
    print("August 10, 2017 (dividend date):")
    print(f"  Our adj[{aug10_2017_idx}] = ${our_adj[aug10_2017_idx]:.6f}")
    print(f"  Quandl adj[{aug10_2017_idx}] = ${row_aug10['adj_close'][0]:.6f}")
    print(
        f"  Error: {abs(our_adj[aug10_2017_idx] - row_aug10['adj_close'][0]) / row_aug10['adj_close'][0] * 100:.10f}%"
    )
    print()

    # Formula from Aug 11 to Aug 10
    close_aug10 = close_vals[aug10_2017_idx]
    close_aug11 = close_vals[aug11_idx]
    split_aug11 = split_vals[aug11_idx]
    div_aug11 = div_vals[aug11_idx]

    factor_to_aug10 = (close_aug10 / split_aug11 - div_aug11) / close_aug11
    our_aug10_from_aug11 = our_adj[aug11_idx] * factor_to_aug10

    print(
        "Formula: adj[Aug10] = adj[Aug11] × (close[Aug10] / split[Aug11] - div[Aug11]) / close[Aug11]"
    )
    print(
        f"  = {our_adj[aug11_idx]:.2f} × ({close_aug10:.2f} / {split_aug11} - {div_aug11:.2f}) / {close_aug11:.2f}"
    )
    print(f"  = {our_adj[aug11_idx]:.2f} × {factor_to_aug10:.10f}")
    print(f"  = ${our_aug10_from_aug11:.6f}")
    print()

    # Show calculation for Aug 9 (day before dividend)
    print("August 9, 2017 (day before dividend):")
    print(f"  Our adj[{aug9_idx}] = ${our_adj[aug9_idx]:.6f}")
    print(f"  Quandl adj[{aug9_idx}] = ${row_aug9['adj_close'][0]:.6f}")
    print(
        f"  Error: {abs(our_adj[aug9_idx] - row_aug9['adj_close'][0]) / row_aug9['adj_close'][0] * 100:.10f}%"
    )
    print()

    # Formula from Aug 10 to Aug 9
    close_aug9 = close_vals[aug9_idx]
    split_aug10 = split_vals[aug10_2017_idx]
    div_aug10 = div_vals[aug10_2017_idx]

    factor_to_aug9 = (close_aug9 / split_aug10 - div_aug10) / close_aug10
    our_aug9_from_aug10 = our_adj[aug10_2017_idx] * factor_to_aug9

    print(
        "Formula: adj[Aug9] = adj[Aug10] × (close[Aug9] / split[Aug10] - div[Aug10]) / close[Aug10]"
    )
    print(
        f"  = {our_adj[aug10_2017_idx]:.2f} × ({close_aug9:.2f} / {split_aug10} - {div_aug10:.2f}) / {close_aug10:.2f}"
    )
    print(f"  = {our_adj[aug10_2017_idx]:.2f} × {factor_to_aug9:.10f}")
    print(f"  = ${our_aug9_from_aug10:.6f}")
    print()

    # The KEY: What's the factor difference?
    quandl_aug9 = row_aug9["adj_close"][0]
    quandl_aug10 = row_aug10["adj_close"][0]

    quandl_ratio = quandl_aug9 / quandl_aug10
    our_ratio = our_adj[aug9_idx] / our_adj[aug10_2017_idx]

    print("=" * 80)
    print("KEY COMPARISON")
    print("=" * 80)
    print()
    print("Ratio of adj[Aug9] / adj[Aug10]:")
    print(f"  Quandl: {quandl_ratio:.10f}")
    print(f"  Ours:   {our_ratio:.10f}")
    print(f"  Difference: {abs(quandl_ratio - our_ratio):.10f}")
    print()

    # What's Quandl's factor?
    quandl_factor = quandl_aug9 / quandl_aug10

    print(f"Quandl's factor from Aug10 to Aug9: {quandl_factor:.10f}")
    print(f"Our factor from Aug10 to Aug9:      {factor_to_aug9:.10f}")
    print()

    # Reverse-engineer what Quandl is doing
    # quandl_factor = (close_aug9 / split - div_X) / close_aug10
    # So: div_X = (close_aug9 / split) - (quandl_factor * close_aug10)

    implied_div = (close_aug9 / split_aug10) - (quandl_factor * close_aug10)

    print("Reverse-engineering Quandl's formula:")
    print(f"  If Quandl uses factor = {quandl_factor:.10f}")
    print("  Then: factor = (close[Aug9] / split - div_X) / close[Aug10]")
    print("  Solving for div_X:")
    print("    div_X = (close[Aug9] / split) - (factor × close[Aug10])")
    print(
        f"    div_X = ({close_aug9:.2f} / {split_aug10}) - ({quandl_factor:.10f} × {close_aug10:.2f})"
    )
    print(f"    div_X = ${implied_div:.10f}")
    print()

    print(f"But Quandl has div[Aug10] = ${div_aug10:.2f}")
    print(f"Implied div from formula = ${implied_div:.10f}")
    print()

    # AHA! What if Quandl rounds the dividend adjustment?
    # Or maybe they're using a slightly different close price?

    # Check: What if Quandl rounds adj_close at each dividend?
    print("=" * 80)
    print("FINAL HYPOTHESIS")
    print("=" * 80)
    print()

    # The constant factor is 1.0002486210
    # This is (1 + 0.63/252.96) approximately
    # Where 252.96 ≈ close price around that time

    ratio_error = 1.0002486210
    implied_adjustment = (ratio_error - 1) * close_aug10

    print(f"Constant error factor: {ratio_error:.10f}")
    print(f"This equals: 1 + {(ratio_error - 1) * 100:.6f}%")
    print()
    print(
        f"If this were from dividend: {(ratio_error - 1):.10f} × {close_aug10:.2f} = ${implied_adjustment:.6f}"
    )
    print(f"Actual dividend on Aug 10: ${div_aug10:.2f}")
    print()

    # Check if it's a rounding issue
    # What if Quandl rounds adj_close to nearest cent at each dividend?
    adj_aug10_rounded = np.round(quandl_aug10, 2)
    factor_with_rounding = (close_aug9 / split_aug10 - div_aug10) / close_aug10
    adj_aug9_with_rounding = adj_aug10_rounded * factor_with_rounding

    print("With rounding adj_close to cents:")
    print(f"  adj[Aug10] rounded = ${adj_aug10_rounded:.2f}")
    print(f"  adj[Aug9] = ${adj_aug9_with_rounding:.6f}")
    print(f"  Quandl adj[Aug9] = ${quandl_aug9:.6f}")
    print(f"  Match: {abs(adj_aug9_with_rounding - quandl_aug9) < 0.001}")
    print()

    return {
        "aug9_idx": aug9_idx,
        "aug10_idx": aug10_2017_idx,
        "quandl_factor": quandl_factor,
        "our_factor": factor_to_aug9,
    }


if __name__ == "__main__":
    results = final_diagnosis()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The 0.02486% error is a systematic offset that originates from")
    print("the August 10, 2017 dividend calculation.")
    print()
    print("Quandl's backward adjustment produces a slightly different factor")
    print("for this dividend compared to our formula.")
    print()
    print("This constant factor then propagates backwards to all earlier dates,")
    print("causing the constant 0.02486% error we observe.")
    print()
    print("Possible causes:")
    print("1. Rounding convention difference in how dividends are applied")
    print("2. Different reference price used for dividend adjustment")
    print("3. Quandl may use a slightly modified formula for dividend events")
