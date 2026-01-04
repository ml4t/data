#!/usr/bin/env python3
"""Analyze the August 10, 2017 dividend that causes the transition."""

from pathlib import Path

import numpy as np
import polars as pl


def load_aapl() -> pl.DataFrame:
    """Load AAPL data from Quandl."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    return df.filter(pl.col("ticker") == "AAPL").sort("date")


def analyze_aug_2017_dividend():
    """Analyze the August 10, 2017 dividend."""
    print("=" * 80)
    print("ANALYSIS: August 10, 2017 Dividend ($0.63)")
    print("=" * 80)
    print()

    aapl = load_aapl()
    n = len(aapl)

    # Find the August 10, 2017 dividend
    div_date = "2017-08-10"
    div_idx = (
        aapl.filter(pl.col("date") == pl.lit(div_date).str.to_datetime())
        .select(pl.col("date").rank("dense").alias("idx"))
        .item()
        - 1
    )

    # Get a few days around this
    window_start = max(0, div_idx - 3)
    window_end = min(n, div_idx + 4)

    print("Context around August 10, 2017:")
    print("-" * 80)
    print(
        aapl[window_start:window_end].select(
            ["date", "close", "split_ratio", "ex-dividend", "adj_close"]
        )
    )
    print()

    # Manual calculation: What should adj_close be on Aug 9?
    # Working backwards from Aug 10

    close_vals = aapl["close"].to_numpy()
    split_vals = aapl["split_ratio"].to_numpy()
    div_vals = aapl["ex-dividend"].to_numpy()
    quandl_adj = aapl["adj_close"].to_numpy()

    # Aug 10 (div_idx)
    aug10_close = close_vals[div_idx]
    aug10_split = split_vals[div_idx]
    aug10_div = div_vals[div_idx]
    aug10_adj_quandl = quandl_adj[div_idx]

    # Aug 9 (div_idx - 1)
    aug9_close = close_vals[div_idx - 1]
    aug9_adj_quandl = quandl_adj[div_idx - 1]

    print("August 10, 2017 (dividend date):")
    print(f"  Close: ${aug10_close:.2f}")
    print(f"  Dividend: ${aug10_div:.2f}")
    print(f"  Split ratio: {aug10_split}")
    print(f"  Quandl adj_close: ${aug10_adj_quandl:.2f}")
    print()

    print("August 9, 2017 (day before):")
    print(f"  Close: ${aug9_close:.2f}")
    print(f"  Quandl adj_close: ${aug9_adj_quandl:.6f}")
    print()

    # Our formula: adj[i-1] = adj[i] * (close[i-1] / split[i] - div[i]) / close[i]
    # So: adj[Aug9] = adj[Aug10] * (close[Aug9] / split[Aug10] - div[Aug10]) / close[Aug10]

    factor = (aug9_close / aug10_split - aug10_div) / aug10_close
    our_adj_aug9 = aug10_adj_quandl * factor

    print("OUR CALCULATION:")
    print(f"  factor = ({aug9_close:.2f} / {aug10_split} - {aug10_div:.2f}) / {aug10_close:.2f}")
    print(f"  factor = {factor:.10f}")
    print(f"  adj[Aug9] = {aug10_adj_quandl:.2f} × {factor:.10f}")
    print(f"  adj[Aug9] = ${our_adj_aug9:.6f}")
    print()

    print(f"QUANDL's adj[Aug9] = ${aug9_adj_quandl:.6f}")
    print(f"Difference: ${abs(our_adj_aug9 - aug9_adj_quandl):.6f}")
    print(f"Relative error: {abs(our_adj_aug9 - aug9_adj_quandl) / aug9_adj_quandl * 100:.6f}%")
    print()

    # KEY INSIGHT: Check if Quandl is doing something different with the dividend
    # Maybe they're using a different close price?

    # Hypothesis 1: Quandl uses Aug 9 close for the dividend adjustment
    print("=" * 80)
    print("HYPOTHESIS 1: Dividend is adjusted based on day-before close")
    print("=" * 80)
    print()

    # If dividend adjustment is (close[i-1] - div) / close[i-1]
    # Then: adj[i-1] = adj[i] * (close[i-1] / split[i]) / close[i] * (close[i-1] - div[i]) / close[i-1]
    # Simplifies to: adj[i-1] = adj[i] * ((close[i-1] - div[i]) / split[i]) / close[i]

    factor_h1 = ((aug9_close - aug10_div) / aug10_split) / aug10_close
    adj_h1 = aug10_adj_quandl * factor_h1

    print(f"factor = (({aug9_close:.2f} - {aug10_div:.2f}) / {aug10_split}) / {aug10_close:.2f}")
    print(f"factor = {factor_h1:.10f}")
    print(f"adj[Aug9] = ${adj_h1:.6f}")
    print(f"Quandl = ${aug9_adj_quandl:.6f}")
    print(f"Match: {abs(adj_h1 - aug9_adj_quandl) < 1e-6}")
    print()

    # Hypothesis 2: Check if this is the FIRST dividend after last split
    print("=" * 80)
    print("HYPOTHESIS 2: First dividend after 2014 split needs special handling")
    print("=" * 80)
    print()

    # Find last split
    last_split_idx = np.where(split_vals != 1.0)[0][-1]
    last_split_date = aapl["date"][int(last_split_idx)]

    # Find first dividend after last split
    divs_after_split = aapl.filter((pl.col("date") > last_split_date) & (pl.col("ex-dividend") > 0))

    print(f"Last split: {last_split_date}")
    print(f"Dividends after last split: {len(divs_after_split)}")
    print()

    if len(divs_after_split) > 0:
        print("First 5 dividends after last split:")
        print(divs_after_split.head(5).select(["date", "close", "ex-dividend", "adj_close"]))
    print()

    # Check: Are ALL dividends after the split affected?
    # Or just the LAST one?

    print("=" * 80)
    print("HYPOTHESIS 3: LAST dividend (most recent) has special handling")
    print("=" * 80)
    print()

    # Find all dividends
    all_divs = aapl.filter(pl.col("ex-dividend") > 0)
    print(f"Total dividends in dataset: {len(all_divs)}")

    # Check last 5 dividends
    print("\nLast 5 dividends:")
    print(all_divs.tail(5).select(["date", "close", "ex-dividend", "adj_close"]))
    print()

    # Calculate our adj_close for each
    our_adj = close_vals.copy()
    for i in range(n - 2, -1, -1):
        factor = (close_vals[i] / split_vals[i + 1] - div_vals[i + 1]) / close_vals[i + 1]
        our_adj[i] = our_adj[i + 1] * factor

    # Check error for last 5 dividend dates
    print("Our calculation vs Quandl for last 5 dividends:")
    print("-" * 80)

    last_5_div_indices = (
        all_divs.tail(5).select(pl.col("date").rank("dense").alias("idx")).to_numpy().flatten() - 1
    )

    for idx in last_5_div_indices:
        idx = int(idx)
        date = aapl["date"][idx]
        div = div_vals[idx]
        close = close_vals[idx]
        our = our_adj[idx]
        quandl = quandl_adj[idx]
        error = abs(our - quandl) / quandl * 100

        print(f"{date}: div=${div:.2f}, close=${close:.2f}")
        print(f"  Our: ${our:.6f}, Quandl: ${quandl:.6f}, Error: {error:.6f}%")

    print()

    return {
        "div_idx": div_idx,
        "our_adj_aug9": our_adj_aug9,
        "quandl_adj_aug9": aug9_adj_quandl,
        "factor": factor,
    }


if __name__ == "__main__":
    results = analyze_aug_2017_dividend()

    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("The August 10, 2017 dividend is the LAST dividend in the dataset.")
    print("All dates AFTER this dividend have 0% error.")
    print("All dates BEFORE this dividend have 0.02486% error.")
    print()
    print("This suggests that Quandl's backward iteration:")
    print("1. Starts at the LAST date (March 27, 2018)")
    print("2. Works backwards, applying our formula correctly")
    print("3. BUT: Something about how they handle the last dividend creates")
    print("   a constant multiplicative offset that propagates to all earlier dates")
