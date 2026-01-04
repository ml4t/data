#!/usr/bin/env python3
"""Find the source of the constant 0.02486% offset."""

from pathlib import Path

import numpy as np
import polars as pl


def load_aapl() -> pl.DataFrame:
    """Load AAPL data from Quandl."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    return df.filter(pl.col("ticker") == "AAPL").sort("date")


def find_constant_offset():
    """Find where the constant offset originates."""
    print("=" * 80)
    print("FINDING SOURCE OF CONSTANT 0.02486% OFFSET")
    print("=" * 80)
    print()

    aapl = load_aapl()
    n = len(aapl)

    # Calculate our adjusted close
    close_vals = aapl["close"].to_numpy()
    split_vals = aapl["split_ratio"].to_numpy()
    div_vals = aapl["ex-dividend"].to_numpy()
    quandl_adj = aapl["adj_close"].to_numpy()

    our_adj = close_vals.copy()
    for i in range(n - 2, -1, -1):
        factor = (close_vals[i] / split_vals[i + 1] - div_vals[i + 1]) / close_vals[i + 1]
        our_adj[i] = our_adj[i + 1] * factor

    # Calculate error
    error = np.abs(our_adj - quandl_adj) / quandl_adj * 100

    # Find where error changes
    print("1. ERROR PROFILE ANALYSIS")
    print("-" * 80)
    print(f"Min error: {error.min():.10f}%")
    print(f"Max error: {error.max():.10f}%")
    print(f"Mean error: {error.mean():.10f}%")
    print(f"Std error: {error.std():.10f}%")
    print()

    # Find where error transitions
    # Is it constant, then drops to zero?
    zero_error_mask = error < 1e-10
    high_error_mask = error > 0.01

    zero_count = zero_error_mask.sum()
    high_count = high_error_mask.sum()

    print(f"Records with ~0% error: {zero_count:,} ({zero_count / n * 100:.1f}%)")
    print(f"Records with >0.01% error: {high_count:,} ({high_count / n * 100:.1f}%)")
    print()

    # Find the transition point
    if zero_count > 0 and high_count > 0:
        # Find first zero error
        first_zero_idx = np.where(zero_error_mask)[0][0]
        last_high_idx = np.where(high_error_mask)[0][-1]

        print(f"First record with ~0% error: index {first_zero_idx}")
        print(f"Last record with >0.01% error: index {last_high_idx}")
        print()

        # Show transition zone
        transition_start = max(0, last_high_idx - 5)
        transition_end = min(n, first_zero_idx + 5)

        print("TRANSITION ZONE:")
        print("-" * 80)
        print(
            f"{'Index':<6} {'Date':<12} {'Close':>8} {'Split':>6} {'Dividend':>8} {'Error %':>10}"
        )
        print("-" * 80)

        for i in range(transition_start, transition_end):
            date = str(aapl["date"][i])[:10]
            close = close_vals[i]
            split = split_vals[i]
            div = div_vals[i]
            err = error[i]

            marker = ""
            if i == last_high_idx:
                marker = " <<< LAST HIGH"
            elif i == first_zero_idx:
                marker = " <<< FIRST ZERO"

            print(f"{i:<6} {date:<12} {close:>8.2f} {split:>6.2f} {div:>8.2f} {err:>9.6f}%{marker}")

        print()

        # Check what corporate action happened at transition
        print("2. CORPORATE ACTION AT TRANSITION")
        print("-" * 80)

        # Find corporate actions around transition
        search_start = max(0, last_high_idx - 10)
        search_end = min(n, first_zero_idx + 10)

        corp_actions = aapl[search_start:search_end].filter(
            (pl.col("split_ratio") != 1.0) | (pl.col("ex-dividend") > 0)
        )

        if len(corp_actions) > 0:
            print("Corporate actions near transition:")
            print(corp_actions.select(["date", "close", "split_ratio", "ex-dividend", "adj_close"]))
        else:
            print("No corporate actions found near transition")
        print()

        # The key question: Is the error related to a specific split or dividend?
        # Let's check the LAST corporate action before the most recent date
        print("3. MOST RECENT CORPORATE ACTION (working backwards from end)")
        print("-" * 80)

        # Find last split
        last_split_idx = np.where(split_vals != 1.0)[0]
        if len(last_split_idx) > 0:
            last_split_idx = last_split_idx[-1]
            last_split = aapl[int(last_split_idx)]
            print("Last split:")
            print(f"  Date: {last_split['date'][0]}")
            print(f"  Split ratio: {last_split['split_ratio'][0]}")
            print(f"  Close: ${last_split['close'][0]:.2f}")
            print(f"  Adj close: ${last_split['adj_close'][0]:.2f}")
            print(f"  Error at this point: {error[last_split_idx]:.6f}%")
            print()

        # Find last dividend
        last_div_idx = np.where(div_vals > 0)[0]
        if len(last_div_idx) > 0:
            last_div_idx = last_div_idx[-1]
            last_div = aapl[int(last_div_idx)]
            print("Last dividend:")
            print(f"  Date: {last_div['date'][0]}")
            print(f"  Dividend: ${last_div['ex-dividend'][0]:.2f}")
            print(f"  Close: ${last_div['close'][0]:.2f}")
            print(f"  Adj close: ${last_div['adj_close'][0]:.2f}")
            print(f"  Error at this point: {error[last_div_idx]:.6f}%")
            print()

        # Hypothesis: The offset might be from the LAST event affecting ALL prior dates
        # Let's manually calculate what the ratio should be
        print("4. HYPOTHESIS: Mismatch in Last Corporate Action Handling")
        print("-" * 80)

        # Calculate what the cumulative adjustment should be at first date
        first_close = close_vals[0]
        first_adj_quandl = quandl_adj[0]
        first_adj_ours = our_adj[0]

        total_factor_quandl = first_adj_quandl / first_close
        total_factor_ours = first_adj_ours / first_close

        print(f"First date ({aapl['date'][0]}):")
        print(f"  Close: ${first_close:.2f}")
        print(f"  Quandl adj_close: ${first_adj_quandl:.6f}")
        print(f"  Our adj_close: ${first_adj_ours:.6f}")
        print()
        print("Total adjustment factors:")
        print(f"  Quandl: {total_factor_quandl:.10f}x")
        print(f"  Ours: {total_factor_ours:.10f}x")
        print(f"  Ratio: {total_factor_ours / total_factor_quandl:.10f}")
        print()

        # This ratio should tell us what constant factor we're off by
        ratio = total_factor_ours / total_factor_quandl
        print(f"We are off by a constant factor of: {ratio:.10f}")
        print(f"Percentage difference: {(ratio - 1) * 100:.10f}%")
        print()

        return {
            "last_high_idx": last_high_idx,
            "first_zero_idx": first_zero_idx,
            "constant_ratio": ratio,
        }

    return None


if __name__ == "__main__":
    results = find_constant_offset()

    if results:
        print("=" * 80)
        print("KEY FINDING")
        print("=" * 80)
        print()
        print(f"There is a constant multiplicative factor of {results['constant_ratio']:.10f}")
        print(f"that affects all dates from index 0 to {results['last_high_idx']}")
        print()
        print("This suggests:")
        print("1. A corporate action near the transition is being handled slightly differently")
        print("2. OR: Quandl's reference price at the end differs from ours by this factor")
        print("3. OR: There's a rounding convention we're not matching")
