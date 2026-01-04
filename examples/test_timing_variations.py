"""Test different timing assumptions for when splits and dividends apply.

Maybe:
- split_ratio[i] means split applies at date[i]
- ex-dividend[i] means dividend applies at date[i]
- Or they apply at different times in the calculation
"""

from pathlib import Path

import numpy as np
import polars as pl


def timing_v1_both_at_next(df: pl.DataFrame) -> np.ndarray:
    """Both split and dividend apply at [i+1] (our current approach)."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    adj_vals = close_vals.copy()
    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        adjustment_factor = (close_today / split_next - div_next) / close_next
        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def timing_v2_split_today_div_next(df: pl.DataFrame) -> np.ndarray:
    """Split applies at [i], dividend at [i+1]."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    adj_vals = close_vals.copy()
    for i in range(n - 2, -1, -1):
        split_today = split_vals[i]
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        adjustment_factor = (close_today / split_today - div_next) / close_next
        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def timing_v3_both_at_today(df: pl.DataFrame) -> np.ndarray:
    """Both split and dividend apply at [i]."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    adj_vals = close_vals.copy()
    for i in range(n - 2, -1, -1):
        split_today = split_vals[i]
        div_today = div_vals[i]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        adjustment_factor = (close_today / split_today - div_today) / close_next
        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def timing_v4_div_today_split_next(df: pl.DataFrame) -> np.ndarray:
    """Dividend applies at [i], split at [i+1]."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    adj_vals = close_vals.copy()
    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        div_today = div_vals[i]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        adjustment_factor = (close_today / split_next - div_today) / close_next
        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def timing_v5_explicit_sep_factors(df: pl.DataFrame) -> np.ndarray:
    """Calculate split and dividend as completely separate backward passes."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    # Pass 1: Apply splits only
    split_adj = close_vals.copy()
    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        # Just the split part
        split_factor = close_today / (split_next * close_next)
        split_adj[i] = split_adj[i + 1] * split_factor

    # Pass 2: Apply dividends to split-adjusted prices
    total_adj = split_adj.copy()
    for i in range(n - 2, -1, -1):
        div_next = div_vals[i + 1]
        split_adj_next = split_adj[i + 1]
        split_adj_today = split_adj[i]

        # Dividend adjustment
        if split_adj_next > 0:
            div_factor = (split_adj_today - div_next) / split_adj_next
            total_adj[i] = total_adj[i + 1] * div_factor

    return total_adj


def test_timing(symbol: str, timing_func, timing_name: str):
    """Test a timing variation."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    df = df.filter(pl.col("ticker") == symbol).sort("date")

    test_adj = timing_func(df)
    actual_adj = df["adj_close"].to_numpy()

    errors = np.abs((test_adj - actual_adj) / actual_adj) * 100
    max_error = errors.max()
    mean_error = errors.mean()
    pct_under_half = (errors < 0.5).sum() / len(errors) * 100

    print(
        f"{symbol:8} | {timing_name:50} | Max: {max_error:8.4f}% | Mean: {mean_error:10.6f}% | <0.5%: {pct_under_half:6.1f}%"
    )

    return max_error


def main():
    """Test timing variations."""
    timings = [
        ("v1: Both at [i+1] (current)", timing_v1_both_at_next),
        ("v2: Split[i], Div[i+1]", timing_v2_split_today_div_next),
        ("v3: Both at [i]", timing_v3_both_at_today),
        ("v4: Div[i], Split[i+1]", timing_v4_div_today_split_next),
        ("v5: Two separate backward passes", timing_v5_explicit_sep_factors),
    ]

    stocks = ["AAPL", "ATI", "RIG", "DWSN"]

    print(f"\n{'=' * 140}")
    print("Testing Different Timing Assumptions")
    print(f"{'=' * 140}\n")

    for timing_name, timing_func in timings:
        print(f"\n{timing_name}:")
        print(f"{'-' * 140}")
        for symbol in stocks:
            try:
                test_timing(symbol, timing_func, timing_name)
            except Exception as e:
                print(f"{symbol:8} | ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
