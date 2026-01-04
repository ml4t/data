"""Test different reverse split handling approaches.

The current approach of ignoring split_ratio for reverse splits is clearly wrong.
"""

from pathlib import Path

import numpy as np
import polars as pl


def current_broken_approach(df: pl.DataFrame) -> np.ndarray:
    """Current approach that ignores split_ratio for reverse splits."""
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

        if split_next >= 1.0:
            adjustment_factor = (close_today / split_next - div_next) / close_next
        else:
            # BROKEN: ignores split_ratio
            adjustment_factor = (close_today - div_next) / close_next

        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def use_split_ratio_for_reverse_too(df: pl.DataFrame) -> np.ndarray:
    """Use the SAME formula for all splits (normal and reverse)."""
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

        # SAME formula for all splits
        adjustment_factor = (close_today / split_next - div_next) / close_next
        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def multiply_by_split_for_reverse(df: pl.DataFrame) -> np.ndarray:
    """For reverse splits, multiply instead of divide."""
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

        if split_next >= 1.0:
            # Normal split: divide
            adjustment_factor = (close_today / split_next - div_next) / close_next
        else:
            # Reverse split: multiply
            adjustment_factor = (close_today * split_next - div_next) / close_next

        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def test_formula(ticker: str, formula_func, formula_name: str):
    """Test a formula on one stock."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    df = df.filter(pl.col("ticker") == ticker).sort("date")

    test_adj = formula_func(df)
    actual_adj = df["adj_close"].to_numpy()

    errors = np.abs((test_adj - actual_adj) / actual_adj) * 100
    max_error = errors.max()
    mean_error = errors.mean()

    print(f"{ticker:8} | {formula_name:50} | Max: {max_error:8.4f}% | Mean: {mean_error:8.4f}%")

    return max_error


def main():
    """Test on stocks with reverse splits."""
    formulas = [
        ("Current (broken): ignore split for reverse", current_broken_approach),
        ("Same formula for all splits", use_split_ratio_for_reverse_too),
        ("Multiply by split for reverse splits", multiply_by_split_for_reverse),
    ]

    # Test on stocks with reverse splits that are currently failing
    test_stocks = [
        ("ANIP", "3 reverse splits, 99% error"),
        ("INSM", "2 reverse splits, 97.5% error"),
        ("ATI", "1 reverse split, was 104% error"),
        ("AAPL", "1 normal split (control)"),
    ]

    print("=" * 120)
    print("TESTING REVERSE SPLIT HANDLING")
    print("=" * 120)
    print()

    for ticker, description in test_stocks:
        print(f"\n{ticker} - {description}:")
        print("-" * 120)
        for formula_name, formula_func in formulas:
            try:
                test_formula(ticker, formula_func, formula_name)
            except Exception as e:
                print(f"{ticker:8} | {formula_name:50} | ERROR: {e}")


if __name__ == "__main__":
    main()
