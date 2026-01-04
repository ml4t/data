"""Test formulas that use consistent price types (all raw or all adjusted).

Hypothesis: Maybe we're mixing adjusted and raw prices incorrectly.
"""

from pathlib import Path

import numpy as np
import polars as pl


def pure_iterative_with_raw(df: pl.DataFrame) -> np.ndarray:
    """Use ONLY raw prices in the formula (no mixing with adjusted)."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    # Start with raw close at the end
    adj_vals = np.zeros(n)
    adj_vals[n - 1] = close_vals[n - 1]  # Last date: adj = raw

    for i in range(n - 2, -1, -1):
        close_i = close_vals[i]
        close_next = close_vals[i + 1]
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]

        # Calculate what adj[i] should be based on adj[i+1] and corporate actions
        # Standard formula: adj[i] = adj[i+1] * (close[i]/split - div) / close[i+1]
        adj_vals[i] = adj_vals[i + 1] * (close_i / split_next - div_next) / close_next

    return adj_vals


def forward_then_backward(df: pl.DataFrame) -> np.ndarray:
    """Calculate cumulative factors forward, then apply backward."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    # Calculate cumulative split factor (forward from oldest to newest)
    cumulative_split = np.ones(n)
    for i in range(1, n):
        cumulative_split[i] = cumulative_split[i - 1] * split_vals[i]

    # Split-adjusted prices
    split_adj = close_vals / cumulative_split

    # Now calculate dividend adjustment factors (backward)
    div_factors = np.ones(n)
    for i in range(n - 2, -1, -1):
        div_next = div_vals[i + 1]
        split_adj_next = split_adj[i + 1]

        if split_adj_next > 0:
            div_factors[i] = div_factors[i + 1] * ((split_adj_next - div_next) / split_adj_next)

    # Final adjusted prices
    return split_adj * div_factors


def exact_quandl_check(df: pl.DataFrame) -> np.ndarray:
    """Try to exactly replicate the pattern we see in Quandl adj factors.

    From examination: implied_adj_factor = adj_close / close
    Factor changes at splits and dividends.
    """
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()
    actual_adj = df["adj_close"].to_numpy()

    # Calculate the actual adjustment factors Quandl uses
    actual_adj / close_vals

    # Try to replicate those factors
    test_factors = np.ones(n)

    # Go backward
    for i in range(n - 2, -1, -1):
        # Start with tomorrow's factor
        test_factors[i] = test_factors[i + 1]

        # Check if there's a split tomorrow
        if split_vals[i + 1] != 1.0:
            test_factors[i] /= split_vals[i + 1]

        # Check if there's a dividend tomorrow
        if div_vals[i + 1] > 0:
            # What multiplier did Quandl use?
            # Try: (adj_close[i+1] - div) / adj_close[i+1]
            adj_next = actual_adj[i + 1]
            if adj_next > 0:
                test_factors[i] *= (adj_next - div_vals[i + 1]) / adj_next

    return close_vals * test_factors


def try_yahoo_formula(df: pl.DataFrame) -> np.ndarray:
    """Use Yahoo Finance's documented formula.

    According to many sources, Yahoo uses:
    adj[i] = adj[i+1] * (close[i] / close[i+1]) * (close[i+1] - div[i+1]) / close[i+1]

    Simplified: adj[i] = adj[i+1] * (close[i] - div[i+1]) / close[i+1]

    But with splits: adj[i] = adj[i+1] * (close[i] / split - div) / close[i+1]
    """
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    adj_vals = close_vals.copy()

    for i in range(n - 2, -1, -1):
        close_i = close_vals[i]
        close_next = close_vals[i + 1]
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]

        # Yahoo formula
        adj_vals[i] = adj_vals[i + 1] * ((close_i / split_next) - div_next) / close_next

    return adj_vals


def test_formula(symbol: str, formula_func, formula_name: str):
    """Test a formula."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    df = df.filter(pl.col("ticker") == symbol).sort("date")

    test_adj = formula_func(df)
    actual_adj = df["adj_close"].to_numpy()

    errors = np.abs((test_adj - actual_adj) / actual_adj) * 100
    max_error = errors.max()
    errors.mean()
    pct_under_half = (errors < 0.5).sum() / len(errors) * 100

    print(
        f"{symbol:8} | {formula_name:60} | Max: {max_error:8.4f}% | <0.5%: {pct_under_half:6.1f}%"
    )

    # For AAPL, show where the max error occurs
    if symbol == "AAPL" and max_error > 0.01:
        max_idx = errors.argmax()
        date_at_max = df["date"][max_idx]
        print(f"           Max error at index {max_idx}, date {date_at_max}")

    return max_error


def main():
    """Test price consistency approaches."""
    formulas = [
        ("Pure iterative with raw prices", pure_iterative_with_raw),
        ("Forward split, backward dividend", forward_then_backward),
        ("Exact Quandl factor replication", exact_quandl_check),
        ("Yahoo Finance formula", try_yahoo_formula),
    ]

    stocks = ["AAPL", "ATI", "RIG"]

    print(f"\n{'=' * 160}")
    print("Testing Price Consistency Approaches")
    print(f"{'=' * 160}\n")

    for formula_name, formula_func in formulas:
        print(f"\n{formula_name}:")
        print(f"{'-' * 160}")
        for symbol in stocks:
            try:
                test_formula(symbol, formula_func, formula_name)
            except Exception as e:
                print(f"{symbol:8} | ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
