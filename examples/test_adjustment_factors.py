"""Test if Quandl uses cumulative adjustment factors instead of iterative price calculation.

Key hypothesis: Quandl calculates:
1. adjustment_factor[t] = product of all corporate action multipliers from t to end
2. adj_close[t] = close[t] * adjustment_factor[t]

Instead of the iterative backward approach we've been using.
"""

from pathlib import Path

import numpy as np
import polars as pl


def calculate_adjustment_factors_v1(df: pl.DataFrame) -> np.ndarray:
    """Calculate adjustment factors as cumulative product from end to beginning.

    Standard approach from reference document:
    - For splits: multiply by (1/split_ratio)
    - For dividends: multiply by (close[i] - div[i]) / close[i]
    """
    n = len(df)

    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    # Start with factor = 1.0 at the end (most recent date)
    factors = np.ones(n)

    # Work backwards
    for i in range(n - 2, -1, -1):
        # Events happen at i+1 (tomorrow)
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]

        # Start with tomorrow's factor
        factors[i] = factors[i + 1]

        # Apply split adjustment
        if split_next != 1.0:
            factors[i] *= 1.0 / split_next

        # Apply dividend adjustment
        if div_next > 0 and close_next > 0:
            div_multiplier = (close_next - div_next) / close_next
            factors[i] *= div_multiplier

    return factors


def calculate_adjustment_factors_v2(df: pl.DataFrame) -> np.ndarray:
    """Try using adjusted close from next day in dividend calculation."""
    n = len(df)

    df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()
    adj_close_vals = df["adj_close"].to_numpy()

    factors = np.ones(n)

    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]
        adj_close_next = adj_close_vals[i + 1]

        factors[i] = factors[i + 1]

        if split_next != 1.0:
            factors[i] *= 1.0 / split_next

        if div_next > 0 and adj_close_next > 0:
            div_multiplier = (adj_close_next - div_next) / adj_close_next
            factors[i] *= div_multiplier

    return factors


def calculate_adjustment_factors_v3(df: pl.DataFrame) -> np.ndarray:
    """Use split-adjusted close for dividend calculation.

    Maybe Quandl:
    1. First calculates split-adjusted close
    2. Then uses split-adj close for dividend multiplier
    """
    n = len(df)

    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    # Calculate cumulative split factor
    cumulative_split = np.ones(n)
    for i in range(n - 2, -1, -1):
        cumulative_split[i] = cumulative_split[i + 1] * split_vals[i + 1]

    # Split-adjusted close
    split_adj_close = close_vals / cumulative_split

    # Now calculate dividend factors using split-adj close
    factors = np.ones(n) / cumulative_split  # Start with split adjustment

    for i in range(n - 2, -1, -1):
        div_next = div_vals[i + 1]
        split_adj_close_next = split_adj_close[i + 1]

        if div_next > 0 and split_adj_close_next > 0:
            div_multiplier = (split_adj_close_next - div_next) / split_adj_close_next
            factors[i] *= div_multiplier

    return factors


def test_factor_approach(symbol: str, formula_func, formula_name: str):
    """Test adjustment factor approach."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    df = df.filter(pl.col("ticker") == symbol).sort("date")

    # Calculate adjustment factors
    factors = formula_func(df)

    # Apply to get adjusted close
    close_vals = df["close"].to_numpy()
    test_adj_close = close_vals * factors

    # Compare to Quandl's adjusted close
    actual_adj = df["adj_close"].to_numpy()

    errors = np.abs((test_adj_close - actual_adj) / actual_adj) * 100
    max_error = errors.max()
    mean_error = errors.mean()
    pct_under_half = (errors < 0.5).sum() / len(errors) * 100

    print(
        f"{symbol:8} | {formula_name:50} | Max: {max_error:8.4f}% | Mean: {mean_error:10.6f}% | <0.5%: {pct_under_half:6.1f}%"
    )

    return max_error


def main():
    """Test adjustment factor approaches."""
    formulas = [
        ("Factors v1: (1/split) * (close-div)/close", calculate_adjustment_factors_v1),
        ("Factors v2: Use adj_close in div calc", calculate_adjustment_factors_v2),
        ("Factors v3: Split-adjusted close for div", calculate_adjustment_factors_v3),
    ]

    stocks = ["AAPL", "ATI", "RIG", "DWSN"]

    print(f"\n{'=' * 140}")
    print("Testing Adjustment Factor Approaches")
    print(f"{'=' * 140}\n")

    for formula_name, formula_func in formulas:
        print(f"\n{formula_name}:")
        print(f"{'-' * 140}")
        for symbol in stocks:
            try:
                test_factor_approach(symbol, formula_func, formula_name)
            except Exception as e:
                print(f"{symbol:8} | ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
