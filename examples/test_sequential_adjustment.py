"""Test if Quandl applies splits and dividends SEQUENTIALLY rather than combined.

Based on the reference document, the standard approach is:
1. Calculate split-adjusted prices
2. Then apply dividend adjustments to the split-adjusted prices

This is different from our combined formula.
"""

from pathlib import Path

import numpy as np
import polars as pl


def apply_splits_first_then_dividends(df: pl.DataFrame) -> pl.DataFrame:
    """Apply splits first to get split-adjusted prices, then apply dividends.

    This follows the reference document's guidance:
    - Line 90: P_adj[t] = P_raw[t] / ratio
    - Line 311: P_adj[t] *= (P_raw[t_ex] - D) / P_raw[t_ex]
    """
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()

    # Step 1: Apply splits only (backward)
    split_adj = close_vals.copy()
    for i in range(n - 2, -1, -1):
        split_vals[i + 1]
        # Backward split adjustment: multiply by cumulative split ratios
        # Or: split_adj[i] = split_adj[i+1] * (close[i] / close[i+1]) / split_next

        # Actually, let me think about this differently...
        # If we're going backwards and a split happens at i+1:
        # The split-adjusted price at i should be: split_adj[i] = close[i] / split_next
        # NO, that's forward adjustment.

        # For backward adjustment, we need cumulative splits
        pass

    # Actually, let me use cumulative split ratios
    # Cumulative split from each date forward
    cumulative_split = np.ones(n)
    for i in range(n - 2, -1, -1):
        cumulative_split[i] = cumulative_split[i + 1] * split_vals[i + 1]

    split_adj = close_vals / cumulative_split

    # Step 2: Apply dividends to split-adjusted prices
    total_adj = split_adj.copy()
    for i in range(n - 2, -1, -1):
        div_next = div_vals[i + 1]
        split_adj_next = split_adj[i + 1]

        if split_adj_next > 0:
            div_multiplier = (split_adj_next - div_next) / split_adj_next
            total_adj[i] = total_adj[i + 1] * div_multiplier
        else:
            total_adj[i] = total_adj[i + 1]

    return result.with_columns(pl.Series("test_adj_close", total_adj))


def apply_using_raw_prices_for_div(df: pl.DataFrame) -> pl.DataFrame:
    """Apply dividends using RAW prices (not split-adjusted) in the multiplier.

    Maybe Quandl calculates: (close_raw[i+1] - div) / close_raw[i+1]
    instead of using split-adjusted prices.
    """
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()

    # Cumulative split adjustment
    cumulative_split = np.ones(n)
    for i in range(n - 2, -1, -1):
        cumulative_split[i] = cumulative_split[i + 1] * split_vals[i + 1]

    # Apply both split and dividend together, but use RAW close in dividend calc
    adj_vals = close_vals.copy()
    for i in range(n - 2, -1, -1):
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]  # RAW close

        # Split adjustment
        1.0 / cumulative_split[i]

        # Dividend multiplier using RAW prices
        div_multiplier = (close_next - div_next) / close_next if close_next > 0 else 1.0

        # Combined
        adj_vals[i] = (
            adj_vals[i + 1] * div_multiplier * (cumulative_split[i + 1] / cumulative_split[i])
        )

    return result.with_columns(pl.Series("test_adj_close", adj_vals))


def iterative_backward_adjustment(df: pl.DataFrame) -> pl.DataFrame:
    """Pure iterative backward adjustment from reference document.

    The key is that we go backwards one day at a time, applying:
    - adj[i] = adj[i+1] * adjustment_factor

    Where adjustment_factor accounts for both splits and dividends.
    """
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()

    adj_vals = close_vals.copy()

    for i in range(n - 2, -1, -1):
        # Tomorrow's values (used to calculate today's adjustment)
        close_i = close_vals[i]
        close_next = close_vals[i + 1]
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]

        # Standard formula from many sources:
        # adj[i] = adj[i+1] * (close[i] - div_next) / (close[i+1] / split_next)
        # Simplified: adj[i] = adj[i+1] * (close[i] - div_next) * split_next / close[i+1]

        if close_next > 0:
            adjustment_factor = (close_i - div_next) * split_next / close_next
            adj_vals[i] = adj_vals[i + 1] * adjustment_factor
        else:
            adj_vals[i] = adj_vals[i + 1]

    return result.with_columns(pl.Series("test_adj_close", adj_vals))


def test_formula(symbol: str, formula_func, formula_name: str):
    """Test a single formula on a stock."""

    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    df = df.filter(pl.col("ticker") == symbol).sort("date")

    result_df = formula_func(df)
    test_adj = result_df["test_adj_close"].to_numpy()
    actual_adj = df["adj_close"].to_numpy()

    errors = np.abs((test_adj - actual_adj) / actual_adj) * 100

    max_error = errors.max()
    mean_error = errors.mean()
    pct_under_half = (errors < 0.5).sum() / len(errors) * 100

    print(
        f"{symbol:8} | {formula_name:40} | Max: {max_error:8.4f}% | Mean: {mean_error:10.6f}% | <0.5%: {pct_under_half:6.1f}%"
    )

    return max_error, mean_error


def main():
    """Test new formulas on all problem stocks."""
    formulas = [
        ("Splits first, then dividends", apply_splits_first_then_dividends),
        ("Raw prices for dividend calc", apply_using_raw_prices_for_div),
        ("Iterative: (close[i]-div)*split/close[i+1]", iterative_backward_adjustment),
    ]

    stocks = ["AAPL", "ATI", "RIG", "DWSN"]

    print(f"\n{'=' * 120}")
    print("Testing Sequential Adjustment Approaches")
    print(f"{'=' * 120}\n")

    for formula_name, formula_func in formulas:
        print(f"\n{formula_name}:")
        print(f"{'-' * 120}")
        for symbol in stocks:
            try:
                test_formula(symbol, formula_func, formula_name)
            except Exception as e:
                print(f"{symbol:8} | ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
