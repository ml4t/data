"""Test formulas with proper rounding for financial calculations.

Money should be rounded to cents (2 decimal places).
Test on 100 random stocks to get error distribution.
"""

from pathlib import Path

import numpy as np
import polars as pl


def formula_no_rounding(df: pl.DataFrame) -> np.ndarray:
    """Current formula without rounding."""
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
            adjustment_factor = (close_today - div_next) / close_next

        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def formula_round_each_step(df: pl.DataFrame) -> np.ndarray:
    """Round adjusted price at each iteration to 2 decimal places."""
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
            adjustment_factor = (close_today - div_next) / close_next

        adj_vals[i] = np.round(adj_vals[i + 1] * adjustment_factor, 2)

    return adj_vals


def formula_round_intermediate(df: pl.DataFrame) -> np.ndarray:
    """Round intermediate calculations to 2 decimals."""
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

        # Round intermediate values
        if split_next >= 1.0:
            numerator = np.round(close_today / split_next - div_next, 2)
        else:
            numerator = np.round(close_today - div_next, 2)

        adjustment_factor = numerator / close_next
        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def formula_round_4_decimals(df: pl.DataFrame) -> np.ndarray:
    """Maybe Quandl uses 4 decimal places for intermediate calculations."""
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
            adjustment_factor = (close_today - div_next) / close_next

        adj_vals[i] = np.round(adj_vals[i + 1] * adjustment_factor, 4)

    return adj_vals


def formula_round_6_decimals(df: pl.DataFrame) -> np.ndarray:
    """Test with 6 decimal places."""
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
            adjustment_factor = (close_today - div_next) / close_next

        adj_vals[i] = np.round(adj_vals[i + 1] * adjustment_factor, 6)

    return adj_vals


def test_formula_on_stock(df: pl.DataFrame, formula_func) -> dict:
    """Test a formula on one stock."""
    test_adj = formula_func(df)
    actual_adj = df["adj_close"].to_numpy()

    errors = np.abs((test_adj - actual_adj) / actual_adj) * 100
    max_error = errors.max()
    mean_error = errors.mean()
    median_error = np.median(errors)

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "median_error": median_error,
        "pct_under_0.5": (errors < 0.5).sum() / len(errors) * 100,
        "pct_under_0.01": (errors < 0.01).sum() / len(errors) * 100,
    }


def test_on_100_random_stocks():
    """Test formulas on 100 random stocks."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)

    # Get list of all tickers
    all_tickers = df["ticker"].unique().to_list()
    print(f"Total tickers in dataset: {len(all_tickers)}")

    # Randomly select 100
    np.random.seed(42)
    selected_tickers = np.random.choice(all_tickers, size=min(100, len(all_tickers)), replace=False)

    formulas = [
        ("No rounding", formula_no_rounding),
        ("Round to 2 decimals each step", formula_round_each_step),
        ("Round intermediate to 2 decimals", formula_round_intermediate),
        ("Round to 4 decimals each step", formula_round_4_decimals),
        ("Round to 6 decimals each step", formula_round_6_decimals),
    ]

    # Store results for each formula
    formula_results = {name: [] for name, _ in formulas}

    print(f"\nTesting on {len(selected_tickers)} random stocks...\n")

    for ticker in selected_tickers:
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")

        # Skip stocks with too little data
        if len(ticker_df) < 100:
            continue

        for formula_name, formula_func in formulas:
            try:
                result = test_formula_on_stock(ticker_df, formula_func)
                formula_results[formula_name].append(result)
            except Exception:
                # Skip stocks that cause errors
                pass

    # Print summary statistics for each formula
    print("=" * 140)
    print("SUMMARY STATISTICS ACROSS ~100 RANDOM STOCKS")
    print("=" * 140)
    print()

    for formula_name, _ in formulas:
        results = formula_results[formula_name]
        if not results:
            continue

        max_errors = [r["max_error"] for r in results]
        mean_errors = [r["mean_error"] for r in results]
        pct_under_half = [r["pct_under_0.5"] for r in results]
        pct_under_001 = [r["pct_under_0.01"] for r in results]

        print(f"\n{formula_name}:")
        print(f"  Stocks tested: {len(results)}")
        print("  Max error across all stocks:")
        print(f"    Min:    {np.min(max_errors):.6f}%")
        print(f"    Median: {np.median(max_errors):.6f}%")
        print(f"    Mean:   {np.mean(max_errors):.6f}%")
        print(f"    Max:    {np.max(max_errors):.6f}%")
        print(f"    95th %: {np.percentile(max_errors, 95):.6f}%")
        print("  ")
        print("  Mean error across all stocks:")
        print(f"    Median: {np.median(mean_errors):.6f}%")
        print(f"    Mean:   {np.mean(mean_errors):.6f}%")
        print("  ")
        print(
            f"  Stocks with >99% dates under 0.5% error: {sum(1 for p in pct_under_half if p > 99)}/{len(results)} ({sum(1 for p in pct_under_half if p > 99) / len(results) * 100:.1f}%)"
        )
        print(
            f"  Stocks with >99% dates under 0.01% error: {sum(1 for p in pct_under_001 if p > 99)}/{len(results)} ({sum(1 for p in pct_under_001 if p > 99) / len(results) * 100:.1f}%)"
        )

    print()
    print("=" * 140)


if __name__ == "__main__":
    test_on_100_random_stocks()
