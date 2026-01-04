"""Systematically test different adjustment formulas to reverse-engineer Quandl's methodology.

The goal: Match Quandl's adj_close column by figuring out their exact formula.
"""

import numpy as np
import polars as pl


def test_formula_1_current(df: pl.DataFrame) -> pl.DataFrame:
    """Current formula: (close[i] / split[i+1] - div[i+1]) / close[i+1]"""
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()
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

    return result.with_columns(pl.Series("test_adj_close", adj_vals))


def test_formula_2_separate_multiply(df: pl.DataFrame) -> pl.DataFrame:
    """Reference formula: Apply splits and dividends as separate multipliers.

    adj[i] = adj[i+1] * (1 / split[i+1]) * ((close[i+1] - div[i+1]) / close[i+1])
    """
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()
    adj_vals = close_vals.copy()

    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]

        # Separate multipliers
        split_multiplier = 1.0 / split_next if split_next > 0 else 1.0
        div_multiplier = (close_next - div_next) / close_next if close_next > 0 else 1.0

        adj_vals[i] = adj_vals[i + 1] * split_multiplier * div_multiplier

    return result.with_columns(pl.Series("test_adj_close", adj_vals))


def test_formula_3_use_raw_price(df: pl.DataFrame) -> pl.DataFrame:
    """Use raw price in numerator instead of adjusted price.

    adj[i] = adj[i+1] * (close[i] / split[i+1] - div[i+1]) / close[i+1]
    """
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()
    adj_vals = close_vals.copy()

    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        # Use raw close[i] in numerator
        adj_vals[i] = adj_vals[i + 1] * (close_today / split_next - div_next) / close_next

    return result.with_columns(pl.Series("test_adj_close", adj_vals))


def test_formula_4_split_on_today(df: pl.DataFrame) -> pl.DataFrame:
    """Apply split to today's close instead of tomorrow's.

    adj[i] = adj[i+1] * ((close[i] / split[i]) - div[i+1]) / close[i+1]
    """
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()
    adj_vals = close_vals.copy()

    for i in range(n - 2, -1, -1):
        split_today = split_vals[i]
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        # Apply split to today's close
        adj_vals[i] = adj_vals[i + 1] * (close_today / split_today - div_next) / close_next

    return result.with_columns(pl.Series("test_adj_close", adj_vals))


def test_formula_5_div_on_today(df: pl.DataFrame) -> pl.DataFrame:
    """Apply dividend on today instead of tomorrow.

    adj[i] = adj[i+1] * (close[i] / split[i+1] - div[i]) / close[i+1]
    """
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()
    adj_vals = close_vals.copy()

    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        div_today = div_vals[i]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        # Apply dividend from today
        adj_vals[i] = adj_vals[i + 1] * (close_today / split_next - div_today) / close_next

    return result.with_columns(pl.Series("test_adj_close", adj_vals))


def test_formula_6_no_split_in_numerator(df: pl.DataFrame) -> pl.DataFrame:
    """Don't divide by split in numerator, only as separate multiplier.

    adj[i] = adj[i+1] * (1/split[i+1]) * (close[i] - div[i+1]) / close[i+1]
    """
    result = df.clone()
    n = len(result)

    close_vals = result["close"].to_numpy()
    split_vals = result["split_ratio"].to_numpy()
    div_vals = result["ex-dividend"].to_numpy()
    adj_vals = close_vals.copy()

    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        # Split as separate multiplier, not in numerator
        split_multiplier = 1.0 / split_next if split_next > 0 else 1.0
        price_factor = (close_today - div_next) / close_next

        adj_vals[i] = adj_vals[i + 1] * split_multiplier * price_factor

    return result.with_columns(pl.Series("test_adj_close", adj_vals))


def calculate_error(test_adj: np.ndarray, actual_adj: np.ndarray) -> dict:
    """Calculate error metrics."""
    errors = np.abs((test_adj - actual_adj) / actual_adj) * 100

    return {
        "max_error_pct": errors.max(),
        "mean_error_pct": errors.mean(),
        "median_error_pct": np.median(errors),
        "pct_under_0.5": (errors < 0.5).sum() / len(errors) * 100,
    }


def test_all_formulas(symbol: str = "AAPL"):
    """Test all formula variations on a stock."""
    from pathlib import Path

    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    df = df.filter(pl.col("ticker") == symbol).sort("date")

    print(f"\n{'=' * 80}")
    print(f"Testing formulas on {symbol}")
    print(f"{'=' * 80}\n")

    formulas = [
        ("Current (with reverse split fix)", test_formula_1_current),
        ("Separate multipliers (split * dividend)", test_formula_2_separate_multiply),
        ("Use raw price in numerator", test_formula_3_use_raw_price),
        ("Split on today instead of tomorrow", test_formula_4_split_on_today),
        ("Dividend on today instead of tomorrow", test_formula_5_div_on_today),
        ("No split in numerator, separate multiplier", test_formula_6_no_split_in_numerator),
    ]

    actual_adj = df["adj_close"].to_numpy()

    results = []
    for name, formula_func in formulas:
        result_df = formula_func(df)
        test_adj = result_df["test_adj_close"].to_numpy()
        error_metrics = calculate_error(test_adj, actual_adj)

        results.append({"formula": name, **error_metrics})

    # Print results table
    print(f"{'Formula':<45} {'Max %':<10} {'Mean %':<10} {'<0.5%':<10}")
    print(f"{'-' * 80}")
    for r in results:
        print(
            f"{r['formula']:<45} {r['max_error_pct']:<10.4f} {r['mean_error_pct']:<10.6f} {r['pct_under_0.5']:<10.1f}%"
        )

    return results


def test_multiple_stocks():
    """Test on stocks with different characteristics."""
    stocks = [
        ("AAPL", "Normal 2:1 split + dividends"),
        ("ATI", "Reverse 1:2 split"),
        ("RIG", "Failed validation (45.9% error)"),
        ("DWSN", "Failed validation (89% error)"),
    ]

    all_results = {}
    for symbol, description in stocks:
        print(f"\n\n{symbol} - {description}")
        try:
            results = test_all_formulas(symbol)
            all_results[symbol] = results
        except Exception as e:
            print(f"Error testing {symbol}: {e}")

    return all_results


if __name__ == "__main__":
    results = test_multiple_stocks()
