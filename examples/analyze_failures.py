"""Analyze the 29% of stocks that fail validation.

What characteristics do failing stocks have?
- Reverse splits?
- Many splits?
- Long dividend history?
- Low prices?
"""

from pathlib import Path

import numpy as np
import polars as pl


def analyze_stock(df: pl.DataFrame) -> dict:
    """Analyze characteristics of a stock."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    # Count corporate actions
    num_splits = (split_vals != 1.0).sum()
    num_reverse_splits = (split_vals < 1.0).sum()
    num_dividends = (div_vals > 0).sum()

    # Price statistics
    min_price = close_vals.min()
    max_price = close_vals.max()
    mean_price = close_vals.mean()

    # Calculate error
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

    actual_adj = df["adj_close"].to_numpy()
    errors = np.abs((adj_vals - actual_adj) / actual_adj) * 100
    max_error = errors.max()
    pct_under_half = (errors < 0.5).sum() / len(errors) * 100

    return {
        "ticker": df["ticker"][0],
        "num_dates": n,
        "num_splits": int(num_splits),
        "num_reverse_splits": int(num_reverse_splits),
        "num_dividends": int(num_dividends),
        "min_price": float(min_price),
        "max_price": float(max_price),
        "mean_price": float(mean_price),
        "max_error": float(max_error),
        "pct_under_0.5": float(pct_under_half),
        "passes": pct_under_half > 99,
    }


def main():
    """Analyze failures vs successes."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)

    # Get 100 random tickers
    all_tickers = df["ticker"].unique().to_list()
    np.random.seed(42)
    selected_tickers = np.random.choice(all_tickers, size=min(100, len(all_tickers)), replace=False)

    results = []
    for ticker in selected_tickers:
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
        if len(ticker_df) < 100:
            continue

        try:
            result = analyze_stock(ticker_df)
            results.append(result)
        except Exception:
            pass

    # Convert to DataFrame for analysis
    results_df = pl.DataFrame(results)

    print("=" * 100)
    print("COMPARISON: PASSING vs FAILING STOCKS")
    print("=" * 100)
    print()

    passing = results_df.filter(pl.col("passes"))
    failing = results_df.filter(~pl.col("passes"))

    print(
        f"Passing stocks: {len(passing)}/{len(results_df)} ({len(passing) / len(results_df) * 100:.1f}%)"
    )
    print(
        f"Failing stocks: {len(failing)}/{len(results_df)} ({len(failing) / len(results_df) * 100:.1f}%)"
    )
    print()

    print("CHARACTERISTICS COMPARISON:")
    print("-" * 100)
    print(f"{'Metric':<30} {'Passing (median)':<25} {'Failing (median)':<25}")
    print("-" * 100)

    metrics = [
        ("Number of dates", "num_dates"),
        ("Number of splits", "num_splits"),
        ("Number of reverse splits", "num_reverse_splits"),
        ("Number of dividends", "num_dividends"),
        ("Min price", "min_price"),
        ("Max price", "max_price"),
        ("Mean price", "mean_price"),
        ("Max error %", "max_error"),
    ]

    for metric_name, col in metrics:
        passing_val = passing[col].median()
        failing_val = failing[col].median()
        print(f"{metric_name:<30} {passing_val:<25.2f} {failing_val:<25.2f}")

    print()
    print("REVERSE SPLITS:")
    print("-" * 100)
    passing_with_reverse = passing.filter(pl.col("num_reverse_splits") > 0)
    failing_with_reverse = failing.filter(pl.col("num_reverse_splits") > 0)

    print(
        f"Passing stocks with reverse splits: {len(passing_with_reverse)}/{len(passing)} ({len(passing_with_reverse) / len(passing) * 100:.1f}%)"
    )
    print(
        f"Failing stocks with reverse splits: {len(failing_with_reverse)}/{len(failing)} ({len(failing_with_reverse) / len(failing) * 100:.1f}%)"
    )

    print()
    print("TOP 10 WORST FAILURES:")
    print("-" * 100)
    worst = results_df.sort("max_error", descending=True).head(10)
    print(
        worst.select(
            [
                "ticker",
                "max_error",
                "num_splits",
                "num_reverse_splits",
                "num_dividends",
                "mean_price",
            ]
        )
    )

    print()
    print("SAMPLE OF PERFECT MATCHES (max error < 0.01%):")
    print("-" * 100)
    perfect = results_df.filter(pl.col("max_error") < 0.01).head(10)
    if len(perfect) > 0:
        print(
            perfect.select(
                [
                    "ticker",
                    "max_error",
                    "num_splits",
                    "num_reverse_splits",
                    "num_dividends",
                    "mean_price",
                ]
            )
        )
    else:
        print("No perfect matches found")


if __name__ == "__main__":
    main()
