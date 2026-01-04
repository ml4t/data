"""Re-analyze but ONLY include stocks that actually have corporate actions.

This is the real test - stocks without corporate actions are trivial.
"""

from pathlib import Path

import numpy as np
import polars as pl


def test_stock(df: pl.DataFrame) -> dict:
    """Test formula on a stock and return results."""
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    # Count corporate actions
    num_splits = (split_vals != 1.0).sum()
    num_dividends = (div_vals > 0).sum()
    total_actions = num_splits + num_dividends

    # Calculate adjusted prices
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
    mean_error = errors.mean()
    pct_under_half = (errors < 0.5).sum() / len(errors) * 100

    return {
        "ticker": df["ticker"][0],
        "num_dates": n,
        "num_splits": int(num_splits),
        "num_dividends": int(num_dividends),
        "total_actions": int(total_actions),
        "max_error": float(max_error),
        "mean_error": float(mean_error),
        "pct_under_0.5": float(pct_under_half),
        "passes": pct_under_half > 99,
    }


def main():
    """Test ONLY on stocks with corporate actions."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)

    # Get 200 random tickers (to ensure we get 100 with corporate actions)
    all_tickers = df["ticker"].unique().to_list()
    np.random.seed(42)
    selected_tickers = np.random.choice(all_tickers, size=min(200, len(all_tickers)), replace=False)

    results = []
    for ticker in selected_tickers:
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
        if len(ticker_df) < 100:
            continue

        try:
            result = test_stock(ticker_df)

            # ONLY include stocks with corporate actions
            if result["total_actions"] > 0:
                results.append(result)

            # Stop once we have 100 stocks with actions
            if len(results) >= 100:
                break

        except Exception:
            pass

    results_df = pl.DataFrame(results)

    print("=" * 100)
    print(f"ANALYSIS: STOCKS WITH CORPORATE ACTIONS ONLY (n={len(results_df)})")
    print("=" * 100)
    print()

    # Overall pass rate
    passing = results_df.filter(pl.col("passes"))
    failing = results_df.filter(not pl.col("passes"))

    print(
        f"PASS RATE: {len(passing)}/{len(results_df)} ({len(passing) / len(results_df) * 100:.1f}%)"
    )
    print(
        f"FAIL RATE: {len(failing)}/{len(results_df)} ({len(failing) / len(results_df) * 100:.1f}%)"
    )
    print()

    # Break down by number of corporate actions
    print("PASS RATE BY NUMBER OF CORPORATE ACTIONS:")
    print("-" * 100)

    buckets = [
        ("1-5 actions", 1, 5),
        ("6-10 actions", 6, 10),
        ("11-20 actions", 11, 20),
        ("21-50 actions", 21, 50),
        ("51-100 actions", 51, 100),
        ("100+ actions", 101, 10000),
    ]

    for bucket_name, min_actions, max_actions in buckets:
        bucket_df = results_df.filter(
            (pl.col("total_actions") >= min_actions) & (pl.col("total_actions") <= max_actions)
        )

        if len(bucket_df) > 0:
            bucket_passing = bucket_df.filter(pl.col("passes"))
            pass_rate = len(bucket_passing) / len(bucket_df) * 100
            median_error = bucket_df["max_error"].median()

            print(
                f"{bucket_name:<20} n={len(bucket_df):<4} Pass: {len(bucket_passing)}/{len(bucket_df)} ({pass_rate:5.1f}%)  Median max error: {median_error:.4f}%"
            )

    print()
    print("STATISTICS BY CORPORATE ACTION TYPE:")
    print("-" * 100)

    # Stocks with splits
    with_splits = results_df.filter(pl.col("num_splits") > 0)
    with_splits_passing = with_splits.filter(pl.col("passes"))
    print(
        f"Stocks with splits:    {len(with_splits_passing)}/{len(with_splits)} pass ({len(with_splits_passing) / len(with_splits) * 100:.1f}%)"
    )

    # Stocks with dividends only (no splits)
    divs_only = results_df.filter((pl.col("num_dividends") > 0) & (pl.col("num_splits") == 0))
    divs_only_passing = divs_only.filter(pl.col("passes"))
    print(
        f"Dividends only:        {len(divs_only_passing)}/{len(divs_only)} pass ({len(divs_only_passing) / len(divs_only) * 100:.1f}%)"
    )

    # Stocks with both
    both = results_df.filter((pl.col("num_dividends") > 0) & (pl.col("num_splits") > 0))
    both_passing = both.filter(pl.col("passes"))
    print(
        f"Both splits & divs:    {len(both_passing)}/{len(both)} pass ({len(both_passing) / len(both) * 100:.1f}%)"
    )

    print()
    print("ERROR DISTRIBUTION:")
    print("-" * 100)
    max_errors = results_df["max_error"].to_numpy()
    print(f"Min:        {np.min(max_errors):.6f}%")
    print(f"25th %ile:  {np.percentile(max_errors, 25):.6f}%")
    print(f"Median:     {np.median(max_errors):.6f}%")
    print(f"75th %ile:  {np.percentile(max_errors, 75):.6f}%")
    print(f"95th %ile:  {np.percentile(max_errors, 95):.6f}%")
    print(f"Max:        {np.max(max_errors):.6f}%")

    print()
    print("BEST 10 (lowest max error):")
    print("-" * 100)
    best = results_df.sort("max_error").head(10)
    print(best.select(["ticker", "max_error", "num_splits", "num_dividends", "total_actions"]))

    print()
    print("WORST 10 (highest max error):")
    print("-" * 100)
    worst = results_df.sort("max_error", descending=True).head(10)
    print(worst.select(["ticker", "max_error", "num_splits", "num_dividends", "total_actions"]))


if __name__ == "__main__":
    main()
