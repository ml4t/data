"""Final validation of the improved corporate actions algorithm.

Tests on 100 random stocks with corporate actions and reports:
- Pass rate
- Error distribution
- Examples of best and worst cases
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.adjustments.core import apply_corporate_actions


def validate_stock(ticker: str, df_full: pl.DataFrame) -> dict:
    """Validate one stock."""
    df = df_full.filter(pl.col("ticker") == ticker).sort("date")

    if len(df) < 100:
        return None

    # Count corporate actions
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()
    num_splits = (split_vals != 1.0).sum()
    num_dividends = (div_vals > 0).sum()
    total_actions = num_splits + num_dividends

    # Skip stocks without corporate actions
    if total_actions == 0:
        return None

    # Apply our algorithm
    result = apply_corporate_actions(df)

    # Compare to Quandl's adjusted prices
    our_adj = result["adj_close"].to_numpy()
    quandl_adj = df["adj_close"].to_numpy()

    errors = np.abs((our_adj - quandl_adj) / quandl_adj) * 100
    max_error = errors.max()
    mean_error = errors.mean()
    median_error = np.median(errors)
    pct_under_0_5 = (errors < 0.5).sum() / len(errors) * 100
    pct_under_0_01 = (errors < 0.01).sum() / len(errors) * 100

    return {
        "ticker": ticker,
        "num_dates": len(df),
        "num_splits": int(num_splits),
        "num_dividends": int(num_dividends),
        "total_actions": int(total_actions),
        "max_error": float(max_error),
        "mean_error": float(mean_error),
        "median_error": float(median_error),
        "pct_under_0.5": float(pct_under_0_5),
        "pct_under_0.01": float(pct_under_0_01),
        "passes": pct_under_0_5 > 99,
    }


def main():
    """Run full validation."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)

    # Get 200 random tickers (to ensure we get 100 with corporate actions)
    all_tickers = df["ticker"].unique().to_list()
    np.random.seed(42)
    selected_tickers = np.random.choice(all_tickers, size=min(200, len(all_tickers)), replace=False)

    print("Validating corporate actions algorithm on 100 random stocks with corporate actions...")
    print()

    results = []
    for ticker in selected_tickers:
        if len(results) >= 100:
            break

        try:
            result = validate_stock(ticker, df)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if not results:
        print("No stocks with corporate actions found!")
        return

    results_df = pl.DataFrame(results)

    print("=" * 100)
    print("FINAL VALIDATION RESULTS")
    print("=" * 100)
    print()

    passing = results_df.filter(pl.col("passes"))
    failing = results_df.filter(not pl.col("passes"))

    print(f"Stocks tested: {len(results_df)}")
    print(
        f"PASS RATE: {len(passing)}/{len(results_df)} ({len(passing) / len(results_df) * 100:.1f}%)"
    )
    print(
        f"FAIL RATE: {len(failing)}/{len(results_df)} ({len(failing) / len(results_df) * 100:.1f}%)"
    )
    print()

    # Error statistics
    max_errors = results_df["max_error"].to_numpy()
    print("ERROR DISTRIBUTION (max error per stock):")
    print(f"  Minimum:     {np.min(max_errors):.6f}%")
    print(f"  25th %ile:   {np.percentile(max_errors, 25):.6f}%")
    print(f"  Median:      {np.median(max_errors):.6f}%")
    print(f"  75th %ile:   {np.percentile(max_errors, 75):.6f}%")
    print(f"  95th %ile:   {np.percentile(max_errors, 95):.6f}%")
    print(f"  Maximum:     {np.max(max_errors):.6f}%")
    print()

    # Pass rate by number of corporate actions
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
    print("BEST 10 (lowest max error):")
    print("-" * 100)
    best = results_df.sort("max_error").head(10)
    print(best.select(["ticker", "max_error", "num_splits", "num_dividends"]))

    print()
    print("WORST 10 (highest max error):")
    print("-" * 100)
    worst = results_df.sort("max_error", descending=True).head(10)
    print(worst.select(["ticker", "max_error", "num_splits", "num_dividends"]))

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
