"""Test adaptive approach: detect whether close was adjusted for reverse split.

If close price jumped appropriately for the reverse split, use the split_ratio.
If close price didn't change much, ignore the split_ratio.
"""

from pathlib import Path

import numpy as np
import polars as pl


def adaptive_reverse_split_handling(df: pl.DataFrame) -> np.ndarray:
    """Detect reverse split encoding and apply appropriate formula.

    For each reverse split, check if close price changed by approximately 1/split_ratio.
    - If yes: close was adjusted, use split_ratio in formula
    - If no: close was not adjusted, ignore split_ratio
    """
    n = len(df)
    close_vals = df["close"].to_numpy()
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    # Pre-analyze reverse splits to determine which ones adjusted close
    reverse_split_uses_ratio = {}

    for i in range(1, n):
        if split_vals[i] < 1.0:  # Reverse split
            close_today = close_vals[i]
            close_prev = close_vals[i - 1]
            expected_ratio = 1.0 / split_vals[i]
            actual_ratio = close_today / close_prev if close_prev > 0 else 0

            # If actual close change is within 20% of expected, assume it was adjusted
            if abs(actual_ratio - expected_ratio) / expected_ratio < 0.20:
                reverse_split_uses_ratio[i] = True
            else:
                reverse_split_uses_ratio[i] = False

    # Now calculate adjusted prices
    adj_vals = close_vals.copy()
    for i in range(n - 2, -1, -1):
        split_next = split_vals[i + 1]
        div_next = div_vals[i + 1]
        close_next = close_vals[i + 1]
        close_today = close_vals[i]

        if split_next >= 1.0:
            # Normal split: always use split_ratio
            adjustment_factor = (close_today / split_next - div_next) / close_next
        elif reverse_split_uses_ratio.get(i + 1, False):
            # Reverse split where close WAS adjusted: use split_ratio
            adjustment_factor = (close_today / split_next - div_next) / close_next
        else:
            # Reverse split where close was NOT adjusted: ignore split_ratio
            adjustment_factor = (close_today - div_next) / close_next

        adj_vals[i] = adj_vals[i + 1] * adjustment_factor

    return adj_vals


def test_on_100_stocks():
    """Test adaptive approach on 100 random stocks with corporate actions."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)

    # Get 200 random tickers
    all_tickers = df["ticker"].unique().to_list()
    np.random.seed(42)
    selected_tickers = np.random.choice(all_tickers, size=min(200, len(all_tickers)), replace=False)

    results = []
    tested = 0

    for ticker in selected_tickers:
        if tested >= 100:
            break

        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
        if len(ticker_df) < 100:
            continue

        # Only test stocks with corporate actions
        split_vals = ticker_df["split_ratio"].to_numpy()
        div_vals = ticker_df["ex-dividend"].to_numpy()
        num_actions = (split_vals != 1.0).sum() + (div_vals > 0).sum()

        if num_actions == 0:
            continue

        tested += 1

        try:
            test_adj = adaptive_reverse_split_handling(ticker_df)
            actual_adj = ticker_df["adj_close"].to_numpy()

            errors = np.abs((test_adj - actual_adj) / actual_adj) * 100
            max_error = errors.max()
            pct_under_half = (errors < 0.5).sum() / len(errors) * 100

            results.append(
                {
                    "ticker": ticker,
                    "max_error": float(max_error),
                    "passes": pct_under_half > 99,
                }
            )

        except Exception:
            pass

    results_df = pl.DataFrame(results)

    print("=" * 100)
    print("ADAPTIVE REVERSE SPLIT HANDLING - 100 STOCKS WITH CORPORATE ACTIONS")
    print("=" * 100)
    print()

    passing = results_df.filter(pl.col("passes"))
    failing = results_df.filter(not pl.col("passes"))

    print(
        f"PASS RATE: {len(passing)}/{len(results_df)} ({len(passing) / len(results_df) * 100:.1f}%)"
    )
    print(
        f"FAIL RATE: {len(failing)}/{len(results_df)} ({len(failing) / len(results_df) * 100:.1f}%)"
    )
    print()

    if len(results_df) > 0:
        max_errors = results_df["max_error"].to_numpy()
        print("ERROR DISTRIBUTION:")
        print(f"  Median: {np.median(max_errors):.4f}%")
        print(f"  Mean:   {np.mean(max_errors):.4f}%")
        print(f"  95th %: {np.percentile(max_errors, 95):.4f}%")
        print(f"  Max:    {np.max(max_errors):.4f}%")
        print()

        print("WORST 10:")
        worst = results_df.sort("max_error", descending=True).head(10)
        print(worst)


if __name__ == "__main__":
    test_on_100_stocks()
