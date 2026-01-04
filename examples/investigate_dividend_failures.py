"""Investigate stocks with extreme dividend counts that are failing validation.

CBS: 48 dividends, 262% error
NL: 105 dividends, 97% error
PBI: 119 dividends, 91% error
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.adjustments.core import apply_corporate_actions


def investigate_stock(ticker: str):
    """Deep dive into one failing stock."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df_full = pl.read_parquet(quandl_path)

    df = df_full.filter(pl.col("ticker") == ticker).sort("date")

    print("=" * 100)
    print(f"INVESTIGATING {ticker}")
    print("=" * 100)
    print()

    # Basic stats
    print(f"Total dates: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Corporate actions
    split_vals = df["split_ratio"].to_numpy()
    div_vals = df["ex-dividend"].to_numpy()

    num_splits = (split_vals != 1.0).sum()
    num_dividends = (div_vals > 0).sum()

    print(f"Number of splits: {num_splits}")
    print(f"Number of dividends: {num_dividends}")
    print()

    # Look at dividends
    dividends = df.filter(pl.col("ex-dividend") > 0)
    print("Dividend statistics:")
    print(f"  Min: ${dividends['ex-dividend'].min():.4f}")
    print(f"  Max: ${dividends['ex-dividend'].max():.4f}")
    print(f"  Mean: ${dividends['ex-dividend'].mean():.4f}")
    print()

    # Apply our algorithm
    result = apply_corporate_actions(df)
    our_adj = result["adj_close"].to_numpy()
    quandl_adj = df["adj_close"].to_numpy()

    errors = np.abs((our_adj - quandl_adj) / quandl_adj) * 100

    print("Our algorithm results:")
    print(f"  Max error: {errors.max():.4f}%")
    print(f"  Mean error: {errors.mean():.4f}%")
    print(f"  Median error: {np.median(errors):.4f}%")
    print()

    # Find where error is worst
    worst_idx = int(errors.argmax())
    print(f"Worst error at index {worst_idx}, date {df['date'][worst_idx]}:")
    print(f"  Our adj_close: ${our_adj[worst_idx]:.6f}")
    print(f"  Quandl adj_close: ${quandl_adj[worst_idx]:.6f}")
    print(f"  Raw close: ${df['close'][worst_idx]:.6f}")
    print(f"  Error: {errors[worst_idx]:.4f}%")
    print()

    # Check if error is at beginning or end
    first_100_errors = errors[:100]
    last_100_errors = errors[-100:]

    print("Error progression:")
    print(
        f"  First 100 dates - max: {first_100_errors.max():.4f}%, mean: {first_100_errors.mean():.4f}%"
    )
    print(
        f"  Last 100 dates - max: {last_100_errors.max():.4f}%, mean: {last_100_errors.mean():.4f}%"
    )
    print()

    # Check the first few dividends in detail
    print("First 5 dividends in detail:")
    print("-" * 100)

    divs = df.filter(pl.col("ex-dividend") > 0).head(5)
    for row in divs.iter_rows(named=True):
        date = row["date"]
        div = row["ex-dividend"]
        close = row["close"]
        adj_close = row["adj_close"]

        # Get index in full dataframe
        idx = int(
            df.filter(pl.col("date") == date).select(pl.int_range(0, pl.len())).to_numpy()[0][0]
        )

        # Get previous day
        if idx > 0:
            prev_close = df["close"][idx - 1]
            prev_adj = quandl_adj[idx - 1]
            our_prev_adj = our_adj[idx - 1]

            # Calculate expected adjustment multiplier
            expected_multiplier = (close - div) / close

            # What Quandl actually did
            quandl_multiplier = adj_close / prev_adj if prev_adj > 0 else 0

            # What we calculated
            our_multiplier = our_adj[idx] / our_prev_adj if our_prev_adj > 0 else 0

            print(f"Date: {date}, Dividend: ${div:.4f}")
            print(f"  Close: ${close:.2f}, Prev close: ${prev_close:.2f}")
            print(f"  Expected multiplier: {expected_multiplier:.8f}")
            print(f"  Quandl multiplier: {quandl_multiplier:.8f}")
            print(f"  Our multiplier: {our_multiplier:.8f}")
            print(f"  Difference: {abs(quandl_multiplier - our_multiplier):.10f}")
            print()

    # Check for any special patterns in the data
    print("Checking for patterns:")
    print("-" * 100)

    # Are there any dates where adj_close == close (no adjustment)?
    no_adj = df.filter(pl.col("adj_close") == pl.col("close"))
    print(f"Dates where adj_close == close: {len(no_adj)}")
    if len(no_adj) > 0:
        print(f"  First: {no_adj['date'].min()}")
        print(f"  Last: {no_adj['date'].max()}")
    print()

    # Check the implied adjustment factor
    implied_factor = quandl_adj / df["close"].to_numpy()
    print("Implied adjustment factor (adj_close / close):")
    print(f"  Min: {implied_factor.min():.6f}")
    print(f"  Max: {implied_factor.max():.6f}")
    print(f"  At start: {implied_factor[0]:.6f}")
    print(f"  At end: {implied_factor[-1]:.6f}")
    print()


def compare_three_stocks():
    """Compare the three worst dividend-heavy stocks."""
    for ticker in ["CBS", "NL", "PBI"]:
        investigate_stock(ticker)
        print("\n\n")


if __name__ == "__main__":
    compare_three_stocks()
