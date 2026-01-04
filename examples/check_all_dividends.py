#!/usr/bin/env python3
"""Check all dividends for the pattern."""

from pathlib import Path

import numpy as np
import polars as pl


def check_all_dividends():
    """Check all dividends."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    aapl = df.filter(pl.col("ticker") == "AAPL").sort("date")

    len(aapl)
    close_vals = aapl["close"].to_numpy()
    aapl["split_ratio"].to_numpy()
    div_vals = aapl["ex-dividend"].to_numpy()
    adj_vals = aapl["adj_close"].to_numpy()

    # Get all dividends
    div_indices = np.where(div_vals > 0)[0]

    print(f"Total dividends: {len(div_indices)}")
    print()

    # Check last 10 dividends
    print("LAST 10 DIVIDENDS:")
    print("=" * 120)
    print(
        f"{'Date':<12} {'Div':>6} {'Close[i]':>9} {'Close[i-1]':>9} {'Adj[i]':>11} {'Adj[i-1] Q':>11} {'Adj[i-1] Us':>11} {'Error %':>10}"
    )
    print("=" * 120)

    for idx in div_indices[-10:]:
        if idx == 0:
            continue

        prev_idx = idx - 1

        date = str(aapl["date"][int(idx)])[:10]
        div = div_vals[idx]
        close_div = close_vals[idx]
        close_prev = close_vals[prev_idx]
        adj_div = adj_vals[idx]
        adj_prev_q = adj_vals[prev_idx]

        # Our calculation
        factor = (close_prev / 1.0 - div) / close_div
        adj_prev_us = adj_div * factor

        error = abs(adj_prev_us - adj_prev_q) / adj_prev_q * 100

        print(
            f"{date:<12} ${div:>5.2f} ${close_div:>8.2f} ${close_prev:>8.2f} ${adj_div:>10.2f} ${adj_prev_q:>10.6f} ${adj_prev_us:>10.6f} {error:>9.6f}%"
        )

    print()


if __name__ == "__main__":
    check_all_dividends()
