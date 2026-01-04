#!/usr/bin/env python3
"""Analyze the worst-performing stocks from validation."""

from pathlib import Path

import numpy as np
import polars as pl

from ml4t.data.adjustments import apply_corporate_actions


def load_ticker(ticker: str) -> pl.DataFrame:
    """Load data for a specific ticker."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    return df.filter(pl.col("ticker") == ticker).sort("date")


def analyze_stock(ticker: str):
    """Analyze a specific stock."""
    print("=" * 80)
    print(f"ANALYZING: {ticker}")
    print("=" * 80)
    print()

    df = load_ticker(ticker)
    if len(df) == 0:
        print(f"No data found for {ticker}")
        return None

    n = len(df)
    print(f"Records: {n:,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Corporate actions
    splits = df.filter(pl.col("split_ratio") != 1.0)
    dividends = df.filter(pl.col("ex-dividend") > 0)

    print(f"Splits: {len(splits)}")
    if len(splits) > 0:
        print(splits.select(["date", "close", "split_ratio", "adj_close"]))
    print()

    print(f"Dividends: {len(dividends)}")
    if len(dividends) > 0:
        print("  First 5:")
        print(dividends.head(5).select(["date", "close", "ex-dividend", "adj_close"]))
        if len(dividends) > 5:
            print(f"  ... ({len(dividends) - 5} more)")
    print()

    # Calculate our adjusted close
    adjusted = apply_corporate_actions(df, price_cols=["close"])

    our_adj = adjusted["adj_close"].to_numpy()
    quandl_adj = df["adj_close"].to_numpy()

    # Calculate error
    error = np.abs(our_adj - quandl_adj) / quandl_adj * 100

    max_error = error.max()
    mean_error = error.mean()
    max_error_idx = int(error.argmax())

    print("Validation Results:")
    print(f"  Max error: {max_error:.4f}%")
    print(f"  Mean error: {mean_error:.4f}%")
    print()

    # Show where max error occurs
    max_error_row = df[max_error_idx]
    print(f"Max error at index {max_error_idx} ({max_error_row['date'][0]}):")
    print(f"  Close: ${max_error_row['close'][0]:.2f}")
    print(f"  Our adj_close: ${our_adj[max_error_idx]:.6f}")
    print(f"  Quandl adj_close: ${quandl_adj[max_error_idx]:.6f}")
    print()

    # Check corporate actions around max error
    window_start = max(0, max_error_idx - 10)
    window_end = min(n, max_error_idx + 10)

    print("Context (10 days before/after max error):")
    context = df[window_start:window_end]
    print(context.select(["date", "close", "split_ratio", "ex-dividend", "adj_close"]))
    print()

    # Check for reverse splits
    reverse_splits = splits.filter(pl.col("split_ratio") < 1.0)
    if len(reverse_splits) > 0:
        print(f"⚠️  REVERSE SPLITS FOUND: {len(reverse_splits)}")
        print(reverse_splits.select(["date", "close", "split_ratio", "adj_close"]))
        print()

    # Check for unusual dividend/price ratios
    if len(dividends) > 0:
        div_data = dividends.with_columns(
            [(pl.col("ex-dividend") / pl.col("close") * 100).alias("div_yield_pct")]
        )

        high_yield_divs = div_data.filter(pl.col("div_yield_pct") > 10)
        if len(high_yield_divs) > 0:
            print(f"⚠️  UNUSUAL DIVIDENDS (>10% yield): {len(high_yield_divs)}")
            print(high_yield_divs.select(["date", "close", "ex-dividend", "div_yield_pct"]))
            print()

    return {
        "ticker": ticker,
        "max_error": max_error,
        "mean_error": mean_error,
        "splits": len(splits),
        "reverse_splits": len(reverse_splits),
        "dividends": len(dividends),
    }


def main():
    """Analyze all worst stocks."""
    worst_stocks = ["DWSN", "ATI", "RIG", "HHS", "ACXM"]

    results = []
    for ticker in worst_stocks:
        result = analyze_stock(ticker)
        if result:
            results.append(result)
        print("\n\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(
        f"{'Ticker':<8} {'Max Error':>12} {'Mean Error':>12} {'Splits':>8} {'Rev Splits':>12} {'Dividends':>10}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['ticker']:<8} {r['max_error']:>11.2f}% {r['mean_error']:>11.4f}% {r['splits']:>8} {r['reverse_splits']:>12} {r['dividends']:>10}"
        )

    print()
    print("OBSERVATIONS:")
    print()

    # Pattern analysis
    has_reverse = [r for r in results if r["reverse_splits"] > 0]
    no_reverse = [r for r in results if r["reverse_splits"] == 0]

    print(f"Stocks with reverse splits: {len(has_reverse)}")
    if has_reverse:
        avg_error_reverse = np.mean([r["max_error"] for r in has_reverse])
        print(f"  Average max error: {avg_error_reverse:.2f}%")

    print(f"Stocks without reverse splits: {len(no_reverse)}")
    if no_reverse:
        avg_error_no_reverse = np.mean([r["max_error"] for r in no_reverse])
        print(f"  Average max error: {avg_error_no_reverse:.2f}%")

    print()


if __name__ == "__main__":
    main()
