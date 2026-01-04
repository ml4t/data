"""Examine how Quandl actually encodes split_ratio and ex-dividend.

Maybe we're misinterpreting what these columns mean.
"""

from pathlib import Path

import polars as pl


def examine_aapl_split():
    """Examine AAPL's 2014 7-for-1 split in detail."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    aapl = df.filter(pl.col("ticker") == "AAPL").sort("date")

    # AAPL had a 7-for-1 split on June 9, 2014
    # Let's look at dates around that
    split_window = aapl.filter(
        (pl.col("date") >= pl.lit("2014-06-04").str.to_datetime())
        & (pl.col("date") <= pl.lit("2014-06-11").str.to_datetime())
    )

    print("=" * 100)
    print("AAPL 7-for-1 Split (June 9, 2014)")
    print("=" * 100)
    print(split_window.select(["date", "close", "adj_close", "split_ratio", "ex-dividend"]))
    print()

    # Check: did close price drop by 7x?
    june6 = split_window.filter(pl.col("date") == pl.lit("2014-06-06").str.to_datetime())
    june9 = split_window.filter(pl.col("date") == pl.lit("2014-06-09").str.to_datetime())

    close_before = june6["close"][0]
    close_after = june9["close"][0]
    split_ratio_on_june9 = june9["split_ratio"][0]

    print(f"Close before split (June 6): ${close_before:.2f}")
    print(f"Close after split (June 9): ${close_after:.2f}")
    print(f"Ratio: {close_before / close_after:.4f}")
    print(f"split_ratio value on June 9: {split_ratio_on_june9}")
    print()

    # Check: what should the formula be?
    adj_june6 = june6["adj_close"][0]
    adj_june9 = june9["adj_close"][0]

    print(f"Adj close June 6: ${adj_june6:.6f}")
    print(f"Adj close June 9: ${adj_june9:.6f}")
    print(f"Adj ratio: {adj_june6 / adj_june9:.6f}")
    print()


def examine_ati_reverse_split():
    """Examine ATI's reverse split."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    ati = df.filter(pl.col("ticker") == "ATI").sort("date")

    # Find the reverse split
    splits = ati.filter(pl.col("split_ratio") != 1.0)
    print("=" * 100)
    print("ATI Splits")
    print("=" * 100)
    print(splits.select(["date", "close", "adj_close", "split_ratio"]))
    print()

    # Get dates around the reverse split
    if len(splits) > 0:
        split_date = splits["date"][0]
        print(f"\nReverse split date: {split_date}")

        # Get window around split
        split_window = ati.filter(
            (pl.col("date") >= split_date - pl.duration(days=5))
            & (pl.col("date") <= split_date + pl.duration(days=5))
        )

        print("\nDates around reverse split:")
        print(split_window.select(["date", "close", "adj_close", "split_ratio", "ex-dividend"]))


def examine_dividends():
    """Examine how dividends are encoded."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    aapl = df.filter(pl.col("ticker") == "AAPL").sort("date")

    # Find dividend dates
    dividends = aapl.filter(pl.col("ex-dividend") > 0).tail(10)

    print("=" * 100)
    print("AAPL Last 10 Dividends")
    print("=" * 100)
    print(dividends.select(["date", "close", "adj_close", "ex-dividend"]))
    print()

    # For each dividend, check if close dropped by dividend amount
    print("Checking if close price dropped by dividend amount on ex-date:\n")
    for i in range(len(dividends)):
        div_date = dividends["date"][i]
        div_amount = dividends["ex-dividend"][i]
        close_on_exdate = dividends["close"][i]

        # Get previous trading day
        prev_dates = aapl.filter(pl.col("date") < div_date).sort("date").tail(1)
        if len(prev_dates) > 0:
            close_prev = prev_dates["close"][0]
            actual_drop = close_prev - close_on_exdate
            expected_drop = div_amount

            print(
                f"{div_date}: Div=${div_amount:.2f}, Prev close=${close_prev:.2f}, Ex close=${close_on_exdate:.2f}"
            )
            print(
                f"  Expected drop: ${expected_drop:.2f}, Actual drop: ${actual_drop:.2f}, Diff: ${actual_drop - expected_drop:.2f}"
            )
            print()


def check_adjustment_factor_hypothesis():
    """Check if Quandl provides adjustment factors directly."""
    quandl_path = Path.home() / "ml3t/data/equities/quandl/wiki_prices.parquet"
    df = pl.read_parquet(quandl_path)
    aapl = df.filter(pl.col("ticker") == "AAPL").sort("date")

    # Calculate what the adjustment factor SHOULD be
    # adj_close = close * adj_factor
    # So: adj_factor = adj_close / close

    aapl = aapl.with_columns([(pl.col("adj_close") / pl.col("close")).alias("implied_adj_factor")])

    # Look at how this changes over time
    print("=" * 100)
    print("Implied Adjustment Factors (adj_close / close)")
    print("=" * 100)

    # Show around the 2014 split
    split_window = aapl.filter(
        (pl.col("date") >= pl.lit("2014-06-04").str.to_datetime())
        & (pl.col("date") <= pl.lit("2014-06-11").str.to_datetime())
    )
    print("\nAround 2014 split:")
    print(split_window.select(["date", "close", "adj_close", "implied_adj_factor", "split_ratio"]))
    print()

    # Show recent dates (should be close to 1.0)
    print("\nLast 10 dates:")
    print(aapl.tail(10).select(["date", "close", "adj_close", "implied_adj_factor", "ex-dividend"]))
    print()


if __name__ == "__main__":
    examine_aapl_split()
    print("\n")
    examine_ati_reverse_split()
    print("\n")
    examine_dividends()
    print("\n")
    check_adjustment_factor_hypothesis()
