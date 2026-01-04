"""Investigate why our corporate actions algorithm shows 74% error on Yahoo data.

Test case: AAPL 2019-2021 (includes 2020-08-31 4:1 split)

Questions to answer:
1. Does Yahoo's Adj Close match our calculated adj_close?
2. Are corporate actions aligned properly?
3. Is there a methodology difference?
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.adjustments.core import apply_corporate_actions


def fetch_yahoo_data(symbol: str, start: str, end: str):
    """Fetch Yahoo data with both adjusted and unadjusted prices."""
    print(f"Fetching {symbol} from {start} to {end}...")

    ticker = yf.Ticker(symbol)

    # Get price data (unadjusted)
    df_prices = yf.download(
        symbol, start=start, end=end, auto_adjust=False, actions=False, progress=False
    )

    # Get corporate actions
    dividends = ticker.dividends
    splits = ticker.splits

    # Get adjusted prices from Yahoo
    df_adj = yf.download(
        symbol, start=start, end=end, auto_adjust=True, actions=False, progress=False
    )

    print(f"  Price data: {len(df_prices)} rows")
    print(f"  Dividends: {len(dividends)} events")
    print(f"  Splits: {len(splits)} events")

    return df_prices, df_adj, dividends, splits


def create_quandl_format(df_prices, dividends, splits):
    """Convert Yahoo data to Quandl schema format."""
    # Reset index to get dates as column
    df = df_prices.reset_index()

    # Yahoo returns: Date, Open, High, Low, Close, Adj Close, Volume
    # We want: date, open, high, low, close, volume (drop Adj Close)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]

    # Convert to Polars
    df_pl = pl.DataFrame(df)

    # Add corporate actions columns
    # Initialize with zeros
    df_pl = df_pl.with_columns([pl.lit(0.0).alias("ex-dividend"), pl.lit(1.0).alias("split_ratio")])

    # Filter corporate actions to match the date range in our data
    min_date = df_pl["date"].min()
    max_date = df_pl["date"].max()

    # Add dividends
    if len(dividends) > 0:
        # Filter to our date range
        mask = (dividends.index.date >= min_date.date()) & (dividends.index.date <= max_date.date())
        dividends_filtered = dividends[mask]

        div_dates = dividends_filtered.index.date
        div_amounts = dividends_filtered.values

        for div_date, div_amount in zip(div_dates, div_amounts):
            df_pl = df_pl.with_columns(
                [
                    pl.when(pl.col("date").cast(pl.Date) == pl.lit(div_date))
                    .then(pl.lit(float(div_amount)))
                    .otherwise(pl.col("ex-dividend"))
                    .alias("ex-dividend")
                ]
            )

    # Add splits
    if len(splits) > 0:
        # Filter to our date range
        mask = (splits.index.date >= min_date.date()) & (splits.index.date <= max_date.date())
        splits_filtered = splits[mask]

        split_dates = splits_filtered.index.date
        split_ratios = splits_filtered.values

        for split_date, split_ratio in zip(split_dates, split_ratios):
            df_pl = df_pl.with_columns(
                [
                    pl.when(pl.col("date").cast(pl.Date) == pl.lit(split_date))
                    .then(pl.lit(float(split_ratio)))
                    .otherwise(pl.col("split_ratio"))
                    .alias("split_ratio")
                ]
            )

    # Return the dataframe and filtered corporate actions
    if len(dividends) > 0:
        mask = (dividends.index.date >= min_date.date()) & (dividends.index.date <= max_date.date())
        dividends_filtered = dividends[mask]
    else:
        dividends_filtered = dividends

    if len(splits) > 0:
        mask = (splits.index.date >= min_date.date()) & (splits.index.date <= max_date.date())
        splits_filtered = splits[mask]
    else:
        splits_filtered = splits

    return df_pl, dividends_filtered, splits_filtered


def compare_adjustments(df_quandl_format, df_adj_yahoo):
    """Compare our calculated adjustments with Yahoo's."""
    print("\n" + "=" * 100)
    print("APPLYING OUR CORPORATE ACTIONS ALGORITHM")
    print("=" * 100)

    # Apply our algorithm
    result = apply_corporate_actions(df_quandl_format)

    # Get Yahoo's adjusted close (flatten to 1D)
    yahoo_adj = df_adj_yahoo["Close"].values.flatten()

    # Get our calculated adjusted close
    our_adj = result["adj_close"].to_numpy()

    # Compare
    errors = np.abs((our_adj - yahoo_adj) / yahoo_adj) * 100

    num_low_error = (errors < 0.5).sum()
    num_very_low_error = (errors < 0.01).sum()

    print("\nComparison with Yahoo's Adj Close:")
    print(f"  Max error: {errors.max():.4f}%")
    print(f"  Mean error: {errors.mean():.4f}%")
    print(f"  Median error: {np.median(errors):.4f}%")
    print(
        f"  Dates with <0.5% error: {num_low_error}/{len(errors)} ({num_low_error / len(errors) * 100:.1f}%)"
    )
    print(
        f"  Dates with <0.01% error: {num_very_low_error}/{len(errors)} ({num_very_low_error / len(errors) * 100:.1f}%)"
    )

    return result, yahoo_adj, errors


def analyze_around_split(df_quandl, result, yahoo_adj, split_date):
    """Analyze adjustments around the split date."""
    print("\n" + "=" * 100)
    print(f"ANALYSIS AROUND SPLIT DATE: {split_date}")
    print("=" * 100)

    # Get dates as list for indexing
    dates = df_quandl["date"].to_list()

    # Find split date index - compare date objects
    split_idx = None
    for i, d in enumerate(dates):
        # Convert to date if datetime
        d_date = d.date() if hasattr(d, "date") else d
        if d_date == split_date:
            split_idx = i
            break

    if split_idx is None:
        print(f"Split date {split_date} not found in data")
        print(f"Available dates: {dates[0]} to {dates[-1]}")
        return

    print(f"Split at index {split_idx}")

    # Show 5 days before and after
    start_idx = max(0, split_idx - 5)
    end_idx = min(len(df_quandl), split_idx + 6)

    close_vals = df_quandl["close"].to_numpy()
    our_adj_vals = result["adj_close"].to_numpy()
    yahoo_adj_vals = yahoo_adj  # Already numpy array
    split_vals = df_quandl["split_ratio"].to_numpy()
    div_vals = df_quandl["ex-dividend"].to_numpy()

    print(
        f"\n{'Date':<12} {'Close':>10} {'Split':>8} {'Div':>8} {'Our Adj':>12} {'Yahoo Adj':>12} {'Error %':>10}"
    )
    print("-" * 100)

    for i in range(start_idx, end_idx):
        date = dates[i]
        close = float(close_vals[i])
        split = float(split_vals[i])
        div = float(div_vals[i])
        our_adj = float(our_adj_vals[i])
        yahoo_adj_val = float(yahoo_adj_vals[i])
        error = abs((our_adj - yahoo_adj_val) / yahoo_adj_val) * 100

        marker = " <-- SPLIT" if i == split_idx else ""
        print(
            f"{str(date):<12} {close:>10.2f} {split:>8.4f} {div:>8.4f} {our_adj:>12.6f} {yahoo_adj_val:>12.6f} {error:>10.4f}%{marker}"
        )


def check_yahoo_methodology():
    """Test hypothesis: Does Yahoo apply adjustments differently?"""
    print("\n" + "=" * 100)
    print("CHECKING YAHOO'S ADJUSTMENT METHODOLOGY")
    print("=" * 100)

    # Key insight: In backward adjustment, the MOST RECENT date should have adj_close == close
    # Let's check if Yahoo follows this

    symbol = "AAPL"

    # Get recent data (no corporate actions recently)
    df_recent = yf.download(
        symbol,
        start="2024-01-01",
        end="2024-12-31",
        auto_adjust=False,
        actions=True,
        progress=False,
    )

    if len(df_recent) > 0:
        last_date = df_recent.index[-1]
        last_close = float(df_recent["Close"].iloc[-1])
        last_adj_close = float(df_recent["Adj Close"].iloc[-1])

        print(f"\nMost recent date: {last_date}")
        print(f"  Close: ${last_close:.6f}")
        print(f"  Adj Close: ${last_adj_close:.6f}")
        print(f"  Difference: ${abs(last_close - last_adj_close):.6f}")
        print(f"  Match? {abs(last_close - last_adj_close) < 0.01}")

        if abs(last_close - last_adj_close) < 0.01:
            print("\n✅ Yahoo DOES use backward adjustment (most recent date unadjusted)")
        else:
            print("\n❌ Yahoo does NOT use backward adjustment")
            print("   This could explain the mismatch!")


def main():
    """Run full investigation."""
    # Test case: AAPL with 2020 4:1 split
    symbol = "AAPL"
    start = "2019-01-01"
    end = "2021-12-31"

    # Fetch data
    df_prices, df_adj, dividends, splits = fetch_yahoo_data(symbol, start, end)

    # Convert to Quandl format (returns filtered corporate actions)
    df_quandl, dividends_filtered, splits_filtered = create_quandl_format(
        df_prices, dividends, splits
    )

    print("\n" + "=" * 100)
    print("CORPORATE ACTIONS SUMMARY (in our date range)")
    print("=" * 100)

    if len(dividends_filtered) > 0:
        print(f"\nDividends ({len(dividends_filtered)}):")
        print(dividends_filtered)
    else:
        print("\nNo dividends in date range")

    if len(splits_filtered) > 0:
        print(f"\nSplits ({len(splits_filtered)}):")
        print(splits_filtered)
    else:
        print("\nNo splits in date range")

    # Compare our adjustments with Yahoo's
    result, yahoo_adj, errors = compare_adjustments(df_quandl, df_adj)

    # Analyze around split
    if len(splits_filtered) > 0:
        split_date = splits_filtered.index[0].date()
        analyze_around_split(df_quandl, result, yahoo_adj, split_date)

    # Check Yahoo's methodology
    check_yahoo_methodology()

    print("\n" + "=" * 100)
    print("INVESTIGATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
