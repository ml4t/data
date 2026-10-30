#!/usr/bin/env python3
"""Assign CME equity futures sessions to illustrative bars or a supplied Parquet file.

Run from the repository with ``uv run python examples/nasdaq_bars_sessions.py``.
The default CSV contains hand-authored teaching data, not historical market data.
See docs/user-guide/sessions.md for input provenance and timestamp conventions.
"""

import argparse
from pathlib import Path

import polars as pl

from ml4t.data.sessions import SessionAssigner
from ml4t.specs import FeedSpec

SAMPLE_PATH = Path(__file__).resolve().parent / "data/nasdaq_bars_sample.csv"
BAR_COLUMNS = ["datetime", "Open", "High", "Low", "Last", "Volume", "# of Trades"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="External Parquet bars; default: illustrative CSV"
    )
    parser.add_argument("--output", type=Path, default=Path("nasdaq_bars_with_sessions.parquet"))
    parser.add_argument(
        "--timezone", help="Source timezone, required for naive external timestamps"
    )
    parser.add_argument(
        "--bar-label",
        choices=["close", "open"],
        default="close",
        help="Timestamp side (default: close)",
    )
    args = parser.parse_args()
    if args.input is not None and args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")

    if args.input is None:
        print("Using illustrative bars: no vendor, real contract, or aggregation threshold.")
        bars = pl.read_csv(SAMPLE_PATH, try_parse_dates=True)
    else:
        bars = pl.read_parquet(args.input)
    missing = set(BAR_COLUMNS) - set(bars.columns)
    if missing:
        parser.error(f"Input is missing columns: {', '.join(sorted(missing))}")
    dtype = bars.schema["datetime"]
    if not isinstance(dtype, pl.Datetime) or bars.is_empty() or bars["datetime"].null_count():
        parser.error("datetime must contain non-null Datetime values in a nonempty table")
    if dtype.time_zone is None and args.timezone is None:
        parser.error("Naive timestamps require --timezone, for example America/Chicago or UTC")
    if args.input is None and args.bar_label != "close":
        parser.error("The bundled sample uses close labels; use --input for open-labeled bars")

    feed = FeedSpec(
        timezone=args.timezone,
        timestamp_semantics="bar_close" if args.bar_label == "close" else "event_time",
        bar_label="right" if args.bar_label == "close" else "left",
        time_contract_version=2,
    )
    assigned = (
        SessionAssigner(calendar="CME_Equity", feed_spec=feed)
        .assign_sessions(
            bars.rename({"datetime": "timestamp"}),
            bar_frequency="intraday",
            outside_session="raise",
        )
        .rename({"timestamp": "datetime"})
    )
    stats = (
        assigned.group_by("session_date")
        .agg(
            pl.len().alias("n_bars"),
            pl.col("Volume").sum().alias("total_volume"),
            pl.col("# of Trades").sum().alias("total_trades"),
        )
        .sort("session_date")
    )
    assigned.write_parquet(args.output)
    print(stats.write_csv(), end="")
    print(f"Saved {len(assigned)} unchanged bars with session dates to {args.output}")
    print(
        "Session groups preserve session boundaries; chronological validation needs a time-ordered split."
    )


if __name__ == "__main__":
    main()
