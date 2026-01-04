#!/usr/bin/env python3
"""Standalone utility for assigning session dates to any DataFrame.

This script demonstrates how to add session_date columns to data that:
1. Is not stored in ml4t-data (external parquet/CSV files)
2. Needs session assignment for cross-validation or analysis
3. May be bar data (volume bars, trade bars) instead of time bars

Works with any exchange supported by pandas_market_calendars.

Usage:
    python assign_sessions_standalone.py \
        --input data.parquet \
        --output data_with_sessions.parquet \
        --exchange CME \
        --timestamp-column datetime

Supported exchanges: CME, NASDAQ, NYSE, LSE, TSE, HKEX, ASX, SSE, TSX
"""

import argparse
import sys
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger()


def assign_sessions_to_file(
    input_path: str,
    output_path: str | None = None,
    exchange: str = "CME",
    calendar: str | None = None,
    timestamp_column: str = "timestamp",
    inplace: bool = False,
) -> pl.DataFrame:
    """Assign session dates to a parquet/CSV file.

    Args:
        input_path: Path to input file (parquet or CSV)
        output_path: Path to output file (optional, creates {input}_with_sessions.parquet)
        exchange: Exchange code (CME, NASDAQ, NYSE, etc.)
        calendar: Calendar name override (e.g., "CME_Globex_Crypto")
        timestamp_column: Name of timestamp column in data
        inplace: If True, overwrite input file with output

    Returns:
        DataFrame with session_date column added

    Example:
        >>> df = assign_sessions_to_file(
        ...     "nq_bars.parquet",
        ...     exchange="CME",
        ...     timestamp_column="datetime"
        ... )
        >>> print(df.select(["datetime", "session_date"]).head())
    """
    from ml4t.data.sessions import SessionAssigner

    # Read input file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Reading data from {input_path}")

    if input_file.suffix == ".parquet":
        df = pl.read_parquet(input_path)
    elif input_file.suffix == ".csv":
        df = pl.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    logger.info(
        f"Loaded {len(df):,} rows with {len(df.columns)} columns",
        shape=df.shape,
    )

    # Rename timestamp column if needed
    if timestamp_column != "timestamp":
        if timestamp_column not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found. Available columns: {df.columns}"
            )
        df = df.rename({timestamp_column: "timestamp"})
        logger.info(f"Renamed '{timestamp_column}' → 'timestamp'")

    # Check for existing session_date column
    if "session_date" in df.columns:
        logger.warning(
            "session_date column already exists, will be replaced",
            action="drop_existing",
        )
        df = df.drop("session_date")

    # Initialize session assigner
    if calendar:
        assigner = SessionAssigner(calendar)
        logger.info(f"Using calendar: {calendar}")
    else:
        assigner = SessionAssigner.from_exchange(exchange)
        logger.info(f"Using exchange: {exchange}")

    # Assign sessions
    logger.info("Assigning session dates...")
    df_with_sessions = assigner.assign_sessions(df)

    # Rename back if needed
    if timestamp_column != "timestamp":
        df_with_sessions = df_with_sessions.rename({"timestamp": timestamp_column})
        logger.info(f"Renamed 'timestamp' → '{timestamp_column}'")

    # Count sessions
    n_sessions = df_with_sessions["session_date"].n_unique()
    logger.info(
        f"Assigned {n_sessions:,} unique sessions",
        date_range=(
            df_with_sessions["session_date"].min(),
            df_with_sessions["session_date"].max(),
        ),
    )

    # Write output
    if inplace:
        output_file = input_file
    elif output_path:
        output_file = Path(output_path)
    else:
        # Auto-generate output filename
        output_file = input_file.with_stem(f"{input_file.stem}_with_sessions")

    logger.info(f"Writing output to {output_file}")
    df_with_sessions.write_parquet(output_file)

    logger.info(
        "✓ Session assignment complete",
        output=str(output_file),
        rows=len(df_with_sessions),
        sessions=n_sessions,
    )

    return df_with_sessions


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Assign session dates to trading data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # CME futures (crypto or equity index)
  python assign_sessions_standalone.py \\
      --input nq_bars.parquet \\
      --exchange CME \\
      --timestamp-column datetime

  # Use specific calendar (more control)
  python assign_sessions_standalone.py \\
      --input btc_futures.parquet \\
      --calendar CME_Globex_Crypto

  # NASDAQ equities
  python assign_sessions_standalone.py \\
      --input spy_minute.parquet \\
      --exchange NASDAQ

  # Overwrite input file
  python assign_sessions_standalone.py \\
      --input data.parquet \\
      --exchange CME \\
      --inplace

  # Custom output location
  python assign_sessions_standalone.py \\
      --input data.parquet \\
      --output /path/to/output.parquet \\
      --exchange CME

Supported exchanges:
  CME, NASDAQ, NYSE, LSE, TSE, HKEX, ASX, SSE, TSX

For full list of calendars, see:
  https://github.com/rsheftel/pandas_market_calendars
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file (parquet or CSV)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: {input}_with_sessions.parquet)",
    )
    parser.add_argument(
        "--exchange",
        "-e",
        default="CME",
        help="Exchange code (default: CME)",
    )
    parser.add_argument(
        "--calendar",
        "-c",
        help="Calendar name (overrides exchange)",
    )
    parser.add_argument(
        "--timestamp-column",
        "-t",
        default="timestamp",
        help="Name of timestamp column (default: timestamp)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input file with output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG))

    try:
        df = assign_sessions_to_file(
            input_path=args.input,
            output_path=args.output,
            exchange=args.exchange,
            calendar=args.calendar,
            timestamp_column=args.timestamp_column,
            inplace=args.inplace,
        )

        # Display sample
        print("\n" + "=" * 80)
        print("Sample output (first 10 rows):")
        print("=" * 80)
        print(
            df.select(
                [
                    args.timestamp_column if args.timestamp_column in df.columns else "timestamp",
                    "session_date",
                ]
                + [
                    c
                    for c in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "Open",
                        "High",
                        "Low",
                        "Last",
                        "Volume",
                    ]
                    if c in df.columns
                ][:5]
            ).head(10)
        )

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    import logging

    main()
