"""CME Futures Session Management Example

Demonstrates how to:
1. Load minute-level CME futures data
2. Assign session dates based on CME calendar
3. Fill gaps with forward-filled OHLC and zero volume
4. Prepare data for session-aware cross-validation

CME Bitcoin futures trade Sunday 5pm CT → Friday 4pm CT (23 hours/day).
Sessions span multiple calendar days, requiring special handling for:
- Cross-validation (split by session, not calendar date)
- Gap filling (some providers skip zero-volume periods)
- Session aggregation (daily/weekly session summaries)
"""

import polars as pl

from ml4t.data import DataManager

# Constants
SYMBOL = "BTC"
PROVIDER = "databento"  # Institutional futures data
EXCHANGE = "CME"
CALENDAR = "CME_Globex_Crypto"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"


def main():
    """Complete CME futures workflow with session management."""

    # Initialize data manager
    manager = DataManager(storage_path="./data")

    print("=" * 80)
    print("CME Bitcoin Futures - Session Management Workflow")
    print("=" * 80)

    # Step 1: Load minute-level futures data
    print("\n1. Loading CME Bitcoin futures (1-minute bars)...")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Exchange: {EXCHANGE}")
    print(f"   Period: {START_DATE} to {END_DATE}")

    key = manager.load(
        symbol=SYMBOL,
        start=START_DATE,
        end=END_DATE,
        provider=PROVIDER,
        frequency="1min",
        exchange=EXCHANGE,
        calendar=CALENDAR,
    )

    print(f"   ✓ Data loaded and stored at: {key}")

    # Step 2: Read data from storage
    print("\n2. Reading data from storage...")
    df = manager.storage.read(key).collect()
    print(f"   ✓ Loaded {len(df):,} rows")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Check for gaps in data
    df_sorted = df.sort("timestamp")
    time_diffs = df_sorted["timestamp"].diff().dt.total_minutes()
    gaps = time_diffs[time_diffs > 1].drop_nulls()
    print(f"   Gaps detected: {len(gaps)} periods with >1 minute difference")

    # Step 3: Assign session dates
    print("\n3. Assigning CME session dates...")
    print("   CME sessions: Sunday 5pm CT → Friday 4pm CT (23 hours/day)")

    df_with_sessions = manager.assign_sessions(df, exchange=EXCHANGE)
    print("   ✓ Added session_date column")

    # Count unique sessions
    unique_sessions = df_with_sessions["session_date"].n_unique()
    print(f"   Unique sessions: {unique_sessions}")

    # Show sample session data
    sample_session = (
        df_with_sessions.group_by("session_date")
        .agg(
            [
                pl.col("timestamp").min().alias("session_start"),
                pl.col("timestamp").max().alias("session_end"),
                pl.col("timestamp").count().alias("num_bars"),
            ]
        )
        .head(5)
    )

    print("\n   Sample sessions:")
    print(sample_session)

    # Step 4: Fill gaps for complete sessions
    print("\n4. Filling gaps with forward-filled OHLC and zero volume...")
    print("   Strategy:")
    print("   - Forward-fill OHLC prices from last close")
    print("   - Set volume=0 for filled rows")
    print("   - Ensures continuous minute-level data")

    df_complete = manager.complete_sessions(
        df_with_sessions,
        exchange=EXCHANGE,
        fill_gaps=True,
        zero_volume=True,
    )

    rows_added = len(df_complete) - len(df_with_sessions)
    print(f"   ✓ Added {rows_added:,} rows to fill gaps")
    print(f"   Total rows: {len(df_complete):,}")

    # Verify completeness
    time_diffs_complete = df_complete.sort("timestamp")["timestamp"].diff().dt.total_minutes()
    remaining_gaps = time_diffs_complete[time_diffs_complete > 1].drop_nulls()
    print(f"   Remaining gaps: {len(remaining_gaps)}")

    # Step 5: Session-level aggregation
    print("\n5. Computing session-level statistics...")

    session_stats = df_complete.group_by("session_date").agg(
        [
            pl.col("timestamp").count().alias("num_minutes"),
            pl.col("open").first().alias("session_open"),
            pl.col("high").max().alias("session_high"),
            pl.col("low").min().alias("session_low"),
            pl.col("close").last().alias("session_close"),
            pl.col("volume").sum().alias("total_volume"),
            (pl.col("volume") == 0).sum().alias("zero_volume_bars"),
        ]
    )

    print("\n   Session statistics (first 5 sessions):")
    print(session_stats.sort("session_date").head(5))

    # Step 6: Prepare for cross-validation
    print("\n6. Session-aware cross-validation preparation...")
    print("   ✓ Data ready for session-based splits")
    print("   Example usage with qeval library:")
    print("""
    from ml4t.evaluation.cross_validation import TimeSeriesSplit

    # Split by session_date instead of timestamp
    splitter = TimeSeriesSplit(n_splits=5, group_col="session_date")
    for train_idx, test_idx in splitter.split(df_complete):
        train_data = df_complete[train_idx]
        test_data = df_complete[test_idx]
        # Train and evaluate model...
    """)

    # Step 7: Export options
    print("\n7. Export options...")
    print("   Save to Parquet:")
    print('   df_complete.write_parquet("btc_sessions_complete.parquet")')
    print("\n   Save to CSV:")
    print('   df_complete.write_csv("btc_sessions_complete.csv")')

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Symbol: {SYMBOL}")
    print(f"Exchange: {EXCHANGE}")
    print(f"Original rows: {len(df):,}")
    print(f"Rows added (gap filling): {rows_added:,}")
    print(f"Final rows: {len(df_complete):,}")
    print(f"Unique sessions: {unique_sessions}")
    print(f"Average bars per session: {len(df_complete) / unique_sessions:,.0f}")
    print("Expected bars per session: 1,380 (23 hours × 60 minutes)")
    print("\n✓ Data ready for backtesting and cross-validation!")


def demonstrate_without_session_filling():
    """Show the problem when NOT using session filling."""
    print("\n" + "=" * 80)
    print("Why Session Filling Matters")
    print("=" * 80)

    manager = DataManager(storage_path="./data")

    # Load data
    key = manager.load(
        symbol=SYMBOL,
        start=START_DATE,
        end=END_DATE,
        provider=PROVIDER,
        frequency="1min",
        exchange=EXCHANGE,
    )

    df = manager.storage.read(key).collect()

    print("\nWithout gap filling:")
    print(f"  Total rows: {len(df):,}")

    # Check for gaps
    df_sorted = df.sort("timestamp")
    time_diffs = df_sorted["timestamp"].diff().dt.total_minutes()
    gaps = time_diffs[time_diffs > 1].drop_nulls()

    print(f"  Gaps > 1 minute: {len(gaps)}")
    print(f"  Largest gap: {gaps.max():.0f} minutes")

    print("\nProblems without filling:")
    print("  ✗ Missing bars distort technical indicators (e.g., moving averages)")
    print("  ✗ Session aggregation incorrect (incomplete sessions)")
    print("  ✗ Cross-validation splits may have imbalanced sessions")
    print("  ✗ Backtest slippage calculations affected by gaps")

    print("\nWith gap filling (forward-fill OHLC, zero volume):")
    df_complete = manager.complete_sessions(df, exchange=EXCHANGE)
    print(f"  Total rows: {len(df_complete):,}")
    print(f"  Rows added: {len(df_complete) - len(df):,}")

    time_diffs_complete = df_complete.sort("timestamp")["timestamp"].diff().dt.total_minutes()
    remaining_gaps = time_diffs_complete[time_diffs_complete > 1].drop_nulls()
    print(f"  Remaining gaps: {len(remaining_gaps)}")

    print("\nBenefits:")
    print("  ✓ Continuous time series for indicators")
    print("  ✓ Complete sessions for aggregation")
    print("  ✓ Balanced cross-validation splits")
    print("  ✓ Realistic backtest execution")


if __name__ == "__main__":
    # Run main workflow
    main()

    # Demonstrate importance of gap filling
    demonstrate_without_session_filling()

    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("1. Use df_complete for feature engineering (indicators, labels)")
    print("2. Split by session_date for cross-validation")
    print("3. Train models respecting session boundaries")
    print("4. Backtest with session-aware execution")
    print("\nSee qfeatures and qeval libraries for next steps!")
