#!/usr/bin/env python3
"""Example: Assign session dates to NASDAQ 100 futures bar data.

This example demonstrates how to:
1. Load bar data (volume/trade bars, not time bars) from external source
2. Assign CME session dates for proper cross-validation splits
3. Analyze data by trading session
4. Export for use in backtesting/ML pipelines

Bar data characteristics:
- Irregular timestamps (bars close when volume/trade threshold reached)
- CME futures sessions: Sunday 5pm CT → Friday 4pm CT
- Session date = date when session ENDS (4pm date)

Usage:
    python nasdaq_bars_sessions.py
"""

from pathlib import Path

import polars as pl

from ml4t.data.sessions import SessionAssigner

# Path to bar data
BAR_DATA_PATH = Path.home() / "clients/chimera/bias_strategy/data/processed/nqu25_processed.parquet"
OUTPUT_PATH = (
    Path.home() / "clients/chimera/bias_strategy/data/processed/nqu25_with_sessions.parquet"
)


def main():
    """Assign sessions to NASDAQ 100 futures bar data."""
    print("=" * 80)
    print("NASDAQ 100 Futures - Bar Data Session Assignment")
    print("=" * 80)

    # 1. Load bar data
    print(f"\n[1] Loading bar data from {BAR_DATA_PATH}...")
    df = pl.read_parquet(BAR_DATA_PATH)

    print(f"    Loaded: {len(df):,} bars")
    print(f"    Columns: {len(df.columns)}")
    print(f"    Date range: {df['datetime'].min()} → {df['datetime'].max()}")

    # Show sample
    print("\n    Sample bars:")
    print(df.select(["datetime", "Open", "High", "Low", "Last", "Volume", "# of Trades"]).head(5))

    # 2. Rename datetime → timestamp for SessionAssigner
    print("\n[2] Preparing data for session assignment...")
    df = df.rename({"datetime": "timestamp"})

    # 3. Assign CME sessions
    print("\n[3] Assigning CME futures sessions...")
    print("    Calendar: CME_Globex_Equity")
    print("    Session times: Sunday 5pm CT → Friday 4pm CT (23 hours/day)")

    assigner = SessionAssigner(calendar_name="CME_Equity")
    df_with_sessions = assigner.assign_sessions(df)

    # Rename back
    df_with_sessions = df_with_sessions.rename({"timestamp": "datetime"})

    # 4. Session statistics
    print("\n[4] Session statistics:")
    n_sessions = df_with_sessions["session_date"].n_unique()
    print(f"    Total sessions: {n_sessions:,}")

    # Bars per session
    session_stats = (
        df_with_sessions.group_by("session_date")
        .agg(
            [
                pl.count().alias("n_bars"),
                pl.col("Volume").sum().alias("total_volume"),
                pl.col("# of Trades").sum().alias("total_trades"),
            ]
        )
        .sort("session_date")
    )

    print("\n    Bars per session statistics:")
    print(f"      Mean: {session_stats['n_bars'].mean():.0f} bars")
    print(f"      Median: {session_stats['n_bars'].median():.0f} bars")
    print(f"      Min: {session_stats['n_bars'].min()} bars")
    print(f"      Max: {session_stats['n_bars'].max()} bars")

    # Show first and last sessions
    print("\n    First 5 sessions:")
    print(session_stats.head(5))

    print("\n    Last 5 sessions:")
    print(session_stats.tail(5))

    # 5. Cross-validation implications
    print("\n[5] Cross-validation considerations:")
    print("    ✓ Sessions can be used as CV split boundaries")
    print("    ✓ Prevents data leakage across trading sessions")
    print("    ✓ Each fold contains complete trading sessions")

    # Example: how many bars in first 10 sessions
    first_10_sessions = session_stats.head(10)["session_date"].to_list()
    first_10_bars = df_with_sessions.filter(pl.col("session_date").is_in(first_10_sessions))
    print("\n    Example CV fold (first 10 sessions):")
    print(f"      Sessions: {len(first_10_sessions)}")
    print(f"      Bars: {len(first_10_bars):,}")
    print(
        f"      Date range: {first_10_bars['datetime'].min()} → {first_10_bars['datetime'].max()}"
    )

    # 6. Save output
    print(f"\n[6] Saving output to {OUTPUT_PATH}...")
    df_with_sessions.write_parquet(OUTPUT_PATH)
    print(f"    ✓ Saved {len(df_with_sessions):,} bars with session dates")

    # 7. Verify output
    print("\n[7] Verification:")
    df_verify = pl.read_parquet(OUTPUT_PATH)
    print(f"    ✓ File readable: {len(df_verify):,} bars")
    print(f"    ✓ session_date column: {df_verify['session_date'].dtype}")
    print(f"    ✓ Sessions: {df_verify['session_date'].n_unique():,}")

    print("\n" + "=" * 80)
    print("✓ Session assignment complete!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Use session_date for time-series cross-validation")
    print("  2. Group by session_date for session-level analysis")
    print("  3. Filter by session_date for specific date ranges")
    print("  4. Join with other session-aware datasets")

    print("\nExample usage in cross-validation:")
    print("""
    import polars as pl
    from sklearn.model_selection import GroupKFold

    # Load data with sessions
    df = pl.read_parquet("nqu25_with_sessions.parquet")

    # Create features and target
    X = df.select([...feature_columns...])
    y = df.select("target")
    groups = df["session_date"]

    # Group K-Fold ensures sessions don't split across folds
    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(X, y, groups):
        # Each fold contains complete sessions
        train_sessions = groups[train_idx].unique()
        test_sessions = groups[test_idx].unique()
        # No overlap: train_sessions ∩ test_sessions = ∅
    """)


if __name__ == "__main__":
    main()
