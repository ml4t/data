#!/usr/bin/env python3
"""Complete workflow: CME crypto futures with session management.

This example demonstrates the full workflow for working with Wyden CME futures data:
1. Load data from ml4t.data storage (already migrated from crypto-data-pipeline)
2. Assign CME session dates
3. Complete sessions (fill gaps)
4. Session-level analysis
5. Cross-validation with sessions

This is production data stored at ~/clients/wyden/long-short/data/
- BTC futures: 2.79M rows (Dec 2017 → present)
- ETH futures: 1.68M rows (Feb 2021 → present)

Usage:
    python wyden_cme_sessions_complete_workflow.py
"""

from pathlib import Path

import polars as pl

from ml4t.data import DataManager
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage

# Wyden data location
DATA_PATH = Path.home() / "clients/wyden/long-short/data"

# Symbols to process
SYMBOLS = ["BTC", "ETH"]


def main():
    """Complete session management workflow for CME futures."""
    print("=" * 80)
    print("Wyden CME Crypto Futures - Complete Session Workflow")
    print("=" * 80)

    # 1. Initialize DataManager with Wyden storage
    print(f"\n[1] Initializing DataManager with storage: {DATA_PATH}")
    storage = HiveStorage(config=StorageConfig(base_path=str(DATA_PATH)))
    manager = DataManager(storage=storage)

    for symbol in SYMBOLS:
        print(f"\n{'=' * 80}")
        print(f"Processing {symbol} CME Futures")
        print(f"{'=' * 80}")

        # 2. Read data from storage
        print(f"\n[2] Reading {symbol} futures data from storage...")
        key = f"crypto_futures_1min_{symbol}"
        df = storage.read(key).collect()

        print(f"    Loaded: {len(df):,} rows")
        print(f"    Columns: {df.columns}")
        print(f"    Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

        # Check if already has sessions
        if "session_date" in df.columns:
            print("    ⚠️  session_date column already exists (will be replaced)")
            df = df.drop("session_date")

        # 3. Assign CME session dates
        print("\n[3] Assigning CME session dates...")
        print("    Calendar: CME_Globex_Crypto")
        print("    Sessions: Sunday 5pm CT → Friday 4pm CT (23 hours/day)")
        print("    Session date = date when session ENDS (4pm CT)")

        df_with_sessions = manager.assign_sessions(
            df,
            exchange="CME",  # Auto-selects CME_Globex_Crypto calendar
        )

        n_sessions = df_with_sessions["session_date"].n_unique()
        print(f"    ✓ Assigned {n_sessions:,} unique sessions")

        # 4. Complete sessions (fill gaps)
        print("\n[4] Completing sessions (filling gaps)...")
        print("    Strategy: Forward-fill OHLC, zero volume")

        df_complete = manager.complete_sessions(
            df_with_sessions,
            exchange="CME",
            fill_gaps=True,  # Fill missing minutes within sessions
            zero_volume=True,  # Set volume=0 for filled bars
        )

        n_filled = len(df_complete) - len(df_with_sessions)
        print(
            f"    ✓ Filled {n_filled:,} missing bars ({n_filled / len(df_with_sessions) * 100:.2f}% of original)"
        )
        print(f"    Total bars: {len(df_complete):,}")

        # 5. Session statistics
        print("\n[5] Session statistics:")

        session_stats = (
            df_complete.group_by("session_date")
            .agg(
                [
                    pl.count().alias("n_bars"),
                    pl.col("volume").sum().alias("total_volume"),
                    pl.col("close").last().alias("session_close"),
                    pl.col("close").first().alias("session_open"),
                    (pl.col("close").last() - pl.col("close").first()).alias("session_return"),
                ]
            )
            .with_columns(
                [
                    (pl.col("session_return") / pl.col("session_open") * 100).alias(
                        "session_return_pct"
                    )
                ]
            )
            .sort("session_date")
        )

        print(f"    Total sessions: {len(session_stats):,}")
        print("    Bars per session:")
        print(f"      Mean: {session_stats['n_bars'].mean():.0f}")
        print(f"      Expected: {23 * 60} (23 hours × 60 minutes)")
        print(f"      Coverage: {session_stats['n_bars'].mean() / (23 * 60) * 100:.1f}%")

        print("\n    First 3 sessions:")
        print(session_stats.head(3))

        print("\n    Last 3 sessions:")
        print(session_stats.tail(3))

        # 6. Cross-validation setup
        print("\n[6] Cross-validation setup:")
        print("    ✓ Use session_date as group column")
        print("    ✓ Prevents data leakage across sessions")
        print("    ✓ Each fold contains complete trading sessions")

        # Example: split into 5 folds by session
        n_folds = 5
        sessions_per_fold = n_sessions // n_folds

        print(f"\n    Example {n_folds}-fold split:")
        print(f"      Total sessions: {n_sessions:,}")
        print(f"      Sessions per fold: ~{sessions_per_fold:,}")

        for fold in range(n_folds):
            start_idx = fold * sessions_per_fold
            end_idx = (fold + 1) * sessions_per_fold if fold < n_folds - 1 else n_sessions
            fold_sessions = session_stats.slice(start_idx, end_idx - start_idx)
            fold_bars = df_complete.filter(
                pl.col("session_date").is_in(fold_sessions["session_date"])
            )

            print(
                f"      Fold {fold + 1}: "
                f"{len(fold_sessions)} sessions, "
                f"{len(fold_bars):,} bars, "
                f"{fold_sessions['session_date'].min()} → {fold_sessions['session_date'].max()}"
            )

        # 7. Save output with sessions
        print("\n[7] Saving data with sessions...")
        output_key = f"{key}_with_sessions"
        output_path = DATA_PATH / f"{output_key}.parquet"

        df_complete.write_parquet(output_path)
        print(f"    ✓ Saved to {output_path}")
        print(f"    Rows: {len(df_complete):,}")
        print(f"    Columns: {len(df_complete.columns)} (including session_date)")

    # 8. Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"✓ Processed {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}")
    print("✓ Assigned CME session dates")
    print("✓ Filled gaps for complete sessions")
    print("✓ Saved output with session_date column")

    print("\nNext steps:")
    print("  1. Use session_date for GroupKFold cross-validation")
    print("  2. Analyze performance by session (day of week, time patterns)")
    print("  3. Build features using session-aware rolling windows")
    print("  4. Ensure train/test splits don't break session boundaries")

    print("\nExample GroupKFold usage:")
    print("""
    from sklearn.model_selection import GroupKFold

    # Load data with sessions
    df = pl.read_parquet("crypto_futures_1min_BTC_with_sessions.parquet")

    # Prepare features and target
    X = df.select([...feature_columns...]).to_numpy()
    y = df["target"].to_numpy()
    groups = df["session_date"].to_numpy()

    # Group K-Fold: each session stays in one fold
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"Fold {fold + 1}:")
        print(f"  Train sessions: {len(np.unique(groups[train_idx]))}")
        print(f"  Test sessions: {len(np.unique(groups[test_idx]))}")
        print(f"  No overlap: {len(set(groups[train_idx]) & set(groups[test_idx])) == 0}")
    """)


if __name__ == "__main__":
    main()
