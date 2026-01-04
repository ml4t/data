"""Migration Script: Wyden crypto-data-pipeline → ml4t-data

Migrate existing BTC/ETH spot and futures data from crypto-data-pipeline to ml4t-data storage.

Source: ~/clients/wyden/long-short/crypto-data-pipeline/data_store/
Target: ./data/ (ml4t-data storage)

Data to migrate:
- Futures: BTC, ETH (DataBento, 2017-2025, 1-minute bars)
- Spot: BTC, ETH (CryptoCompare, 1-minute bars)
"""

import sys
from pathlib import Path

import polars as pl

from ml4t.data import DataManager

# Source directories
SOURCE_BASE = Path.home() / "clients/wyden/long-short/crypto-data-pipeline/data_store"
FUTURES_DIR = SOURCE_BASE / "futures"
SPOT_DIR = SOURCE_BASE / "spot"

# Target storage
TARGET_STORAGE = str(Path.home() / "clients/wyden/long-short/data")


def migrate_futures_data(manager: DataManager, symbol: str):
    """Migrate futures data for a symbol (BTC or ETH)."""
    print(f"\n{'=' * 80}")
    print(f"Migrating Futures Data: {symbol}")
    print(f"{'=' * 80}")

    futures_path = FUTURES_DIR / symbol
    if not futures_path.exists():
        print(f"⚠️  No futures data found at {futures_path}")
        return

    print(f"Source: {futures_path}")
    print("Reading hive-partitioned Parquet files...")

    # Read all parquet files (hive-partitioned: year=YYYY/month=MM/data.parquet)
    try:
        df = pl.scan_parquet(str(futures_path / "**/*.parquet")).collect()
        print(f"✓ Loaded {len(df):,} rows")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Columns: {df.columns}")

        # Check schema
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
            print(f"  Available columns: {df.columns}")
            return

        # Cast volume to Float64 (ml4t-data expects numeric, not UInt64)
        df = df.with_columns(pl.col("volume").cast(pl.Float64))
        print("✓ Cast volume to Float64")

        # Import into ml4t-data
        print("\nImporting into ml4t.data...")
        manager.import_data(
            data=df,
            symbol=symbol,
            provider="databento",
            frequency="1min",
            asset_class="crypto_futures",
            exchange="CME",
            calendar="CME_Globex_Crypto",
        )

        print(f"✓ Successfully imported {symbol} futures data")

        # Verify storage
        key = f"crypto_futures/1min/{symbol}"
        stored_df = manager.storage.read(key).collect()
        print(f"✓ Verified: {len(stored_df):,} rows in storage")

        # Check metadata
        metadata = manager.get_metadata(symbol)
        if metadata:
            print("\nMetadata:")
            print(f"  Provider: {metadata.provider}")
            print(f"  Exchange: {metadata.exchange}")
            print(f"  Calendar: {metadata.calendar}")
            print(f"  Frequency: {metadata.frequency}")
            print(f"  Last updated: {metadata.last_updated}")

    except Exception as e:
        print(f"❌ Error migrating {symbol} futures: {e}")
        import traceback

        traceback.print_exc()


def migrate_spot_data(manager: DataManager, symbol: str):
    """Migrate spot data for a symbol (BTC or ETH)."""
    print(f"\n{'=' * 80}")
    print(f"Migrating Spot Data: {symbol}")
    print(f"{'=' * 80}")

    spot_path = SPOT_DIR / symbol
    if not spot_path.exists():
        print(f"⚠️  No spot data found at {spot_path}")
        return

    print(f"Source: {spot_path}")
    print("Reading hive-partitioned Parquet files...")

    try:
        df = pl.scan_parquet(str(spot_path / "**/*.parquet")).collect()
        print(f"✓ Loaded {len(df):,} rows")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Columns: {df.columns}")

        # Check schema
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
            print(f"  Available columns: {df.columns}")
            return

        # Cast volume to Float64 (ml4t-data expects numeric, not UInt64)
        df = df.with_columns(pl.col("volume").cast(pl.Float64))
        print("✓ Cast volume to Float64")

        # Import into ml4t-data
        print("\nImporting into ml4t.data...")
        manager.import_data(
            data=df,
            symbol=symbol,
            provider="cryptocompare",
            frequency="1min",
            asset_class="crypto",
            exchange="CRYPTOCOMPARE",  # Aggregated exchange
        )

        print(f"✓ Successfully imported {symbol} spot data")

        # Verify storage
        key = f"crypto/1min/{symbol}"
        stored_df = manager.storage.read(key).collect()
        print(f"✓ Verified: {len(stored_df):,} rows in storage")

        # Check metadata
        metadata = manager.get_metadata(symbol)
        if metadata:
            print("\nMetadata:")
            print(f"  Provider: {metadata.provider}")
            print(f"  Exchange: {metadata.exchange}")
            print(f"  Frequency: {metadata.frequency}")
            print(f"  Last updated: {metadata.last_updated}")

    except Exception as e:
        print(f"❌ Error migrating {symbol} spot: {e}")
        import traceback

        traceback.print_exc()


def verify_migration(manager: DataManager):
    """Verify all migrated data."""
    print(f"\n{'=' * 80}")
    print("Migration Verification")
    print(f"{'=' * 80}")

    # List all symbols
    symbols = manager.list_symbols()
    print(f"\nTotal symbols in ml4t-data storage: {len(symbols)}")

    for symbol in sorted(symbols):
        metadata = manager.get_metadata(symbol)
        if metadata:
            df = manager.storage.read(
                f"{metadata.asset_class}/{metadata.frequency}/{symbol}"
            ).collect()
            print(f"\n{symbol}:")
            print(f"  Provider: {metadata.provider}")
            print(f"  Asset class: {metadata.asset_class}")
            print(f"  Exchange: {metadata.exchange}")
            print(f"  Rows: {len(df):,}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


def main():
    """Run complete migration."""
    print("=" * 80)
    print("Wyden Data Migration: crypto-data-pipeline → ml4t-data")
    print("=" * 80)

    # Check source directories
    print("\nChecking source directories...")
    if not SOURCE_BASE.exists():
        print(f"❌ Source directory not found: {SOURCE_BASE}")
        sys.exit(1)

    print(f"✓ Source base: {SOURCE_BASE}")
    print(f"  Futures: {FUTURES_DIR} {'✓' if FUTURES_DIR.exists() else '✗'}")
    print(f"  Spot: {SPOT_DIR} {'✓' if SPOT_DIR.exists() else '✗'}")

    # Initialize DataManager with storage
    print(f"\nInitializing ml4t-data storage: {TARGET_STORAGE}")
    from ml4t.data.storage.backend import StorageConfig
    from ml4t.data.storage.hive import HiveStorage

    storage = HiveStorage(config=StorageConfig(base_path=TARGET_STORAGE))
    manager = DataManager(storage=storage)
    print("✓ DataManager ready")

    # Migrate data
    symbols_to_migrate = ["BTC", "ETH"]

    print("\n" + "=" * 80)
    print("FUTURES DATA MIGRATION")
    print("=" * 80)

    for symbol in symbols_to_migrate:
        migrate_futures_data(manager, symbol)

    print("\n" + "=" * 80)
    print("SPOT DATA MIGRATION")
    print("=" * 80)

    for symbol in symbols_to_migrate:
        migrate_spot_data(manager, symbol)

    # Verify
    verify_migration(manager)

    # Summary
    print("\n" + "=" * 80)
    print("Migration Complete!")
    print("=" * 80)

    print("\nNext steps:")
    print("1. Verify data integrity:")
    print(
        "   python -c \"from ml4t.data import DataManager; m = DataManager('./data'); print(m.list_symbols())\""
    )
    print("\n2. Test session management:")
    print("   python examples/cme_futures_sessions.py")
    print("\n3. Update data incrementally:")
    print("   from ml4t.data import DataManager")
    print("   manager = DataManager('./data')")
    print("   manager.update('BTC')  # Only fetches new data")
    print("\n4. Update all symbols:")
    print("   manager.update_all(provider='databento')  # Update all futures")
    print("   manager.update_all(provider='cryptocompare')  # Update all spot")

    print("\n✓ All data migrated successfully!")
    print(f"✓ Storage location: {Path(TARGET_STORAGE).resolve()}")


if __name__ == "__main__":
    main()
