"""Inspect Wyden crypto-data-pipeline data before migration.

Quickly check what data exists and its structure.
"""

from pathlib import Path

import polars as pl

SOURCE_BASE = Path.home() / "clients/wyden/long-short/crypto-data-pipeline/data_store"
FUTURES_DIR = SOURCE_BASE / "futures"
SPOT_DIR = SOURCE_BASE / "spot"


def inspect_dataset(path: Path, name: str):
    """Inspect a single dataset."""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")
    print(f"Path: {path}")

    if not path.exists():
        print("❌ Directory not found")
        return

    try:
        # Try to read one file to check schema
        parquet_files = list(path.glob("**/*.parquet"))
        if not parquet_files:
            print("⚠️  No .parquet files found")
            return

        print(f"Found {len(parquet_files)} parquet files")

        # Read first file to check schema
        first_file = parquet_files[0]
        print(f"\nSample file: {first_file.relative_to(path)}")

        df_sample = pl.read_parquet(first_file)
        print(f"Rows in sample: {len(df_sample):,}")
        print("\nSchema:")
        for col, dtype in zip(df_sample.columns, df_sample.dtypes):
            print(f"  {col:20} {dtype}")

        print("\nDate range in sample:")
        if "timestamp" in df_sample.columns:
            print(f"  {df_sample['timestamp'].min()} to {df_sample['timestamp'].max()}")

        # Try to scan all files for total stats
        print(f"\nScanning all {len(parquet_files)} files...")
        df_all = pl.scan_parquet(str(path / "**/*.parquet")).select(["timestamp"]).collect()
        print(f"Total rows: {len(df_all):,}")
        print(f"Full date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")

    except Exception as e:
        print(f"❌ Error reading data: {e}")


def main():
    """Inspect all datasets."""
    print("=" * 80)
    print("Wyden Data Inspection")
    print("=" * 80)
    print(f"Source: {SOURCE_BASE}")

    # Check base directories
    if not SOURCE_BASE.exists():
        print(f"❌ Base directory not found: {SOURCE_BASE}")
        return

    print("\n✓ Base directory exists")
    print(f"  Futures: {FUTURES_DIR.exists()}")
    print(f"  Spot: {SPOT_DIR.exists()}")

    # Inspect futures
    if FUTURES_DIR.exists():
        for symbol_dir in sorted(FUTURES_DIR.glob("*")):
            if symbol_dir.is_dir() and symbol_dir.name in ["BTC", "ETH"]:
                inspect_dataset(symbol_dir, f"Futures - {symbol_dir.name}")

    # Inspect spot
    if SPOT_DIR.exists():
        for symbol_dir in sorted(SPOT_DIR.glob("*")):
            if symbol_dir.is_dir() and symbol_dir.name in ["BTC", "ETH"]:
                inspect_dataset(symbol_dir, f"Spot - {symbol_dir.name}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nData found:")

    if FUTURES_DIR.exists():
        futures_symbols = [
            d.name for d in FUTURES_DIR.glob("*") if d.is_dir() and d.name in ["BTC", "ETH"]
        ]
        print(f"  Futures: {', '.join(futures_symbols) if futures_symbols else 'None'}")

    if SPOT_DIR.exists():
        spot_symbols = [
            d.name for d in SPOT_DIR.glob("*") if d.is_dir() and d.name in ["BTC", "ETH"]
        ]
        print(f"  Spot: {', '.join(spot_symbols) if spot_symbols else 'None'}")

    print("\nNext step: Run migration script")
    print("  python examples/migrate_wyden_data.py")


if __name__ == "__main__":
    main()
