"""Example: Format Conversion Between Stacked and Wide Formats.

This example demonstrates how to convert multi-asset data between stacked
(long) and wide (pivoted) formats using the format conversion utilities.

**When to use this**:
- Converting data for legacy tools that expect wide format
- Preparing data for pandas-based correlation analysis
- Exporting to spreadsheets or visualization tools

**Performance Warning**:
Wide format does NOT scale well beyond ~100 symbols. The canonical format
is stacked/long - use wide format only when necessary for compatibility.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from ml4t.data.core.schemas import MultiAssetSchema
from ml4t.data.utils.format import pivot_to_stacked, pivot_to_wide


def example_basic_conversion():
    """Example 1: Basic stacked to wide conversion."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Stacked → Wide Conversion")
    print("=" * 70)

    # Create sample multi-asset data in canonical stacked format
    df_stacked = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
            ],
            "symbol": ["AAPL", "MSFT", "GOOGL", "AAPL", "MSFT", "GOOGL"],
            "open": [150.0, 370.0, 140.0, 151.0, 371.0, 141.0],
            "high": [152.0, 372.0, 142.0, 153.0, 373.0, 143.0],
            "low": [149.0, 369.0, 139.0, 150.0, 370.0, 140.0],
            "close": [151.0, 371.0, 141.0, 152.0, 372.0, 142.0],
            "volume": [1000000.0, 2000000.0, 1500000.0, 1100000.0, 2100000.0, 1600000.0],
        }
    )

    print("\nOriginal Stacked Format:")
    print(df_stacked)

    # Convert to wide format
    df_wide = pivot_to_wide(df_stacked)

    print("\nWide Format (pivoted):")
    print(df_wide)
    print(f"\nColumns in wide format: {len(df_wide.columns)}")
    print(f"Rows in wide format: {len(df_wide)}")


def example_custom_columns():
    """Example 2: Convert only specific columns to wide format."""
    print("\n" + "=" * 70)
    print("Example 2: Convert Only Specific Columns")
    print("=" * 70)

    # Create sample data
    df_stacked = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "open": [150.0, 370.0, 151.0, 371.0],
            "high": [152.0, 372.0, 153.0, 373.0],
            "low": [149.0, 369.0, 150.0, 370.0],
            "close": [151.0, 371.0, 152.0, 372.0],
            "volume": [1000000.0, 2000000.0, 1100000.0, 2100000.0],
        }
    )

    # Convert only close and volume to wide
    df_wide = pivot_to_wide(df_stacked, value_cols=["close", "volume"])

    print("\nWide Format (only close and volume):")
    print(df_wide)
    print(f"\nColumns: {df_wide.columns}")


def example_round_trip():
    """Example 3: Round-trip conversion (stacked → wide → stacked)."""
    print("\n" + "=" * 70)
    print("Example 3: Round-Trip Conversion")
    print("=" * 70)

    # Original stacked data
    df_original = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "close": [150.0, 370.0, 151.0, 371.0],
            "volume": [1000000.0, 2000000.0, 1100000.0, 2100000.0],
        }
    )

    print("\nOriginal Stacked Format:")
    print(df_original.sort(["timestamp", "symbol"]))

    # Convert to wide
    df_wide = pivot_to_wide(df_original, value_cols=["close", "volume"])
    print("\nWide Format:")
    print(df_wide)

    # Convert back to stacked
    df_back = pivot_to_stacked(df_wide)
    print("\nBack to Stacked Format:")
    print(df_back.sort(["timestamp", "symbol"]))

    # Verify round-trip preservation
    orig_sorted = df_original.sort(["timestamp", "symbol"])
    back_sorted = df_back.sort(["timestamp", "symbol"])

    if orig_sorted.equals(back_sorted):
        print("\n✓ Round-trip conversion preserved all data!")
    else:
        print("\n✗ Round-trip conversion did NOT preserve data")


def example_pandas_workflow():
    """Example 4: Using wide format for pandas analysis."""
    print("\n" + "=" * 70)
    print("Example 4: Pandas Workflow with Wide Format")
    print("=" * 70)

    # Create sample data
    df_stacked = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ],
            "symbol": ["AAPL", "MSFT", "GOOGL"] * 3,
            "close": [150.0, 370.0, 140.0, 151.0, 371.0, 141.0, 152.0, 372.0, 142.0],
        }
    )

    # Convert to wide for pandas correlation analysis
    df_wide = pivot_to_wide(df_stacked, value_cols=["close"])

    print("\nWide Format for Correlation Analysis:")
    print(df_wide)

    # Convert to pandas
    df_pandas = df_wide.to_pandas().set_index("timestamp")

    print("\nCorrelation Matrix:")
    print(df_pandas.corr())

    # Can also convert back to stacked for storage
    df_back = pivot_to_stacked(df_wide)
    print("\nConverted back to stacked for storage:")
    print(df_back)


def example_multiasset_schema_integration():
    """Example 5: Integration with MultiAssetSchema."""
    print("\n" + "=" * 70)
    print("Example 5: MultiAssetSchema Integration")
    print("=" * 70)

    # Create empty multi-asset DataFrame
    df_stacked = MultiAssetSchema.create_empty("equities")

    # Add some data
    df_stacked = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
            ],
            "symbol": ["AAPL", "MSFT"],
            "open": [150.0, 370.0],
            "high": [152.0, 372.0],
            "low": [149.0, 369.0],
            "close": [151.0, 371.0],
            "volume": [1000000.0, 2000000.0],
            "dividends": [0.0, 0.0],
            "splits": [1.0, 1.0],
            "adjusted_close": [151.0, 371.0],
        }
    )

    # Validate before conversion
    is_valid = MultiAssetSchema.validate(df_stacked, strict=False)
    print(f"\nOriginal stacked data is valid: {is_valid}")

    # Convert to wide
    df_wide = pivot_to_wide(df_stacked)
    print("\nWide format:")
    print(df_wide)

    # Convert back to stacked
    df_back = pivot_to_stacked(df_wide)

    # Standardize and validate
    df_back = MultiAssetSchema.standardize_order(df_back)
    is_valid_back = MultiAssetSchema.validate(df_back, strict=False)
    print(f"\nRound-trip stacked data is valid: {is_valid_back}")


def example_scalability_warning():
    """Example 6: Demonstrate scalability limits."""
    print("\n" + "=" * 70)
    print("Example 6: Scalability Warning")
    print("=" * 70)

    # Create data with many symbols
    n_symbols = 150
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

    data = []
    for symbol in symbols:
        data.append(
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
                "symbol": symbol,
                "close": 100.0,
                "volume": 1000000.0,
            }
        )

    df_stacked = pl.DataFrame(data)

    print(f"\nStacked format: {len(df_stacked)} rows, {len(df_stacked.columns)} columns")

    # Convert to wide (will show warning)
    df_wide = pivot_to_wide(df_stacked, value_cols=["close", "volume"])

    print(f"Wide format: {len(df_wide)} rows, {len(df_wide.columns)} columns")
    print(f"\n⚠️  Wide format creates {len(df_wide.columns)} columns for {n_symbols} symbols!")
    print("This is why stacked format is preferred for large symbol sets.")


if __name__ == "__main__":
    # Run all examples
    example_basic_conversion()
    example_custom_columns()
    example_round_trip()
    example_pandas_workflow()
    example_multiasset_schema_integration()
    example_scalability_warning()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
