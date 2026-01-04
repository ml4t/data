"""Integration tests for format conversion with MultiAssetSchema."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from ml4t.data.core.schemas import MultiAssetSchema
from ml4t.data.utils.format import pivot_to_stacked, pivot_to_wide


class TestMultiAssetSchemaIntegration:
    """Test format conversion with MultiAssetSchema validation."""

    def test_pivot_preserves_schema_validity(self):
        """Test that pivoting and unpivoting preserves schema validity."""
        # Create valid multi-asset DataFrame
        df_stacked = MultiAssetSchema.create_empty("equities")

        # Add some data
        df_stacked = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "open": [100.0, 200.0, 101.0, 201.0],
                "high": [102.0, 202.0, 103.0, 203.0],
                "low": [99.0, 199.0, 100.0, 200.0],
                "close": [101.0, 201.0, 102.0, 202.0],
                "volume": [1000000.0, 2000000.0, 1100000.0, 2100000.0],
                "dividends": [0.0, 0.0, 0.5, 0.0],
                "splits": [1.0, 1.0, 1.0, 1.0],
                "adjusted_close": [101.0, 201.0, 102.0, 202.0],
            }
        )

        # Validate original
        assert MultiAssetSchema.validate(df_stacked, strict=True)

        # Convert to wide
        df_wide = pivot_to_wide(df_stacked)

        # Wide format doesn't have symbol column, so validation would fail
        # That's expected - wide format is NOT the canonical format

        # Convert back to stacked
        df_back = pivot_to_stacked(df_wide)

        # After unpivoting, should be valid again
        assert MultiAssetSchema.validate(df_back, strict=True)

    def test_round_trip_with_standardize_order(self):
        """Test round-trip with MultiAssetSchema.standardize_order."""
        # Create multi-asset DataFrame
        df_original = pl.DataFrame(
            {
                "volume": [1000000.0, 2000000.0, 1100000.0, 2100000.0],
                "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "close": [101.0, 201.0, 102.0, 202.0],
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                ],
                "open": [100.0, 200.0, 101.0, 201.0],
                "high": [102.0, 202.0, 103.0, 203.0],
                "low": [99.0, 199.0, 100.0, 200.0],
            }
        )

        # Standardize before conversion
        df_standardized = MultiAssetSchema.standardize_order(df_original)

        # Convert to wide and back
        df_wide = pivot_to_wide(df_standardized)
        df_back = pivot_to_stacked(df_wide)

        # Standardize result
        df_back_standardized = MultiAssetSchema.standardize_order(df_back)

        # Should match original standardized
        assert df_standardized.equals(df_back_standardized)

    def test_add_symbol_column_then_pivot(self):
        """Test converting single-asset to multi-asset then pivoting."""
        # Create single-asset DataFrame
        df_single = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                ],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000000.0, 1100000.0],
            }
        )

        # Add symbol column
        df_multi = MultiAssetSchema.add_symbol_column(df_single, "AAPL")

        # Validate
        assert MultiAssetSchema.validate(df_multi, strict=True)

        # Pivot to wide
        df_wide = pivot_to_wide(df_multi)

        # Check columns
        assert "timestamp" in df_wide.columns
        assert "close_AAPL" in df_wide.columns
        assert "volume_AAPL" in df_wide.columns

    def test_cast_to_schema_after_unpivot(self):
        """Test that unpivoted data can be cast to proper schema."""
        # Create wide DataFrame with all required OHLCV columns
        df_wide = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
                ],
                "open_AAPL": [99, 100],  # Int instead of Float
                "open_MSFT": [199, 200],
                "high_AAPL": [102, 103],
                "high_MSFT": [202, 203],
                "low_AAPL": [98, 99],
                "low_MSFT": [198, 199],
                "close_AAPL": [100, 101],
                "close_MSFT": [200, 201],
                "volume_AAPL": [1000000, 1100000],
                "volume_MSFT": [2000000, 2100000],
            }
        )

        # Unpivot
        df_stacked = pivot_to_stacked(df_wide)

        # Cast to schema
        df_cast = MultiAssetSchema.cast_to_schema(df_stacked)

        # Should now have proper Float64 types
        assert df_cast["open"].dtype == pl.Float64
        assert df_cast["high"].dtype == pl.Float64
        assert df_cast["low"].dtype == pl.Float64
        assert df_cast["close"].dtype == pl.Float64
        assert df_cast["volume"].dtype == pl.Float64

        # Validate
        assert MultiAssetSchema.validate(df_cast, strict=True)
