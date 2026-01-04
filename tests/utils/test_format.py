"""Tests for format conversion utilities."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.data.utils.format import pivot_to_stacked, pivot_to_wide


@pytest.fixture
def stacked_df():
    """Create a sample stacked multi-asset DataFrame."""
    # 10 symbols, 5 days each = 50 rows
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
    timestamps = [
        datetime(2024, 1, 1, 9, 30, tzinfo=UTC),
        datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
        datetime(2024, 1, 3, 9, 30, tzinfo=UTC),
        datetime(2024, 1, 4, 9, 30, tzinfo=UTC),
        datetime(2024, 1, 5, 9, 30, tzinfo=UTC),
    ]

    data = []
    for i, ts in enumerate(timestamps):
        for j, symbol in enumerate(symbols):
            # Generate some simple test data
            base_price = 100 + j * 10
            day_offset = i * 2
            data.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": base_price + day_offset,
                    "high": base_price + day_offset + 2,
                    "low": base_price + day_offset - 1,
                    "close": base_price + day_offset + 1,
                    "volume": 1000000.0 + i * 100000,
                }
            )

    return pl.DataFrame(data)


@pytest.fixture
def stacked_df_small():
    """Create a small stacked DataFrame for basic testing."""
    return pl.DataFrame(
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
        }
    )


class TestPivotToWide:
    """Tests for pivot_to_wide function."""

    def test_basic_pivot(self, stacked_df_small):
        """Test basic pivot to wide format."""
        df_wide = pivot_to_wide(stacked_df_small)

        # Check shape: 2 timestamps, 1 timestamp col + 2 symbols × 5 value cols = 11 columns
        assert len(df_wide) == 2
        assert len(df_wide.columns) == 11  # timestamp + 10 pivoted columns

        # Check timestamp column exists
        assert "timestamp" in df_wide.columns

        # Check pivoted columns exist
        assert "close_AAPL" in df_wide.columns
        assert "close_MSFT" in df_wide.columns
        assert "volume_AAPL" in df_wide.columns
        assert "volume_MSFT" in df_wide.columns

    def test_pivot_preserves_values(self, stacked_df_small):
        """Test that pivoting preserves data values."""
        df_wide = pivot_to_wide(stacked_df_small)

        # Check first row values
        first_row = df_wide[0]
        assert first_row["close_AAPL"][0] == 101.0
        assert first_row["close_MSFT"][0] == 201.0
        assert first_row["volume_AAPL"][0] == 1000000.0
        assert first_row["volume_MSFT"][0] == 2000000.0

        # Check second row values
        second_row = df_wide[1]
        assert second_row["close_AAPL"][0] == 102.0
        assert second_row["close_MSFT"][0] == 202.0

    def test_pivot_with_custom_value_cols(self, stacked_df_small):
        """Test pivoting with custom value columns."""
        df_wide = pivot_to_wide(stacked_df_small, value_cols=["close", "volume"])

        # Should only have close and volume columns
        assert len(df_wide.columns) == 5  # timestamp + 2 symbols × 2 value cols

        assert "close_AAPL" in df_wide.columns
        assert "volume_AAPL" in df_wide.columns
        assert "open_AAPL" not in df_wide.columns
        assert "high_AAPL" not in df_wide.columns

    def test_pivot_with_10_symbols(self, stacked_df):
        """Test pivoting with 10 symbols."""
        df_wide = pivot_to_wide(stacked_df)

        # Check shape: 5 timestamps, 1 timestamp col + 10 symbols × 5 value cols = 51 columns
        assert len(df_wide) == 5
        assert len(df_wide.columns) == 51  # timestamp + 50 pivoted columns

        # Check all symbols appear
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        for symbol in symbols:
            assert f"close_{symbol}" in df_wide.columns
            assert f"volume_{symbol}" in df_wide.columns

    def test_pivot_sorted_by_timestamp(self, stacked_df):
        """Test that result is sorted by timestamp."""
        df_wide = pivot_to_wide(stacked_df)

        timestamps = df_wide["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    def test_pivot_missing_timestamp_column(self, stacked_df_small):
        """Test error when timestamp column is missing."""
        df_no_timestamp = stacked_df_small.drop("timestamp")

        with pytest.raises(ValueError, match="missing required column: 'timestamp'"):
            pivot_to_wide(df_no_timestamp)

    def test_pivot_missing_symbol_column(self, stacked_df_small):
        """Test error when symbol column is missing."""
        df_no_symbol = stacked_df_small.drop("symbol")

        with pytest.raises(ValueError, match="missing required column: 'symbol'"):
            pivot_to_wide(df_no_symbol)

    def test_pivot_duplicate_timestamp_symbol_pairs(self, stacked_df_small):
        """Test error when duplicate (timestamp, symbol) pairs exist."""
        # Add a duplicate row
        duplicate_row = stacked_df_small[0]
        df_with_duplicate = pl.concat([stacked_df_small, duplicate_row])

        with pytest.raises(ValueError, match="duplicate .* pairs"):
            pivot_to_wide(df_with_duplicate)

    def test_pivot_empty_dataframe(self):
        """Test pivoting an empty DataFrame."""
        df_empty = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "symbol": pl.Series([], dtype=pl.Utf8),
                "close": pl.Series([], dtype=pl.Float64),
            }
        )

        df_wide = pivot_to_wide(df_empty, value_cols=["close"])

        # Should have timestamp column but no data
        assert len(df_wide) == 0
        assert "timestamp" in df_wide.columns

    def test_pivot_single_symbol(self):
        """Test pivoting with a single symbol (trivial case)."""
        df_single = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
                "volume": [1000000.0, 1100000.0],
            }
        )

        df_wide = pivot_to_wide(df_single, value_cols=["close", "volume"])

        # Should have 3 columns: timestamp, close_AAPL, volume_AAPL
        assert len(df_wide.columns) == 3
        assert "close_AAPL" in df_wide.columns
        assert "volume_AAPL" in df_wide.columns

    def test_pivot_with_nulls(self):
        """Test pivoting with null values."""
        df_with_nulls = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "AAPL"],
                "close": [100.0, None],
                "volume": [1000000.0, 1100000.0],
            }
        )

        df_wide = pivot_to_wide(df_with_nulls, value_cols=["close", "volume"])

        # Should preserve null values
        assert df_wide["close_AAPL"][1] is None
        assert df_wide["volume_AAPL"][1] == 1100000.0

    def test_pivot_custom_column_names(self):
        """Test pivoting with custom column names."""
        df_custom = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "ticker": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        )

        df_wide = pivot_to_wide(
            df_custom,
            value_cols=["close"],
            timestamp_col="date",
            symbol_col="ticker",
        )

        assert "date" in df_wide.columns
        assert "close_AAPL" in df_wide.columns


class TestPivotToStacked:
    """Tests for pivot_to_stacked function."""

    def test_basic_unpivot(self, stacked_df_small):
        """Test basic unpivot from wide to stacked format."""
        df_wide = pivot_to_wide(stacked_df_small)
        df_stacked = pivot_to_stacked(df_wide)

        # Check shape: should match original
        assert len(df_stacked) == len(stacked_df_small)
        assert set(df_stacked.columns) == set(stacked_df_small.columns)

    def test_unpivot_preserves_values(self, stacked_df_small):
        """Test that unpivoting preserves data values."""
        df_wide = pivot_to_wide(stacked_df_small)
        df_stacked = pivot_to_stacked(df_wide)

        # Sort both for comparison
        original_sorted = stacked_df_small.sort(["timestamp", "symbol"])
        result_sorted = df_stacked.sort(["timestamp", "symbol"])

        # Check values match
        for col in ["close", "volume", "open"]:
            assert original_sorted[col].to_list() == result_sorted[col].to_list()

    def test_unpivot_with_10_symbols(self, stacked_df):
        """Test unpivoting with 10 symbols."""
        df_wide = pivot_to_wide(stacked_df)
        df_stacked = pivot_to_stacked(df_wide)

        # Check shape
        assert len(df_stacked) == len(stacked_df)
        assert "symbol" in df_stacked.columns

        # Check all symbols present
        symbols = df_stacked["symbol"].unique().to_list()
        assert len(symbols) == 10
        expected_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "NVDA",
            "JPM",
            "V",
            "WMT",
        ]
        assert set(symbols) == set(expected_symbols)

    def test_unpivot_sorted_by_timestamp_symbol(self, stacked_df):
        """Test that result is sorted by timestamp then symbol."""
        df_wide = pivot_to_wide(stacked_df)
        df_stacked = pivot_to_stacked(df_wide)

        # Check sorting
        timestamps = df_stacked["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

        # Within each timestamp, symbols should be sorted
        for ts in df_stacked["timestamp"].unique():
            symbols_at_ts = df_stacked.filter(pl.col("timestamp") == ts)["symbol"].to_list()
            assert symbols_at_ts == sorted(symbols_at_ts)

    def test_unpivot_missing_timestamp_column(self):
        """Test error when timestamp column is missing."""
        df_no_timestamp = pl.DataFrame({"close_AAPL": [100.0], "close_MSFT": [200.0]})

        with pytest.raises(ValueError, match="missing required column: 'timestamp'"):
            pivot_to_stacked(df_no_timestamp)

    def test_unpivot_no_pivoted_columns(self):
        """Test error when no pivoted columns found."""
        df_only_timestamp = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
            }
        )

        with pytest.raises(ValueError, match="No pivoted columns found"):
            pivot_to_stacked(df_only_timestamp)

    def test_unpivot_empty_dataframe(self):
        """Test unpivoting an empty wide DataFrame."""
        df_empty_wide = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "close_AAPL": pl.Series([], dtype=pl.Float64),
                "close_MSFT": pl.Series([], dtype=pl.Float64),
            }
        )

        df_stacked = pivot_to_stacked(df_empty_wide)

        # Should have timestamp, symbol, and value columns
        assert "timestamp" in df_stacked.columns
        assert "symbol" in df_stacked.columns
        assert "close" in df_stacked.columns
        assert len(df_stacked) == 0

    def test_unpivot_custom_timestamp_column(self):
        """Test unpivoting with custom timestamp column name."""
        df_wide = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1, tzinfo=UTC)],
                "close_AAPL": [100.0],
                "close_MSFT": [200.0],
            }
        )

        df_stacked = pivot_to_stacked(df_wide, timestamp_col="date")

        assert "date" in df_stacked.columns
        assert "symbol" in df_stacked.columns
        assert "close" in df_stacked.columns

    def test_unpivot_with_nulls(self):
        """Test unpivoting with null values."""
        df_wide = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "close_AAPL": [100.0, None],
                "close_MSFT": [200.0, 201.0],
            }
        )

        df_stacked = pivot_to_stacked(df_wide)

        # Should preserve null values
        aapl_row_2 = df_stacked.filter(
            (pl.col("timestamp") == datetime(2024, 1, 2, tzinfo=UTC)) & (pl.col("symbol") == "AAPL")
        )
        assert aapl_row_2["close"][0] is None

    def test_unpivot_symbol_with_underscore(self):
        """Test unpivoting symbols that contain underscores."""
        df_wide = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "close_BRK_B": [300.0],
                "volume_BRK_B": [1000000.0],
            }
        )

        df_stacked = pivot_to_stacked(df_wide)

        # Should correctly parse 'BRK_B' as the symbol
        assert "BRK_B" in df_stacked["symbol"].to_list()
        assert df_stacked["close"][0] == 300.0
        assert df_stacked["volume"][0] == 1000000.0


class TestRoundTrip:
    """Tests for round-trip conversion (stacked -> wide -> stacked)."""

    def test_round_trip_preserves_data(self, stacked_df):
        """Test that round-trip conversion preserves all data."""
        # Original stacked
        original = stacked_df.sort(["timestamp", "symbol"])

        # Convert to wide
        df_wide = pivot_to_wide(original)

        # Convert back to stacked
        df_back = pivot_to_stacked(df_wide).sort(["timestamp", "symbol"])

        # Should match original
        assert len(original) == len(df_back)
        assert set(original.columns) == set(df_back.columns)

        # Check all values match
        for col in original.columns:
            original_values = original[col].to_list()
            back_values = df_back[col].to_list()
            assert original_values == back_values, f"Column {col} values differ"

    def test_round_trip_with_custom_columns(self, stacked_df_small):
        """Test round-trip with custom value columns."""
        original = stacked_df_small.sort(["timestamp", "symbol"])

        # Convert with only specific columns
        df_wide = pivot_to_wide(original, value_cols=["close", "volume"])
        df_back = pivot_to_stacked(df_wide).sort(["timestamp", "symbol"])

        # Should preserve close and volume
        assert original["close"].to_list() == df_back["close"].to_list()
        assert original["volume"].to_list() == df_back["volume"].to_list()

        # Should NOT have other columns
        assert "open" not in df_back.columns
        assert "high" not in df_back.columns

    def test_round_trip_single_symbol(self):
        """Test round-trip with single symbol."""
        original = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        ).sort(["timestamp", "symbol"])

        df_wide = pivot_to_wide(original, value_cols=["close"])
        df_back = pivot_to_stacked(df_wide).sort(["timestamp", "symbol"])

        assert original["close"].to_list() == df_back["close"].to_list()

    def test_round_trip_with_nulls(self):
        """Test round-trip preserves null values."""
        original = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "close": [100.0, None, 101.0, 201.0],
            }
        ).sort(["timestamp", "symbol"])

        df_wide = pivot_to_wide(original, value_cols=["close"])
        df_back = pivot_to_stacked(df_wide).sort(["timestamp", "symbol"])

        # Compare with null-aware comparison
        for i in range(len(original)):
            orig_val = original["close"][i]
            back_val = df_back["close"][i]
            if orig_val is None:
                assert back_val is None
            else:
                assert orig_val == back_val

    def test_round_trip_many_timestamps(self):
        """Test round-trip with many timestamps."""
        # Create data with 100 timestamps and 10 symbols
        # Use day increments instead of minute increments to avoid overflow
        timestamps = [datetime(2024, 1, 1 + (i // 24), i % 24, 0, tzinfo=UTC) for i in range(100)]
        symbols = [f"SYM{i:02d}" for i in range(10)]

        data = []
        for ts in timestamps:
            for symbol in symbols:
                data.append({"timestamp": ts, "symbol": symbol, "close": 100.0})

        original = pl.DataFrame(data).sort(["timestamp", "symbol"])

        df_wide = pivot_to_wide(original, value_cols=["close"])
        df_back = pivot_to_stacked(df_wide).sort(["timestamp", "symbol"])

        assert len(original) == len(df_back)
        assert original["close"].to_list() == df_back["close"].to_list()
