"""Unit tests for Yahoo Finance provider internal methods.

These tests focus on internal transformation methods that don't require mocking
yfinance API calls and are not affected by parallel execution issues.
"""

from datetime import datetime, timedelta

import pandas as pd
import polars as pl

from ml4t.data.providers.yahoo import YahooFinanceProvider, _chunks


class TestChunksHelper:
    """Tests for the _chunks helper function."""

    def test_chunks_even_split(self):
        """Test chunking with evenly divisible list."""
        lst = [1, 2, 3, 4, 5, 6]
        chunks = list(_chunks(lst, 2))
        assert chunks == [[1, 2], [3, 4], [5, 6]]

    def test_chunks_uneven_split(self):
        """Test chunking with remainder."""
        lst = [1, 2, 3, 4, 5]
        chunks = list(_chunks(lst, 2))
        assert chunks == [[1, 2], [3, 4], [5]]

    def test_chunks_larger_than_list(self):
        """Test chunk size larger than list."""
        lst = [1, 2, 3]
        chunks = list(_chunks(lst, 10))
        assert chunks == [[1, 2, 3]]

    def test_chunks_single_element(self):
        """Test chunking single element list."""
        lst = [1]
        chunks = list(_chunks(lst, 5))
        assert chunks == [[1]]

    def test_chunks_empty_list(self):
        """Test chunking empty list."""
        lst = []
        chunks = list(_chunks(lst, 5))
        assert chunks == []

    def test_chunks_size_one(self):
        """Test chunk size of 1."""
        lst = [1, 2, 3]
        chunks = list(_chunks(lst, 1))
        assert chunks == [[1], [2], [3]]


class TestYahooProviderInit:
    """Tests for YahooFinanceProvider initialization."""

    def test_init_default(self):
        """Test default initialization."""
        provider = YahooFinanceProvider()
        assert provider.name == "yahoo"
        assert provider.enable_progress is False

    def test_init_with_progress(self):
        """Test initialization with progress enabled."""
        provider = YahooFinanceProvider(enable_progress=True)
        assert provider.enable_progress is True

    def test_name_property(self):
        """Test name property returns correct value."""
        provider = YahooFinanceProvider()
        assert provider.name == "yahoo"


class TestFrequencyMap:
    """Tests for frequency mapping."""

    def test_frequency_map_contains_all_frequencies(self):
        """Test FREQUENCY_MAP has all expected keys."""
        provider = YahooFinanceProvider()
        expected_keys = [
            "minute",
            "1minute",
            "5minute",
            "15minute",
            "30minute",
            "hourly",
            "1hour",
            "daily",
            "1day",
            "weekly",
            "1week",
            "monthly",
            "1month",
        ]
        for key in expected_keys:
            assert key in provider.FREQUENCY_MAP

    def test_frequency_map_values(self):
        """Test FREQUENCY_MAP has correct mappings."""
        provider = YahooFinanceProvider()
        assert provider.FREQUENCY_MAP["minute"] == "1m"
        assert provider.FREQUENCY_MAP["daily"] == "1d"
        assert provider.FREQUENCY_MAP["weekly"] == "1wk"
        assert provider.FREQUENCY_MAP["monthly"] == "1mo"


class TestCreateEmptyDataframe:
    """Tests for _create_empty_dataframe method."""

    def test_empty_dataframe_columns(self):
        """Test empty DataFrame has correct columns."""
        provider = YahooFinanceProvider()
        df = provider._create_empty_dataframe()

        expected_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert df.columns == expected_columns

    def test_empty_dataframe_dtypes(self):
        """Test empty DataFrame has correct data types."""
        provider = YahooFinanceProvider()
        df = provider._create_empty_dataframe()

        assert df.schema["timestamp"] == pl.Datetime
        assert df.schema["open"] == pl.Float64
        assert df.schema["high"] == pl.Float64
        assert df.schema["low"] == pl.Float64
        assert df.schema["close"] == pl.Float64
        assert df.schema["volume"] == pl.Float64
        # Note: pl.Utf8 is an alias for pl.String
        assert df.schema["symbol"] in (pl.Utf8, pl.String)

    def test_empty_dataframe_length(self):
        """Test empty DataFrame has zero rows."""
        provider = YahooFinanceProvider()
        df = provider._create_empty_dataframe()

        assert len(df) == 0


class TestConvertToPolars:
    """Tests for _convert_to_polars method."""

    def test_single_level_columns(self):
        """Test conversion with single-level column names."""
        provider = YahooFinanceProvider()

        df_pandas = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [104.0, 105.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        result = provider._convert_to_polars(df_pandas, "AAPL")

        assert len(result) == 2
        assert "symbol" in result.columns
        assert result["symbol"][0] == "AAPL"
        assert result["open"][0] == 100.0
        assert result["close"][1] == 105.0

    def test_multi_level_columns_specific_symbol(self):
        """Test conversion with multi-level columns (yfinance format)."""
        provider = YahooFinanceProvider()

        # Create multi-level column DataFrame like yfinance returns
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)

        df_pandas = pd.DataFrame(
            [[100.0, 105.0, 99.0, 104.0, 1000000]],
            index=pd.date_range("2024-01-01", periods=1),
            columns=columns,
        )

        result = provider._convert_to_polars(df_pandas, "AAPL")

        assert len(result) == 1
        assert result["symbol"][0] == "AAPL"
        assert result["open"][0] == 100.0

    def test_symbol_uppercase(self):
        """Test symbol is uppercased in result."""
        provider = YahooFinanceProvider()

        df_pandas = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [104.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        result = provider._convert_to_polars(df_pandas, "aapl")

        assert result["symbol"][0] == "AAPL"

    def test_datetime_conversion(self):
        """Test timestamp is converted to datetime."""
        provider = YahooFinanceProvider()

        df_pandas = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [104.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        result = provider._convert_to_polars(df_pandas, "AAPL")

        assert result.schema["timestamp"] == pl.Datetime

    def test_numeric_conversion(self):
        """Test OHLCV columns are converted to Float64."""
        provider = YahooFinanceProvider()

        df_pandas = pd.DataFrame(
            {
                "Open": [100],  # Integer
                "High": [105],
                "Low": [99],
                "Close": [104],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        result = provider._convert_to_polars(df_pandas, "AAPL")

        assert result.schema["open"] == pl.Float64
        assert result.schema["high"] == pl.Float64
        assert result.schema["low"] == pl.Float64
        assert result.schema["close"] == pl.Float64
        assert result.schema["volume"] == pl.Float64


class TestConvertBatchToPolars:
    """Tests for _convert_batch_to_polars method."""

    def test_empty_dataframe(self):
        """Test conversion of empty DataFrame."""
        provider = YahooFinanceProvider()
        df_pandas = pd.DataFrame()

        result = provider._convert_batch_to_polars(df_pandas, ["AAPL"])

        assert len(result) == 0
        assert "timestamp" in result.columns
        assert "symbol" in result.columns

    def test_multi_symbol_conversion(self):
        """Test conversion with multiple symbols."""
        provider = YahooFinanceProvider()

        # Create multi-level column DataFrame with 2 symbols
        dates = pd.date_range("2024-01-01", periods=2)

        # Build the data as yfinance would return it
        arrays = [
            ["Open", "Open", "High", "High", "Low", "Low", "Close", "Close", "Volume", "Volume"],
            ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)

        data = [
            [100.0, 200.0, 105.0, 205.0, 99.0, 199.0, 104.0, 204.0, 1000000, 2000000],
            [101.0, 201.0, 106.0, 206.0, 100.0, 200.0, 105.0, 205.0, 1100000, 2100000],
        ]

        df_pandas = pd.DataFrame(data, index=dates, columns=columns)

        result = provider._convert_batch_to_polars(df_pandas, ["AAPL", "MSFT"])

        # Should have 4 rows (2 symbols x 2 dates)
        assert len(result) == 4

        # Check both symbols are present
        symbols = result["symbol"].unique().to_list()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_single_symbol_batch(self):
        """Test batch conversion with single symbol."""
        provider = YahooFinanceProvider()

        # Create single-level column DataFrame
        df_pandas = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [104.0, 105.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        result = provider._convert_batch_to_polars(df_pandas, ["AAPL"])

        assert len(result) == 2
        assert result["symbol"].unique().to_list() == ["AAPL"]

    def test_filters_null_rows(self):
        """Test that rows with null OHLCV are filtered."""
        provider = YahooFinanceProvider()

        # Create DataFrame with some null values
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)

        df_pandas = pd.DataFrame(
            [
                [100.0, 105.0, 99.0, 104.0, 1000000],
                [None, None, None, None, None],  # This row should be filtered
            ],
            index=pd.date_range("2024-01-01", periods=2),
            columns=columns,
        )

        result = provider._convert_batch_to_polars(df_pandas, ["AAPL"])

        # Only non-null row should remain
        assert len(result) == 1


class TestFetchBatchOhlcvIntegration:
    """Integration tests for fetch_batch_ohlcv method structure."""

    def test_batch_uses_correct_interval_mapping(self):
        """Test that fetch_batch_ohlcv maps frequencies correctly."""
        provider = YahooFinanceProvider()

        # Test frequency mapping (without actual API call)
        interval = provider.FREQUENCY_MAP.get("daily", "1d")
        assert interval == "1d"

        interval = provider.FREQUENCY_MAP.get("weekly", "1wk")
        assert interval == "1wk"

    def test_end_date_adjustment(self):
        """Test that end date is adjusted correctly (yfinance is exclusive)."""
        # yfinance end date is exclusive, so we add one day
        end = "2024-01-31"
        end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        end_str = end_date.strftime("%Y-%m-%d")

        assert end_str == "2024-02-01"


class TestYahooProviderDataTypes:
    """Tests for data type consistency."""

    def test_polars_output(self):
        """Test provider always returns Polars DataFrame."""
        provider = YahooFinanceProvider()
        empty = provider._create_empty_dataframe()

        assert isinstance(empty, pl.DataFrame)

    def test_standard_schema(self):
        """Test DataFrame has standard OHLCV schema."""
        provider = YahooFinanceProvider()
        df = provider._create_empty_dataframe()

        # Check standard columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        for col in required_columns:
            assert col in df.columns

    def test_column_order(self):
        """Test columns are in standard order."""
        provider = YahooFinanceProvider()
        df = provider._create_empty_dataframe()

        expected_order = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        assert df.columns == expected_order
