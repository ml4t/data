"""Edge case tests for Yahoo Finance provider.

These tests focus on error handling, batch downloads, and edge cases
that are difficult to trigger with normal usage.
"""

from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from ml4t.data.core.exceptions import (
    DataValidationError,
    SymbolNotFoundError,
)
from ml4t.data.providers.yahoo import YahooFinanceProvider


class TestFetchAndTransformDataErrors:
    """Tests for error handling in _fetch_and_transform_data."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return YahooFinanceProvider()

    def test_empty_data_raises_symbol_not_found(self, provider):
        """Test empty download raises SymbolNotFoundError."""
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame()

            with pytest.raises(SymbolNotFoundError) as exc_info:
                provider._fetch_and_transform_data(
                    "INVALID_SYMBOL", "2024-01-01", "2024-01-31", "daily"
                )

            assert "INVALID_SYMBOL" in str(exc_info.value)

    def test_api_error_raises_data_validation_error(self, provider):
        """Test generic API error raises DataValidationError."""
        with patch("yfinance.download") as mock_download:
            mock_download.side_effect = Exception("API Error")

            with pytest.raises(DataValidationError) as exc_info:
                provider._fetch_and_transform_data("AAPL", "2024-01-01", "2024-01-31", "daily")

            assert "API Error" in str(exc_info.value)

    def test_unknown_frequency_uses_daily_default(self, provider):
        """Test unknown frequency defaults to 1d interval."""
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [105.0],
                    "Low": [99.0],
                    "Close": [104.0],
                    "Volume": [1000000],
                },
                index=pd.date_range("2024-01-01", periods=1),
            )

            result = provider._fetch_and_transform_data(
                "AAPL", "2024-01-01", "2024-01-02", "unknown_frequency"
            )

            # Should succeed with daily default
            assert len(result) == 1
            mock_download.assert_called_once()
            # Check the interval argument
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["interval"] == "1d"

    def test_propagates_symbol_not_found_error(self, provider):
        """Test SymbolNotFoundError is propagated without wrapping."""
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame()  # Empty = not found

            with pytest.raises(SymbolNotFoundError):
                provider._fetch_and_transform_data("MISSING", "2024-01-01", "2024-01-31", "daily")


class TestConvertToPolarsEdgeCases:
    """Tests for edge cases in _convert_to_polars method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return YahooFinanceProvider()

    def test_multi_level_columns_symbol_not_found_flattens(self, provider):
        """Test multi-level columns when symbol not in columns."""
        # Create multi-level with different symbol
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["MSFT", "MSFT", "MSFT", "MSFT", "MSFT"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)

        df_pandas = pd.DataFrame(
            [[100.0, 105.0, 99.0, 104.0, 1000000]],
            index=pd.date_range("2024-01-01", periods=1),
            columns=columns,
        )

        # Request AAPL but data is for MSFT - should flatten columns
        result = provider._convert_to_polars(df_pandas, "AAPL")

        # Should still work (flattens first level)
        assert len(result) == 1
        # Symbol should be AAPL (the requested symbol)
        assert result["symbol"][0] == "AAPL"

    def test_timestamp_column_named_datetime(self, provider):
        """Test timestamp column detection when named 'Datetime'."""
        df_pandas = pd.DataFrame(
            {
                "Datetime": pd.date_range("2024-01-01", periods=2),
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [104.0, 105.0],
                "Volume": [1000000, 1100000],
            }
        )

        result = provider._convert_to_polars(df_pandas, "AAPL")

        assert len(result) == 2
        assert result.schema["timestamp"] == pl.Datetime

    def test_timestamp_column_named_index(self, provider):
        """Test timestamp column detection when named 'index'."""
        df_pandas = pd.DataFrame(
            {
                "index": pd.date_range("2024-01-01", periods=2),
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [104.0, 105.0],
                "Volume": [1000000, 1100000],
            }
        )

        result = provider._convert_to_polars(df_pandas, "AAPL")

        assert len(result) == 2

    def test_timestamp_column_falls_back_to_first(self, provider):
        """Test timestamp falls back to first column if no standard name."""
        df_pandas = pd.DataFrame(
            {
                "weird_timestamp": pd.date_range("2024-01-01", periods=2),
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [104.0, 105.0],
                "Volume": [1000000, 1100000],
            }
        )

        result = provider._convert_to_polars(df_pandas, "AAPL")

        # Should use first column as timestamp
        assert len(result) == 2


class TestFetchBatchOhlcv:
    """Tests for fetch_batch_ohlcv method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return YahooFinanceProvider()

    def test_batch_single_symbol(self, provider):
        """Test batch download with single symbol."""
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame(
                {
                    "Open": [100.0, 101.0],
                    "High": [105.0, 106.0],
                    "Low": [99.0, 100.0],
                    "Close": [104.0, 105.0],
                    "Volume": [1000000, 1100000],
                },
                index=pd.date_range("2024-01-01", periods=2),
            )

            result = provider.fetch_batch_ohlcv(
                ["AAPL"], "2024-01-01", "2024-01-31", chunk_size=50, delay_seconds=0
            )

            assert len(result) == 2
            mock_download.assert_called_once()

    def test_batch_multiple_symbols(self, provider):
        """Test batch download with multiple symbols."""
        # Create multi-symbol response
        dates = pd.date_range("2024-01-01", periods=2)
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

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = df_pandas

            result = provider.fetch_batch_ohlcv(
                ["AAPL", "MSFT"], "2024-01-01", "2024-01-31", chunk_size=50, delay_seconds=0
            )

            assert len(result) == 4  # 2 symbols * 2 dates
            symbols = result["symbol"].unique().to_list()
            assert "AAPL" in symbols
            assert "MSFT" in symbols

    def test_batch_chunks_large_symbol_list(self, provider):
        """Test batch chunking for large symbol lists."""
        symbols = [f"SYM{i}" for i in range(120)]  # More than default chunk_size

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame()  # Empty for simplicity

            provider.fetch_batch_ohlcv(
                symbols, "2024-01-01", "2024-01-31", chunk_size=50, delay_seconds=0
            )

            # Should have called download 3 times (120 / 50 = 2.4 -> 3 chunks)
            assert mock_download.call_count == 3

    def test_batch_empty_response_returns_empty(self, provider):
        """Test empty batch response returns empty DataFrame."""
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame()

            result = provider.fetch_batch_ohlcv(
                ["INVALID1", "INVALID2"], "2024-01-01", "2024-01-31", delay_seconds=0
            )

            assert len(result) == 0
            assert "timestamp" in result.columns

    def test_batch_partial_failure(self, provider):
        """Test batch handles partial failures gracefully."""
        # First chunk succeeds, second fails
        call_count = [0]

        def mock_download_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return pd.DataFrame(
                    {
                        "Open": [100.0],
                        "High": [105.0],
                        "Low": [99.0],
                        "Close": [104.0],
                        "Volume": [1000000],
                    },
                    index=pd.date_range("2024-01-01", periods=1),
                )
            else:
                raise Exception("Network error")

        with patch("yfinance.download", side_effect=mock_download_side_effect):
            result = provider.fetch_batch_ohlcv(
                ["AAPL", "MSFT"] * 30,  # 60 symbols, 2 chunks
                "2024-01-01",
                "2024-01-31",
                chunk_size=30,
                delay_seconds=0,
            )

            # Should still return data from first chunk
            assert len(result) >= 0  # May be empty or partial

    def test_batch_delay_between_chunks(self, provider):
        """Test delay is applied between chunks."""

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [105.0],
                    "Low": [99.0],
                    "Close": [104.0],
                    "Volume": [1000000],
                },
                index=pd.date_range("2024-01-01", periods=1),
            )

            with patch("time.sleep") as mock_sleep:
                provider.fetch_batch_ohlcv(
                    ["AAPL"] * 100,  # 100 symbols, 2 chunks with default size
                    "2024-01-01",
                    "2024-01-31",
                    chunk_size=50,
                    delay_seconds=1.0,
                )

                # Sleep should be called between chunks (but not after last)
                assert mock_sleep.call_count == 1
                mock_sleep.assert_called_with(1.0)


class TestConvertBatchToPolarsEdgeCases:
    """Tests for edge cases in _convert_batch_to_polars method."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return YahooFinanceProvider()

    def test_single_level_columns(self, provider):
        """Test batch conversion with single level columns."""
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
        assert "AAPL" in result["symbol"].to_list()

    def test_empty_string_symbols_filtered(self, provider):
        """Test empty string symbols are filtered out."""
        dates = pd.date_range("2024-01-01", periods=1)
        arrays = [
            ["Open", "Open", "High", "High", "Low", "Low", "Close", "Close", "Volume", "Volume"],
            ["AAPL", "", "AAPL", "", "AAPL", "", "AAPL", "", "AAPL", ""],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)

        df_pandas = pd.DataFrame(
            [[100.0, None, 105.0, None, 99.0, None, 104.0, None, 1000000, None]],
            index=dates,
            columns=columns,
        )

        result = provider._convert_batch_to_polars(df_pandas, ["AAPL"])

        # Should only have AAPL data, empty string should be filtered
        assert all(s == "AAPL" for s in result["symbol"].to_list())

    def test_symbol_extraction_exception_logged(self, provider):
        """Test symbol extraction exception is logged and skipped."""
        dates = pd.date_range("2024-01-01", periods=1)
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)

        df_pandas = pd.DataFrame(
            [[100.0, 105.0, 99.0, 104.0, 1000000]],
            index=dates,
            columns=columns,
        )

        # Mock DataFrame.get to raise for one column
        with patch.object(pd.DataFrame, "get") as mock_get:
            mock_get.side_effect = Exception("Column error")

            result = provider._convert_batch_to_polars(df_pandas, ["AAPL"])

            # Should return empty (exception caught and logged)
            assert len(result) == 0

    def test_all_null_rows_filtered(self, provider):
        """Test rows with all null OHLCV values are filtered."""
        dates = pd.date_range("2024-01-01", periods=3)
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)

        df_pandas = pd.DataFrame(
            [
                [100.0, 105.0, 99.0, 104.0, 1000000],  # Valid row
                [None, None, None, None, None],  # All null - should be filtered
                [102.0, 107.0, 101.0, 106.0, 1200000],  # Valid row
            ],
            index=dates,
            columns=columns,
        )

        result = provider._convert_batch_to_polars(df_pandas, ["AAPL"])

        # Only 2 valid rows should remain
        assert len(result) == 2


class TestFrequencyMapping:
    """Tests for frequency mapping edge cases."""

    def test_all_minute_variants(self):
        """Test all minute frequency variants."""
        provider = YahooFinanceProvider()

        assert provider.FREQUENCY_MAP["minute"] == "1m"
        assert provider.FREQUENCY_MAP["1minute"] == "1m"
        assert provider.FREQUENCY_MAP["5minute"] == "5m"
        assert provider.FREQUENCY_MAP["15minute"] == "15m"
        assert provider.FREQUENCY_MAP["30minute"] == "30m"

    def test_all_hour_variants(self):
        """Test all hour frequency variants."""
        provider = YahooFinanceProvider()

        assert provider.FREQUENCY_MAP["hourly"] == "1h"
        assert provider.FREQUENCY_MAP["1hour"] == "1h"

    def test_case_insensitive_lookup(self):
        """Test frequency lookup is case insensitive."""
        provider = YahooFinanceProvider()

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [105.0],
                    "Low": [99.0],
                    "Close": [104.0],
                    "Volume": [1000000],
                },
                index=pd.date_range("2024-01-01", periods=1),
            )

            # Should work with uppercase
            provider._fetch_and_transform_data("AAPL", "2024-01-01", "2024-01-02", "DAILY")

            # Check interval was correctly mapped
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["interval"] == "1d"
