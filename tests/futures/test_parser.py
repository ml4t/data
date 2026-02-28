"""Tests for futures data parser."""

from datetime import date

import polars as pl
import pytest

from ml4t.data.futures.parser import parse_quandl_chris, parse_quandl_chris_raw


@pytest.fixture
def chris_data_path(tmp_path):
    data = pl.DataFrame(
        {
            "ticker": ["ES", "ES", "CL", "CL", "CL"],
            "date": [
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2014, 3, 3),
                date(2014, 3, 3),
                date(2014, 3, 4),
            ],
            "open": [3200.0, None, 6387.0, 103.0, 104.0],
            "high": [3210.0, None, 6400.0, 104.0, 105.0],
            "low": [3190.0, None, 6300.0, 102.0, 103.0],
            "close": [3205.0, None, 6390.0, 103.5, 104.5],
            "last": [3205.0, None, 6390.0, 103.5, 104.5],
            "settle": [3204.0, 3214.0, None, None, None],
            "volume": [100_000.0, 90_000.0, 74_457.0, 282_447.0, 200_000.0],
            "open_interest": [1000.0, 1100.0, 2000.0, 2100.0, 2200.0],
        }
    )
    path = tmp_path / "chris_futures.parquet"
    data.write_parquet(path)
    return path


class TestParseQuandlCHRIS:
    """Tests for parse_quandl_chris function."""

    def test_parse_es_continuous_data(self, chris_data_path):
        data = parse_quandl_chris("ES", data_path=chris_data_path)

        assert len(data) == 2
        assert data.select("date").n_unique() == 2
        assert set(data.columns) == {"date", "open", "high", "low", "close", "volume", "open_interest"}
        assert data["date"].dtype == pl.Date

    def test_parse_cl_mixed_data_deduplicates_to_front_month(self, chris_data_path):
        data = parse_quandl_chris("CL", data_path=chris_data_path)

        assert len(data) == 2
        assert data.select("date").n_unique() == 2

    def test_front_month_selection_by_volume(self, chris_data_path):
        data = parse_quandl_chris("CL", data_path=chris_data_path)
        row = data.filter(pl.col("date") == date(2014, 3, 3))

        assert len(row) == 1
        assert row["volume"].item() == 282_447.0
        assert row["open"].item() == 103.0

    def test_parse_raw_keeps_duplicate_dates(self, chris_data_path):
        data = parse_quandl_chris_raw("CL", data_path=chris_data_path)

        assert len(data) == 3
        assert data.filter(pl.col("date") == date(2014, 3, 3)).height == 2

    def test_invalid_ticker(self, chris_data_path):
        with pytest.raises(ValueError, match="Ticker.*not found"):
            parse_quandl_chris("INVALID_TICKER_12345", data_path=chris_data_path)

    def test_data_sorted_by_date(self, chris_data_path):
        data = parse_quandl_chris("ES", data_path=chris_data_path)
        assert data["date"].to_list() == sorted(data["date"].to_list())

    def test_no_null_ohlc_values_after_standardization(self, chris_data_path):
        data = parse_quandl_chris("ES", data_path=chris_data_path)

        assert data["open"].null_count() == 0
        assert data["high"].null_count() == 0
        assert data["low"].null_count() == 0
        assert data["close"].null_count() == 0
        assert data["volume"].null_count() == 0

    def test_missing_data_path_raises_actionable_error(self, tmp_path):
        missing = tmp_path / "missing.parquet"
        with pytest.raises(FileNotFoundError, match="legacy CHRIS dataset is no longer available"):
            parse_quandl_chris("ES", data_path=missing)
