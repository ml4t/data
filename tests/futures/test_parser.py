"""Tests for futures data parser."""

from datetime import date

import polars as pl
import pytest

from ml4t.data.futures.parser import parse_quandl_chris


class TestParseQuandlCHRIS:
    """Tests for parse_quandl_chris function."""

    def test_parse_es_continuous_data(self):
        """Test parsing ES - should return continuous data with no duplicates."""
        data = parse_quandl_chris("ES")

        # Check no duplicates
        total_rows = len(data)
        unique_dates = data.select("date").unique().count().item()
        assert total_rows == unique_dates, "ES should have no duplicate dates"

        # Check columns
        expected_columns = {"date", "open", "high", "low", "close", "volume", "open_interest"}
        assert set(data.columns) == expected_columns

        # Check data types
        assert data["date"].dtype == pl.Date
        assert data["open"].dtype == pl.Float64
        assert data["volume"].dtype == pl.Float64

    def test_parse_cl_mixed_data(self):
        """Test parsing CL - should deduplicate mixed contracts to front month only."""
        data = parse_quandl_chris("CL")

        # Check no duplicates after parsing
        total_rows = len(data)
        unique_dates = data.select("date").unique().count().item()
        assert total_rows == unique_dates, "CL should be deduplicated to single row per date"

        # Check columns
        expected_columns = {"date", "open", "high", "low", "close", "volume", "open_interest"}
        assert set(data.columns) == expected_columns

    def test_front_month_selection_by_volume(self):
        """Test that front month is selected by highest volume on duplicate dates."""
        # For CL on 2014-03-03, we know there were two rows:
        # - Row 1: open=6387¢, volume=74,457 (back month)
        # - Row 2: open=103$, volume=282,447 (front month)
        # Parser should select Row 2 (higher volume)

        data = parse_quandl_chris("CL")
        row_2014_03_03 = data.filter(pl.col("date") == date(2014, 3, 3))

        assert len(row_2014_03_03) == 1, "Should have exactly one row for 2014-03-03"

        # Check that front month was selected (higher volume, price in dollars ~$103)
        open_price = row_2014_03_03["open"].item()
        volume = row_2014_03_03["volume"].item()

        # Front month should have volume ~282,447 and price ~$103 (not 6387¢)
        assert volume > 200_000, f"Expected front month volume >200k, got {volume}"
        assert 100 < open_price < 110, f"Expected front month price ~$103, got ${open_price}"

    def test_invalid_ticker(self):
        """Test that invalid ticker raises ValueError."""
        with pytest.raises(ValueError, match="Ticker.*not found"):
            parse_quandl_chris("INVALID_TICKER_12345")

    def test_data_sorted_by_date(self):
        """Test that returned data is sorted by date."""
        data = parse_quandl_chris("ES")

        dates = data["date"].to_list()
        assert dates == sorted(dates), "Data should be sorted by date"

    def test_no_null_ohlc_values(self):
        """Test that OHLC columns have no null values after parsing."""
        data = parse_quandl_chris("ES")

        # OHLC should never be null (fallback logic fills them)
        assert data["open"].null_count() == 0, "open should have no nulls"
        assert data["high"].null_count() == 0, "high should have no nulls"
        assert data["low"].null_count() == 0, "low should have no nulls"
        assert data["close"].null_count() == 0, "close should have no nulls"

        # volume should have no nulls
        assert data["volume"].null_count() == 0, "volume should have no nulls"

        # open_interest CAN be null (acceptable)
        # No assertion for open_interest

    def test_date_range_coverage(self):
        """Test that parsed data covers expected date range."""
        data = parse_quandl_chris("ES")

        min_date = data["date"].min()
        max_date = data["date"].max()

        # ES data in Quandl CHRIS should span ~2014-2021
        assert min_date.year >= 2014, f"Expected data to start >= 2014, got {min_date}"
        assert max_date.year <= 2021, f"Expected data to end <= 2021, got {max_date}"

    @pytest.mark.skip(reason="Requires specific Quandl CHRIS data format - skipped for CI")
    def test_realistic_price_ranges(self):
        """Test that prices are in realistic ranges (not obviously wrong units)."""
        # Get ES data for 2020 (known range ~2000-4000)
        data = parse_quandl_chris("ES")
        data_2020 = data.filter(pl.col("date").dt.year() == 2020)

        if len(data_2020) > 0:
            mean_close = data_2020["close"].mean()

            # E-mini S&P 500 in 2020 should be ~2000-4000 points
            assert 2000 < mean_close < 4000, (
                f"Expected ES 2020 mean price ~2000-4000, got {mean_close}. "
                "Might indicate price unit issue."
            )
