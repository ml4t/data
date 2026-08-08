"""Tests for futures data parser."""

from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from ml4t.data.futures.parser import parse_quandl_chris, parse_quandl_chris_raw
from ml4t.data.futures.schema import MAJOR_CONTRACTS


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
        assert set(data.columns) == {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        }
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
        assert data["close"].to_list() == [63.9, 103.5, 104.5]

    def test_mixed_crude_units_use_declared_normalized_range(self, tmp_path):
        """Sub-$10, negative, and ordinary dollar quotes normalize without a magnitude cutoff."""
        data = pl.DataFrame(
            {
                "ticker": ["CL", "CL", "CL"],
                "date": [date(1998, 12, 1), date(2020, 4, 20), date(2024, 1, 2)],
                "open": [975.0, -3763.0, 72.0],
                "high": [980.0, -3500.0, 73.0],
                "low": [970.0, -4000.0, 71.0],
                "close": [975.0, -3763.0, 72.0],
                "last": [975.0, -3763.0, 72.0],
                "settle": [975.0, -3763.0, 72.0],
                "volume": [1000.0, 1000.0, 1000.0],
                "open_interest": [100.0, 100.0, 100.0],
            }
        )
        path = tmp_path / "mixed-crude.parquet"
        data.write_parquet(path)

        parsed = parse_quandl_chris_raw("CL", data_path=path)

        assert parsed["close"].to_list() == [9.75, -37.63, 72.0]

    def test_mixed_crude_units_reject_values_outside_declared_range(self, tmp_path):
        """Ambiguous source corruption fails instead of returning a plausible wrong scale."""
        data = pl.DataFrame(
            {
                "ticker": ["CL"],
                "date": [date(2024, 1, 2)],
                "open": [50_000.0],
                "high": [50_000.0],
                "low": [50_000.0],
                "close": [50_000.0],
                "last": [50_000.0],
                "settle": [50_000.0],
                "volume": [1000.0],
                "open_interest": [100.0],
            }
        )
        path = tmp_path / "invalid-crude.parquet"
        data.write_parquet(path)

        with pytest.raises(ValueError, match="fit neither dollars nor cents"):
            parse_quandl_chris_raw("CL", data_path=path)

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

    def test_large_index_price_is_not_reinterpreted_as_cents(self, tmp_path):
        data = pl.DataFrame(
            {
                "ticker": ["ES"],
                "date": [date(2026, 1, 2)],
                "open": [5000.0],
                "high": [5010.0],
                "low": [4990.0],
                "close": [5004.0],
                "last": [5004.0],
                "settle": [5004.0],
                "volume": [100_000.0],
                "open_interest": [1_000.0],
            }
        )
        path = tmp_path / "large-index-price.parquet"
        data.write_parquet(path)

        parsed = parse_quandl_chris("ES", data_path=path)

        assert parsed["close"].item() == 5004.0

    def test_declared_cents_unit_is_applied_without_magnitude_guessing(self, tmp_path):
        data = pl.DataFrame(
            {
                "ticker": ["ES", "ES"],
                "date": [date(2026, 1, 2), date(2026, 1, 3)],
                "open": [5004.0, 99.0],
                "high": [5004.0, 99.0],
                "low": [5004.0, 99.0],
                "close": [5004.0, 99.0],
                "last": [5004.0, 99.0],
                "settle": [5004.0, 99.0],
                "volume": [100_000.0, 100_000.0],
                "open_interest": [1_000.0, 1_000.0],
            }
        )
        path = tmp_path / "declared-cents.parquet"
        data.write_parquet(path)
        cents_spec = replace(MAJOR_CONTRACTS["ES"], price_quote_unit="cents")

        parsed = parse_quandl_chris("ES", data_path=path, contract_spec=cents_spec)

        assert parsed["close"].to_list() == [50.04, 0.99]

    def test_parser_output_converts_to_contract_value_once(self, tmp_path):
        data = pl.DataFrame(
            {
                "ticker": ["C"],
                "date": [date(2026, 1, 2)],
                "open": [450.0],
                "high": [451.0],
                "low": [449.0],
                "close": [450.0],
                "last": [450.0],
                "settle": [450.0],
                "volume": [100_000.0],
                "open_interest": [1_000.0],
            }
        )
        path = tmp_path / "corn.parquet"
        data.write_parquet(path)

        parsed = parse_quandl_chris("C", data_path=path)
        spec = MAJOR_CONTRACTS["C"]

        assert parsed["close"].item() == 4.5
        assert spec.calculate_contract_value(4.5, price_unit="dollars") == 22_500.0
        assert spec.calculate_contract_value(450.0, price_unit="cents") == 22_500.0
        assert spec.calculate_contract_value(450.0) == 22_500.0

    def test_index_points_cannot_be_converted_to_money(self):
        """Index points require their contract multiplier, not a fabricated FX conversion."""
        spec = MAJOR_CONTRACTS["ES"]

        with pytest.raises(ValueError, match="Cannot convert between dollars and index_points"):
            spec.calculate_contract_value(5000.0, price_unit="dollars")

    def test_unknown_ticker_requires_contract_spec(self, tmp_path):
        data = pl.DataFrame(
            {
                "ticker": ["KC"],
                "date": [date(2026, 1, 2)],
                "open": [300.0],
                "high": [301.0],
                "low": [299.0],
                "close": [300.0],
                "last": [300.0],
                "settle": [300.0],
                "volume": [1_000.0],
                "open_interest": [100.0],
            }
        )
        path = tmp_path / "coffee.parquet"
        data.write_parquet(path)

        with pytest.raises(ValueError, match="Contract specification required for ticker 'KC'"):
            parse_quandl_chris("KC", data_path=path)

    def test_contract_spec_ticker_must_match_request(self, chris_data_path):
        with pytest.raises(ValueError, match="does not match requested 'ES'"):
            parse_quandl_chris(
                "ES",
                data_path=chris_data_path,
                contract_spec=MAJOR_CONTRACTS["CL"],
            )

    def test_contract_spec_rejects_unsupported_price_unit(self):
        with pytest.raises(ValueError, match="Unsupported price quote unit"):
            replace(MAJOR_CONTRACTS["ES"], price_quote_unit="Cents")
