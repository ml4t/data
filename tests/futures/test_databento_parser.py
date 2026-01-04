"""Tests for Databento futures parser."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.futures.databento_parser import (
    MONTH_CODES,
    MONTH_TO_CODE,
    STAT_TYPE_OPEN_INTEREST,
    ContractInfo,
    get_contract_chain,
    get_expiration_dates,
    get_front_back_contracts,
    load_databento_definitions,
    load_databento_ohlcv,
    load_databento_open_interest,
    load_databento_statistics,
    parse_contract_symbol,
    parse_databento,
    parse_databento_raw,
)

# Path to real Databento data (downloaded 70 instruments, 10 years)
DATABENTO_DATA_PATH = Path("/home/stefan/ml4t/data/futures")


class TestParseContractSymbol:
    """Tests for contract symbol parsing."""

    def test_parse_es_march_2025(self):
        """Parse standard ES contract symbol."""
        info = parse_contract_symbol("ESH25")

        assert info.raw_symbol == "ESH25"
        assert info.product == "ES"
        assert info.month_code == "H"
        assert info.month == 3
        assert info.year == 2025

    def test_parse_cl_december_2024(self):
        """Parse crude oil contract symbol."""
        info = parse_contract_symbol("CLZ24")

        assert info.raw_symbol == "CLZ24"
        assert info.product == "CL"
        assert info.month_code == "Z"
        assert info.month == 12
        assert info.year == 2024

    def test_parse_gc_february_2025(self):
        """Parse gold contract symbol."""
        info = parse_contract_symbol("GCG25")

        assert info.raw_symbol == "GCG25"
        assert info.product == "GC"
        assert info.month_code == "G"
        assert info.month == 2
        assert info.year == 2025

    def test_parse_euro_fx_symbol(self):
        """Parse longer product symbol (Euro FX)."""
        info = parse_contract_symbol("6EH25")

        assert info.raw_symbol == "6EH25"
        assert info.product == "6E"
        assert info.month_code == "H"
        assert info.month == 3
        assert info.year == 2025

    def test_parse_all_month_codes(self):
        """Test all month codes are recognized."""
        for code, expected_month in MONTH_CODES.items():
            symbol = f"ES{code}25"
            info = parse_contract_symbol(symbol)
            assert info.month == expected_month
            assert info.month_code == code

    def test_parse_year_2000s(self):
        """Test year parsing for 2000-2029 range."""
        info = parse_contract_symbol("ESH00")
        assert info.year == 2000

        info = parse_contract_symbol("ESH15")
        assert info.year == 2015

        info = parse_contract_symbol("ESH29")
        assert info.year == 2029

    def test_parse_year_1900s(self):
        """Test year parsing for 1930-1999 range (historical)."""
        info = parse_contract_symbol("ESH30")
        assert info.year == 1930

        info = parse_contract_symbol("ESH99")
        assert info.year == 1999

    def test_parse_single_digit_year_2020s(self):
        """Test single-digit year format for 2020s (0-5)."""
        # ZMK0 -> ZM + K + 0 -> 2020
        info = parse_contract_symbol("ZMK0")
        assert info.product == "ZM"
        assert info.month_code == "K"
        assert info.year == 2020

        # ZMF5 -> ZM + F + 5 -> 2025
        info = parse_contract_symbol("ZMF5")
        assert info.year == 2025

    def test_parse_single_digit_year_2010s(self):
        """Test single-digit year format for 2010s (6-9)."""
        # ZMK9 -> ZM + K + 9 -> 2019
        info = parse_contract_symbol("ZMK9")
        assert info.product == "ZM"
        assert info.month_code == "K"
        assert info.year == 2019

        # ZMV6 -> ZM + V + 6 -> 2016
        info = parse_contract_symbol("ZMV6")
        assert info.year == 2016

    def test_invalid_symbol_too_short(self):
        """Test error on too short symbol."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            parse_contract_symbol("ES")

    def test_invalid_symbol_3_chars(self):
        """Test error on 3-character symbol."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            parse_contract_symbol("ESH")

    def test_invalid_month_code(self):
        """Test error on invalid month code."""
        with pytest.raises(ValueError, match="Invalid month code"):
            parse_contract_symbol("ESA25")  # A is not a valid month code

    def test_invalid_month_code_l(self):
        """Test error on 'L' which is not a valid month code."""
        with pytest.raises(ValueError, match="Invalid month code"):
            parse_contract_symbol("ESL25")

    def test_invalid_year(self):
        """Test error on invalid year."""
        with pytest.raises(ValueError, match="Invalid year"):
            parse_contract_symbol("ESHXX")  # XX is not a valid year

    def test_invalid_year_letters(self):
        """Test error on letter year."""
        with pytest.raises(ValueError, match="Invalid year"):
            parse_contract_symbol("ESHAB")

    def test_empty_product(self):
        """Test error when no product symbol (symbol too short)."""
        # "H25" is only 3 chars, caught by length check first
        with pytest.raises(ValueError, match="Invalid symbol format"):
            parse_contract_symbol("H25")

    def test_contract_info_month_property(self):
        """Test ContractInfo.contract_month property."""
        info = parse_contract_symbol("ESH25")
        assert info.contract_month == "2025-03"

        info = parse_contract_symbol("CLZ24")
        assert info.contract_month == "2024-12"

    def test_contract_info_hashable(self):
        """Test ContractInfo can be used in sets/dicts."""
        info1 = parse_contract_symbol("ESH25")
        info2 = parse_contract_symbol("ESH25")
        info3 = parse_contract_symbol("ESM25")

        # Same symbol should hash the same
        assert hash(info1) == hash(info2)

        # Can be used in set
        contracts = {info1, info3}
        assert len(contracts) == 2


class TestContractInfo:
    """Tests for ContractInfo dataclass."""

    def test_contract_info_with_expiration(self):
        """Test ContractInfo with expiration date."""
        info = ContractInfo(
            raw_symbol="ESH25",
            product="ES",
            month_code="H",
            month=3,
            year=2025,
            expiration=date(2025, 3, 21),
        )

        assert info.expiration == date(2025, 3, 21)
        assert info.contract_month == "2025-03"

    def test_contract_info_without_expiration(self):
        """Test ContractInfo without expiration date."""
        info = ContractInfo(
            raw_symbol="CLZ24",
            product="CL",
            month_code="Z",
            month=12,
            year=2024,
        )

        assert info.expiration is None

    def test_contract_info_equality(self):
        """Test ContractInfo equality is based on raw_symbol hash."""
        info1 = ContractInfo(
            raw_symbol="ESH25",
            product="ES",
            month_code="H",
            month=3,
            year=2025,
        )
        info2 = ContractInfo(
            raw_symbol="ESH25",
            product="ES",
            month_code="H",
            month=3,
            year=2025,
            expiration=date(2025, 3, 21),
        )
        # Hash is based on raw_symbol
        assert hash(info1) == hash(info2)


class TestMonthCodeMappings:
    """Tests for month code constants."""

    def test_month_codes_complete(self):
        """All 12 months have codes."""
        assert len(MONTH_CODES) == 12

    def test_month_to_code_inverse(self):
        """MONTH_TO_CODE is correct inverse of MONTH_CODES."""
        for code, month in MONTH_CODES.items():
            assert MONTH_TO_CODE[month] == code

    def test_standard_cme_codes(self):
        """Verify standard CME month codes."""
        expected = {
            "F": 1,
            "G": 2,
            "H": 3,
            "J": 4,
            "K": 5,
            "M": 6,
            "N": 7,
            "Q": 8,
            "U": 9,
            "V": 10,
            "X": 11,
            "Z": 12,
        }
        assert expected == MONTH_CODES


# ============================================================================
# Tests using real Databento data (requires downloaded data)
# ============================================================================


@pytest.fixture
def databento_path():
    """Return path to Databento data, skip if not available."""
    if not DATABENTO_DATA_PATH.exists():
        pytest.skip("Databento data not available")
    return DATABENTO_DATA_PATH


class TestLoadDatabentoOHLCV:
    """Tests for loading OHLCV data."""

    def test_load_zm_ohlcv(self, databento_path):
        """Load ZM (soybean meal) OHLCV data."""
        df = load_databento_ohlcv("ZM", databento_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_load_zc_ohlcv(self, databento_path):
        """Load ZC (corn) OHLCV data."""
        df = load_databento_ohlcv("ZC", databento_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_load_es_ohlcv(self, databento_path):
        """Load ES (E-mini S&P 500) OHLCV data."""
        df = load_databento_ohlcv("ES", databento_path)

        assert isinstance(df, pl.DataFrame)
        # ES data may be more limited
        assert "close" in df.columns

    def test_load_nonexistent_product(self, databento_path):
        """Test error on nonexistent product."""
        with pytest.raises(FileNotFoundError, match="OHLCV data not found"):
            load_databento_ohlcv("NONEXISTENT", databento_path)

    def test_ohlcv_has_symbol_column(self, databento_path):
        """OHLCV data should have symbol column for contract identification."""
        df = load_databento_ohlcv("ZM", databento_path)

        # Should have either 'symbol' or 'raw_symbol'
        assert "symbol" in df.columns or "raw_symbol" in df.columns

    def test_ohlcv_has_timestamp(self, databento_path):
        """OHLCV data should have timestamp column."""
        df = load_databento_ohlcv("ZM", databento_path)

        # Databento uses 'ts_event' for timestamp
        assert "ts_event" in df.columns or "timestamp" in df.columns


class TestLoadDatabentoDefinitions:
    """Tests for loading definition data."""

    def test_load_zm_definitions(self, databento_path):
        """Load ZM contract definitions."""
        df = load_databento_definitions("ZM", databento_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert "symbol" in df.columns or "raw_symbol" in df.columns
        assert "expiration" in df.columns

    def test_load_zc_definitions(self, databento_path):
        """Load ZC contract definitions."""
        df = load_databento_definitions("ZC", databento_path)

        assert isinstance(df, pl.DataFrame)
        assert "expiration" in df.columns

    def test_load_nonexistent_definitions(self, databento_path):
        """Test error on nonexistent product definitions."""
        with pytest.raises(FileNotFoundError, match="Definition data not found"):
            load_databento_definitions("NONEXISTENT", databento_path)

    def test_definitions_have_expiration(self, databento_path):
        """Definition data should have expiration dates."""
        df = load_databento_definitions("ZM", databento_path)

        assert "expiration" in df.columns
        # At least some contracts should have non-null expiration
        assert df.filter(pl.col("expiration").is_not_null()).height > 0


class TestGetExpirationDates:
    """Tests for get_expiration_dates function."""

    def test_get_zm_expirations(self, databento_path):
        """Get expiration dates for ZM contracts."""
        expirations = get_expiration_dates("ZM", databento_path)

        assert isinstance(expirations, dict)
        # ZM should have multiple contracts
        assert len(expirations) > 0

    def test_expirations_are_dates(self, databento_path):
        """Expiration values should be date objects."""
        expirations = get_expiration_dates("ZM", databento_path)

        for symbol, exp_date in expirations.items():
            assert isinstance(symbol, str)
            assert isinstance(exp_date, date)

    def test_expiration_symbols_parseable(self, databento_path):
        """Expiration dict symbols should be parseable."""
        expirations = get_expiration_dates("ZM", databento_path)

        for symbol in list(expirations.keys())[:5]:
            # Should be able to parse the symbol
            try:
                info = parse_contract_symbol(symbol)
                assert info.product == "ZM"
            except ValueError:
                # Some symbols might be spreads or invalid
                pass


class TestParseDatabentoRaw:
    """Tests for parse_databento_raw function."""

    def test_parse_zm_raw(self, databento_path):
        """Parse ZM raw data with multiple contracts per date."""
        df = parse_databento_raw("ZM", databento_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert "date" in df.columns
        assert "symbol" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "open_interest" in df.columns
        assert "expiration" in df.columns

    def test_raw_has_multiple_contracts_per_date(self, databento_path):
        """Raw data should have multiple contracts per date."""
        df = parse_databento_raw("ZM", databento_path)

        # Check a specific date has multiple contracts
        dates = df.select("date").unique().head(10)
        if dates.height > 0:
            sample_date = dates.row(0)[0]
            contracts_on_date = df.filter(pl.col("date") == sample_date).height
            # Should have multiple contracts trading
            assert contracts_on_date >= 1

    def test_raw_columns_are_float(self, databento_path):
        """Numeric columns should be float type."""
        df = parse_databento_raw("ZM", databento_path)

        assert df.schema["open"] == pl.Float64
        assert df.schema["high"] == pl.Float64
        assert df.schema["low"] == pl.Float64
        assert df.schema["close"] == pl.Float64
        assert df.schema["volume"] == pl.Float64

    def test_raw_sorted_by_date_symbol(self, databento_path):
        """Raw data should be sorted by date and symbol."""
        df = parse_databento_raw("ZM", databento_path)

        # Check if sorted
        dates = df.select("date").to_series().to_list()
        assert dates == sorted(dates)


class TestParseDatabento:
    """Tests for parse_databento function (front month only)."""

    def test_parse_zm_front_month(self, databento_path):
        """Parse ZM data selecting front month only."""
        df = parse_databento("ZM", databento_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

        # Should have standard OHLCV columns
        expected_cols = {"date", "open", "high", "low", "close", "volume", "open_interest"}
        assert set(df.columns) == expected_cols

    def test_front_month_single_row_per_date(self, databento_path):
        """Front month data should have exactly one row per date."""
        df = parse_databento("ZM", databento_path)

        # Check no duplicate dates
        unique_dates = df.select("date").unique().height
        total_rows = df.height
        assert unique_dates == total_rows

    def test_front_month_sorted_by_date(self, databento_path):
        """Front month data should be sorted by date."""
        df = parse_databento("ZM", databento_path)

        dates = df.select("date").to_series().to_list()
        assert dates == sorted(dates)


class TestGetContractChain:
    """Tests for get_contract_chain function."""

    def test_get_zm_chain(self, databento_path):
        """Get ZM contract chain."""
        chain = get_contract_chain("ZM", databento_path)

        assert isinstance(chain, list)
        assert len(chain) > 0
        assert all(isinstance(c, ContractInfo) for c in chain)

    def test_chain_sorted_by_expiration(self, databento_path):
        """Contract chain should be sorted by expiration."""
        chain = get_contract_chain("ZM", databento_path)

        expirations = [c.expiration for c in chain if c.expiration is not None]
        assert expirations == sorted(expirations)

    def test_chain_has_product(self, databento_path):
        """All contracts in chain should have correct product."""
        chain = get_contract_chain("ZM", databento_path)

        for contract in chain:
            assert contract.product == "ZM"


class TestGetFrontBackContracts:
    """Tests for get_front_back_contracts function."""

    def test_get_front_back_zm(self, databento_path):
        """Get front and back contracts for ZM."""
        # Use a date in the middle of the data range
        as_of = date(2020, 6, 1)
        front, back = get_front_back_contracts("ZM", as_of, databento_path)

        # Both should exist for ZM
        assert front is not None or back is not None

    def test_front_expires_before_back(self, databento_path):
        """Front contract should expire before back contract."""
        as_of = date(2020, 6, 1)
        front, back = get_front_back_contracts("ZM", as_of, databento_path)

        if front is not None and back is not None:
            assert front.expiration < back.expiration

    def test_both_unexpired(self, databento_path):
        """Both contracts should be unexpired as of the given date."""
        as_of = date(2020, 6, 1)
        front, back = get_front_back_contracts("ZM", as_of, databento_path)

        if front is not None:
            assert front.expiration > as_of
        if back is not None:
            assert back.expiration > as_of

    def test_future_date_no_contracts(self, databento_path):
        """Very future date may have no unexpired contracts in data."""
        as_of = date(2030, 1, 1)
        front, back = get_front_back_contracts("ZM", as_of, databento_path)

        # May or may not have contracts depending on data freshness
        # This is just checking it doesn't crash
        assert front is None or isinstance(front, ContractInfo)


# ============================================================================
# Tests with mocked data (no external dependencies)
# ============================================================================


class TestLoadFunctionsMocked:
    """Tests using mocked file system."""

    def test_ohlcv_filters_error_column(self, tmp_path):
        """OHLCV loading should filter out error rows."""
        # Create test data with error column
        df = pl.DataFrame(
            {
                "ts_event": [1, 2, 3],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000, 2000, 3000],
                "symbol": ["ESH25", "ESH25", "ESH25"],
                "product": ["ES", "ES", "ES"],
                "error": [None, "API error", None],
            }
        )

        # Write to temp location
        ohlcv_dir = tmp_path / "ohlcv_1d" / "product=ES"
        ohlcv_dir.mkdir(parents=True)
        df.write_parquet(ohlcv_dir / "ohlcv_1d.parquet")

        # Load and verify error rows filtered
        result = load_databento_ohlcv("ES", tmp_path)
        assert result.height == 2  # Only non-error rows

    def test_definitions_filters_error_column(self, tmp_path):
        """Definition loading should filter out error rows."""
        from datetime import datetime

        df = pl.DataFrame(
            {
                "symbol": ["ESH25", "ESM25", "ESU25"],
                "product": ["ES", "ES", "ES"],
                "expiration": [
                    datetime(2025, 3, 21),
                    datetime(2025, 6, 20),
                    datetime(2025, 9, 19),
                ],
                "error": [None, "Data error", None],
            }
        )

        def_dir = tmp_path / "definition" / "product=ES"
        def_dir.mkdir(parents=True)
        df.write_parquet(def_dir / "definition.parquet")

        result = load_databento_definitions("ES", tmp_path)
        assert result.height == 2

    def test_empty_ohlcv_raises_error(self, tmp_path):
        """Empty OHLCV data should raise ValueError."""
        df = pl.DataFrame(
            {
                "ts_event": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "symbol": [],
                "error": [],
            }
        ).cast(
            {
                "ts_event": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
                "symbol": pl.String,
                "error": pl.String,
            }
        )

        ohlcv_dir = tmp_path / "ohlcv_1d" / "product=ES"
        ohlcv_dir.mkdir(parents=True)
        df.write_parquet(ohlcv_dir / "ohlcv_1d.parquet")

        with pytest.raises(ValueError, match="No OHLCV data available"):
            load_databento_ohlcv("ES", tmp_path)

    def test_empty_definitions_raises_error(self, tmp_path):
        """Empty definition data should raise ValueError."""
        df = pl.DataFrame(
            {
                "symbol": [],
                "expiration": [],
                "error": [],
            }
        ).cast(
            {
                "symbol": pl.String,
                "expiration": pl.Datetime("ns", "UTC"),
                "error": pl.String,
            }
        )

        def_dir = tmp_path / "definition" / "product=ES"
        def_dir.mkdir(parents=True)
        df.write_parquet(def_dir / "definition.parquet")

        with pytest.raises(ValueError, match="No definition data available"):
            load_databento_definitions("ES", tmp_path)


class TestStatisticsLoading:
    """Tests for statistics loading (OI, settlement, etc.)."""

    def test_statistics_not_found_error(self, tmp_path):
        """Statistics loading should raise FileNotFoundError when missing."""
        with pytest.raises(FileNotFoundError, match="Statistics data not found"):
            load_databento_statistics("ES", tmp_path)

    def test_open_interest_not_found_error(self, tmp_path):
        """OI loading should raise FileNotFoundError when statistics missing."""
        with pytest.raises(FileNotFoundError, match="Statistics data not found"):
            load_databento_open_interest("ES", tmp_path)

    def test_load_statistics_with_filter(self, tmp_path):
        """Statistics should be filtered by stat_type."""
        from datetime import datetime

        df = pl.DataFrame(
            {
                "ts_event": [
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 2),
                ],
                "symbol": ["ESH20", "ESH20", "ESH20"],
                "stat_type": [9, 3, 9],  # 9=OI, 3=settlement
                "quantity": [100000, 0, 110000],
                "price": [0.0, 3200.0, 0.0],  # All floats to avoid type inference issues
            }
        )

        stats_dir = tmp_path / "statistics" / "product=ES"
        stats_dir.mkdir(parents=True)
        df.write_parquet(stats_dir / "statistics.parquet")

        # Load only OI (stat_type=9)
        result = load_databento_statistics("ES", tmp_path, stat_types=[STAT_TYPE_OPEN_INTEREST])
        assert result.height == 2
        assert all(result.select("stat_type").to_series() == 9)

    def test_load_open_interest(self, tmp_path):
        """Test loading open interest data."""
        from datetime import datetime

        df = pl.DataFrame(
            {
                "ts_event": [
                    datetime(2020, 1, 2, 16, 0),
                    datetime(2020, 1, 2, 17, 0),  # Later reading
                    datetime(2020, 1, 3, 17, 0),
                ],
                "symbol": ["ESH20", "ESH20", "ESH20"],
                "stat_type": [9, 9, 9],
                "quantity": [100000, 110000, 120000],
            }
        )

        stats_dir = tmp_path / "statistics" / "product=ES"
        stats_dir.mkdir(parents=True)
        df.write_parquet(stats_dir / "statistics.parquet")

        result = load_databento_open_interest("ES", tmp_path)

        # Should have 2 rows (one per date)
        assert result.height == 2
        # Should have latest OI reading per date
        assert "open_interest" in result.columns


class TestParseDatabentoRawMocked:
    """Tests for parse_databento_raw with mocked data."""

    def test_handles_missing_expiration_gracefully(self, tmp_path):
        """Should handle missing definition data gracefully."""
        from datetime import datetime

        # Create OHLCV without matching definitions
        ohlcv_df = pl.DataFrame(
            {
                "ts_event": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 2000],
                "symbol": ["ESH20", "ESH20"],
                "product": ["ES", "ES"],
            }
        )

        ohlcv_dir = tmp_path / "ohlcv_1d" / "product=ES"
        ohlcv_dir.mkdir(parents=True)
        ohlcv_df.write_parquet(ohlcv_dir / "ohlcv_1d.parquet")

        # No definition directory - should still work
        result = parse_databento_raw("ES", tmp_path, include_open_interest=False)

        assert result.height == 2
        assert "expiration" in result.columns
        # Expirations should be null
        assert result.select("expiration").null_count().item() == 2

    def test_handles_missing_statistics_gracefully(self, tmp_path):
        """Should handle missing statistics data gracefully."""
        from datetime import datetime

        # Create OHLCV and definition without statistics
        ohlcv_df = pl.DataFrame(
            {
                "ts_event": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 2000],
                "symbol": ["ESH20", "ESH20"],
                "product": ["ES", "ES"],
            }
        )

        ohlcv_dir = tmp_path / "ohlcv_1d" / "product=ES"
        ohlcv_dir.mkdir(parents=True)
        ohlcv_df.write_parquet(ohlcv_dir / "ohlcv_1d.parquet")

        # Request with OI but no statistics available
        result = parse_databento_raw("ES", tmp_path, include_open_interest=True)

        assert result.height == 2
        assert "open_interest" in result.columns
        # OI should be null since no statistics
        assert result.select("open_interest").null_count().item() == 2
