"""Integration tests for DataManager.batch_load_universe() method."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.data.data_manager import DataManager
from ml4t.data.managers.batch_manager import BatchManager
from ml4t.data.universe import Universe


@pytest.fixture
def data_manager():
    """Create DataManager instance with mock provider."""
    return DataManager(output_format="polars")


class TestBatchLoadUniverse:
    """Test DataManager.batch_load_universe() integration."""

    def test_batch_load_universe_validates_universe_name(self, data_manager):
        """batch_load_universe() raises error for invalid universe name."""
        with pytest.raises(ValueError, match="Invalid universe"):
            data_manager.batch_load_universe(
                universe="invalid_universe",
                start="2024-01-01",
                end="2024-01-31",
                provider="mock",
            )

    def test_batch_load_universe_accepts_case_insensitive_name(self, data_manager, monkeypatch):
        """batch_load_universe() accepts case-insensitive universe names."""

        # Mock Universe.get to avoid actual data fetching
        def mock_get(name):
            if name.upper().replace("_", "") == "SP500":
                return ["AAPL", "MSFT"]
            raise ValueError(f"Unknown universe {name}")

        monkeypatch.setattr(Universe, "get", mock_get)

        # Mock batch_load to avoid actual provider calls
        def mock_batch_load(self, symbols, start, end, **kwargs):
            return pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)] * len(symbols),
                    "symbol": symbols,
                    "open": [100.0] * len(symbols),
                    "high": [101.0] * len(symbols),
                    "low": [99.0] * len(symbols),
                    "close": [100.5] * len(symbols),
                    "volume": [1000.0] * len(symbols),
                }
            )

        monkeypatch.setattr(BatchManager, "batch_load", mock_batch_load)

        # These should all work (case-insensitive)
        df1 = data_manager.batch_load_universe("sp500", "2024-01-01", "2024-01-31")
        df2 = data_manager.batch_load_universe("SP500", "2024-01-01", "2024-01-31")
        df3 = data_manager.batch_load_universe("Sp500", "2024-01-01", "2024-01-31")

        # All should have same symbols
        assert df1["symbol"].unique().sort().to_list() == df2["symbol"].unique().sort().to_list()
        assert df2["symbol"].unique().sort().to_list() == df3["symbol"].unique().sort().to_list()

    def test_batch_load_universe_returns_multiasset_dataframe(self, data_manager, monkeypatch):
        """batch_load_universe() returns multi-asset DataFrame with symbol column."""
        # Mock Universe.get to return small list
        monkeypatch.setattr(Universe, "get", lambda name: ["AAPL", "MSFT", "GOOGL"])

        # Mock batch_load to return synthetic data
        def mock_batch_load(self, symbols, start, end, **kwargs):
            rows = []
            for symbol in symbols:
                for i in range(3):  # 3 rows per symbol
                    rows.append(
                        {
                            "timestamp": datetime(2024, 1, 1 + i, tzinfo=UTC),
                            "symbol": symbol,
                            "open": 100.0 + i,
                            "high": 101.0 + i,
                            "low": 99.0 + i,
                            "close": 100.5 + i,
                            "volume": 1000.0 * (i + 1),
                        }
                    )
            return pl.DataFrame(rows)

        monkeypatch.setattr(BatchManager, "batch_load", mock_batch_load)

        df = data_manager.batch_load_universe("sp500", "2024-01-01", "2024-01-03")

        # Should have symbol column
        assert "symbol" in df.columns
        assert "timestamp" in df.columns

        # Should have data for all symbols
        symbols = df["symbol"].unique().sort().to_list()
        assert symbols == ["AAPL", "GOOGL", "MSFT"]

        # Should have 3 rows per symbol = 9 total
        assert len(df) == 9

    def test_batch_load_universe_passes_parameters_to_batch_load(self, data_manager, monkeypatch):
        """batch_load_universe() passes parameters through to batch_load()."""
        # Mock Universe.get
        monkeypatch.setattr(Universe, "get", lambda name: ["AAPL"])

        # Track what parameters batch_load receives
        received_params = {}

        def mock_batch_load(
            self, symbols, start, end, frequency, provider, max_workers, fail_on_partial, **kwargs
        ):
            received_params["symbols"] = symbols
            received_params["start"] = start
            received_params["end"] = end
            received_params["frequency"] = frequency
            received_params["provider"] = provider
            received_params["max_workers"] = max_workers
            received_params["fail_on_partial"] = fail_on_partial
            received_params["extra"] = kwargs

            return pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                    "symbol": ["AAPL"],
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000.0],
                }
            )

        monkeypatch.setattr(BatchManager, "batch_load", mock_batch_load)

        # Call with specific parameters
        data_manager.batch_load_universe(
            universe="sp500",
            start="2024-01-01",
            end="2024-01-31",
            frequency="hourly",
            provider="yahoo",
            max_workers=8,
            fail_on_partial=True,
            custom_param="test_value",
        )

        # Verify parameters were passed through
        assert received_params["start"] == "2024-01-01"
        assert received_params["end"] == "2024-01-31"
        assert received_params["frequency"] == "hourly"
        assert received_params["provider"] == "yahoo"
        assert received_params["max_workers"] == 8
        assert received_params["fail_on_partial"] is True
        assert received_params["extra"]["custom_param"] == "test_value"

    def test_batch_load_universe_with_mock_provider(self, data_manager):
        """batch_load_universe() works with mock provider (integration test)."""
        # Use a very small custom universe for testing
        Universe.add_custom("test_tiny", ["AAPL", "MSFT"])

        try:
            df = data_manager.batch_load_universe(
                universe="test_tiny",
                start="2024-01-01",
                end="2024-01-03",
                provider="mock",
                fail_on_partial=False,  # Mock provider might fail
            )

            # Should return multi-asset DataFrame
            assert isinstance(df, pl.DataFrame)
            assert "symbol" in df.columns
            assert "timestamp" in df.columns
            assert len(df) > 0

        finally:
            # Cleanup
            Universe.remove_custom("test_tiny")

    def test_batch_load_universe_error_propagation(self, data_manager, monkeypatch):
        """batch_load_universe() propagates errors from batch_load()."""
        # Mock Universe.get
        monkeypatch.setattr(Universe, "get", lambda name: ["AAPL"])

        # Mock batch_load to raise error
        def mock_batch_load(self, symbols, start, end, **kwargs):
            raise ValueError("Mock error from batch_load")

        monkeypatch.setattr(BatchManager, "batch_load", mock_batch_load)

        with pytest.raises(ValueError, match="Mock error from batch_load"):
            data_manager.batch_load_universe("sp500", "2024-01-01", "2024-01-31")

    def test_batch_load_universe_with_different_universes(self, data_manager, monkeypatch):
        """batch_load_universe() works with different universe types."""
        # Mock Universe.get to return different lists
        universe_symbols = {
            "SP500": ["AAPL", "MSFT"],
            "NASDAQ100": ["GOOGL", "AMZN"],
            "CRYPTO_TOP_100": ["BTC", "ETH"],
            "FOREX_MAJORS": ["EURUSD", "GBPUSD"],
        }

        def mock_get(name):
            normalized = name.upper().replace("-", "_").replace(" ", "_")
            # Handle fuzzy matching
            for key in universe_symbols:
                if key.replace("_", "") == normalized.replace("_", ""):
                    return universe_symbols[key]
            raise ValueError(f"Unknown universe {name}")

        monkeypatch.setattr(Universe, "get", mock_get)

        # Mock batch_load
        def mock_batch_load(self, symbols, start, end, **kwargs):
            return pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)] * len(symbols),
                    "symbol": symbols,
                    "open": [100.0] * len(symbols),
                    "high": [101.0] * len(symbols),
                    "low": [99.0] * len(symbols),
                    "close": [100.5] * len(symbols),
                    "volume": [1000.0] * len(symbols),
                }
            )

        monkeypatch.setattr(BatchManager, "batch_load", mock_batch_load)

        # Test each universe type
        df_sp500 = data_manager.batch_load_universe("sp500", "2024-01-01", "2024-01-31")
        assert set(df_sp500["symbol"].to_list()) == {"AAPL", "MSFT"}

        df_nasdaq = data_manager.batch_load_universe("nasdaq100", "2024-01-01", "2024-01-31")
        assert set(df_nasdaq["symbol"].to_list()) == {"GOOGL", "AMZN"}

        df_crypto = data_manager.batch_load_universe("crypto_top_100", "2024-01-01", "2024-01-31")
        assert set(df_crypto["symbol"].to_list()) == {"BTC", "ETH"}

        df_forex = data_manager.batch_load_universe("forex_majors", "2024-01-01", "2024-01-31")
        assert set(df_forex["symbol"].to_list()) == {"EURUSD", "GBPUSD"}


class TestBatchLoadUniverseEdgeCases:
    """Test edge cases for batch_load_universe()."""

    def test_batch_load_universe_empty_universe(self, data_manager, monkeypatch):
        """batch_load_universe() handles empty universe gracefully."""
        # Mock Universe.get to return empty list
        monkeypatch.setattr(Universe, "get", lambda name: [])

        # Mock batch_load to handle empty list
        def mock_batch_load(self, symbols, start, end, **kwargs):
            if not symbols:
                raise ValueError("symbols list cannot be empty")
            return pl.DataFrame()

        monkeypatch.setattr(BatchManager, "batch_load", mock_batch_load)

        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            data_manager.batch_load_universe("empty", "2024-01-01", "2024-01-31")

    def test_batch_load_universe_date_validation(self, data_manager, monkeypatch):
        """batch_load_universe() validates dates (delegated to batch_load)."""
        # Mock Universe.get
        monkeypatch.setattr(Universe, "get", lambda name: ["AAPL"])

        # Don't mock batch_load - let it validate dates

        # Invalid date format should be caught by batch_load -> _validate_dates
        with pytest.raises(ValueError, match="Invalid date format"):
            data_manager.batch_load_universe("sp500", "invalid-date", "2024-01-31")

        # End before start should be caught
        with pytest.raises(ValueError, match="End date must be after start date"):
            data_manager.batch_load_universe("sp500", "2024-12-31", "2024-01-01")
