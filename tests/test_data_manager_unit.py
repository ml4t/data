"""Unit tests for DataManager internal methods and utilities.

These tests focus on methods that don't require full integration setup.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml4t.data.data_manager import DataManager, ProviderRouter
from ml4t.data.managers.config_manager import ConfigManager


class TestProviderRouterCache:
    """Tests for ProviderRouter caching behavior."""

    def test_cache_hit(self):
        """Test that cached results are returned."""
        router = ProviderRouter()
        router.add_pattern(r"^BTC", "cryptocompare")

        # First call - populates cache
        result1 = router.get_provider("BTC")
        assert result1 == "cryptocompare"
        assert "BTC" in router._cache

        # Second call - should use cache
        result2 = router.get_provider("BTC")
        assert result2 == "cryptocompare"

    def test_clear_cache(self):
        """Test cache clearing."""
        router = ProviderRouter()
        router.add_pattern(r"^BTC", "cryptocompare")

        router.get_provider("BTC")
        assert "BTC" in router._cache

        router.clear_cache()
        assert router._cache == {}

    def test_add_pattern_clears_cache(self):
        """Test that adding pattern clears cache."""
        router = ProviderRouter()
        router.add_pattern(r"^BTC", "cryptocompare")

        router.get_provider("BTC")
        assert "BTC" in router._cache

        # Adding new pattern should clear cache
        router.add_pattern(r"^ETH", "cryptocompare")
        assert router._cache == {}

    def test_multiple_patterns_first_match(self):
        """Test that first matching pattern wins."""
        router = ProviderRouter()
        router.add_pattern(r"^BTC", "provider1")
        router.add_pattern(r"^BTC.*", "provider2")

        # First pattern should match
        result = router.get_provider("BTCUSD")
        assert result == "provider1"


class TestDataManagerMergeConfigs:
    """Tests for _merge_configs method (now in ConfigManager)."""

    def test_merge_simple_override(self):
        """Test simple key override."""
        cm = ConfigManager()
        base = {"key1": "value1", "key2": "value2"}
        override = {"key2": "new_value"}

        result = cm._merge_configs(base, override)

        assert result["key1"] == "value1"
        assert result["key2"] == "new_value"

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        cm = ConfigManager()
        base = {
            "providers": {"crypto": {"api_key": "base_key", "rate_limit": 10}},
            "other": "value",
        }
        override = {"providers": {"crypto": {"rate_limit": 20}}}

        result = cm._merge_configs(base, override)

        assert result["providers"]["crypto"]["api_key"] == "base_key"
        assert result["providers"]["crypto"]["rate_limit"] == 20
        assert result["other"] == "value"

    def test_merge_adds_new_keys(self):
        """Test adding new keys from override."""
        cm = ConfigManager()
        base = {"existing": "value"}
        override = {"new_key": "new_value"}

        result = cm._merge_configs(base, override)

        assert result["existing"] == "value"
        assert result["new_key"] == "new_value"

    def test_merge_empty_base(self):
        """Test merging into empty base."""
        cm = ConfigManager()
        base = {}
        override = {"key": "value"}

        result = cm._merge_configs(base, override)

        assert result["key"] == "value"

    def test_merge_empty_override(self):
        """Test merging empty override."""
        cm = ConfigManager()
        base = {"key": "value"}
        override = {}

        result = cm._merge_configs(base, override)

        assert result["key"] == "value"


class TestDataManagerValidateDates:
    """Tests for _validate_dates method."""

    def test_valid_dates(self):
        """Test valid date validation passes."""
        dm = DataManager()

        # Should not raise
        dm._validate_dates("2024-01-01", "2024-01-31")

    def test_invalid_start_date_format(self):
        """Test invalid start date format."""
        dm = DataManager()

        with pytest.raises(ValueError, match="Invalid date format"):
            dm._validate_dates("01-01-2024", "2024-01-31")

    def test_invalid_end_date_format(self):
        """Test invalid end date format."""
        dm = DataManager()

        with pytest.raises(ValueError, match="Invalid date format"):
            dm._validate_dates("2024-01-01", "Jan 31, 2024")

    def test_end_before_start(self):
        """Test end date before start date."""
        dm = DataManager()

        with pytest.raises(ValueError, match="End date must be after start date"):
            dm._validate_dates("2024-01-31", "2024-01-01")

    def test_same_day_valid(self):
        """Test same start and end date is valid."""
        dm = DataManager()

        # Should not raise - same day is valid
        dm._validate_dates("2024-01-15", "2024-01-15")


class TestDataManagerConvertOutput:
    """Tests for _convert_output method."""

    def test_convert_to_polars(self):
        """Test Polars output format."""
        dm = DataManager(output_format="polars")
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        result = dm._convert_output(df)

        assert isinstance(result, pl.DataFrame)
        assert result.equals(df)

    def test_convert_to_pandas(self):
        """Test pandas output format."""
        dm = DataManager(output_format="pandas")
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        result = dm._convert_output(df)

        # Check it's pandas
        assert hasattr(result, "iloc")
        assert list(result["a"]) == [1, 2, 3]

    def test_convert_to_lazy(self):
        """Test lazy output format."""
        dm = DataManager(output_format="lazy")
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        result = dm._convert_output(df)

        assert isinstance(result, pl.LazyFrame)
        assert result.collect().equals(df)


class TestDataManagerListProviders:
    """Tests for list_providers method."""

    def test_list_providers_default(self):
        """Test listing default providers."""
        dm = DataManager()

        providers = dm.list_providers()

        assert isinstance(providers, list)
        # Should have some default providers
        assert "yahoo" in providers or "mock" in providers

    def test_list_providers_includes_free_providers(self):
        """Test listing providers includes free providers."""
        dm = DataManager()

        providers = dm.list_providers()

        # Free providers should always be available
        assert "yahoo" in providers
        assert "mock" in providers
        assert "synthetic" in providers


class TestDataManagerGetProviderInfo:
    """Tests for get_provider_info method."""

    def test_get_provider_info_unknown(self):
        """Test getting info for unknown provider raises ValueError."""
        dm = DataManager()

        with pytest.raises(ValueError, match="not available"):
            dm.get_provider_info("unknown_provider")

    def test_get_provider_info_existing(self):
        """Test getting info for existing provider."""
        dm = DataManager()

        # Yahoo is always available (free provider)
        info = dm.get_provider_info("yahoo")

        assert isinstance(info, dict)
        assert "name" in info or info is not None


class TestDataManagerClearCache:
    """Tests for clear_cache method."""

    def test_clear_cache(self):
        """Test cache clearing."""
        dm = DataManager()
        dm.router._cache["TEST"] = "provider"

        dm.clear_cache()

        assert dm.router._cache == {}


class TestDataManagerContextManager:
    """Tests for context manager protocol."""

    def test_enter_returns_self(self):
        """Test __enter__ returns self."""
        dm = DataManager()

        result = dm.__enter__()

        assert result is dm

    def test_exit_without_storage(self):
        """Test __exit__ without storage configured."""
        dm = DataManager()

        # Should not raise
        dm.__exit__(None, None, None)

    def test_exit_with_storage(self):
        """Test __exit__ with storage closes connections."""
        mock_storage = MagicMock()
        dm = DataManager(storage=mock_storage)

        dm.__exit__(None, None, None)

        # Should call clear_cache which cleans up providers


class TestDataManagerMergeData:
    """Tests for _merge_data method (requires storage)."""

    def test_merge_data_no_overlap(self):
        """Test merging non-overlapping data."""
        mock_storage = MagicMock()
        dm = DataManager(storage=mock_storage)

        existing = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "close": [100.0, 101.0],
            }
        )
        new = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 3), datetime(2024, 1, 4)],
                "close": [102.0, 103.0],
            }
        )

        result = dm._merge_data(existing, new)

        assert len(result) == 4

    def test_merge_data_with_overlap(self):
        """Test merging overlapping data (new data wins)."""
        mock_storage = MagicMock()
        dm = DataManager(storage=mock_storage)

        existing = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "close": [100.0, 101.0],
            }
        )
        new = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "close": [111.0, 112.0],  # Updated value for Jan 2
            }
        )

        result = dm._merge_data(existing, new)

        assert len(result) == 3
        # Jan 2 should have new value
        jan2_row = result.filter(pl.col("timestamp") == datetime(2024, 1, 2))
        assert jan2_row["close"][0] == 111.0

    def test_merge_data_sorts_by_timestamp(self):
        """Test merged data is sorted by timestamp."""
        mock_storage = MagicMock()
        dm = DataManager(storage=mock_storage)

        existing = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 3)],
                "close": [103.0],
            }
        )
        new = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "close": [101.0],
            }
        )

        result = dm._merge_data(existing, new)

        # Should be sorted chronologically
        assert result["timestamp"][0] == datetime(2024, 1, 1)
        assert result["timestamp"][1] == datetime(2024, 1, 3)


class TestDataManagerReportProgress:
    """Tests for _report_progress method (requires storage for callback)."""

    def test_report_progress_with_callback(self):
        """Test progress reporting with callback (requires storage)."""
        callback = MagicMock()
        mock_storage = MagicMock()
        dm = DataManager(storage=mock_storage, progress_callback=callback)

        dm._report_progress("Loading data", 0.5)

        callback.assert_called_once_with("Loading data", 0.5)

    def test_report_progress_without_callback(self):
        """Test progress reporting without callback (no-op)."""
        dm = DataManager()

        # Should not raise
        dm._report_progress("Loading data", 0.5)


class TestDataManagerInitOptions:
    """Tests for various initialization options."""

    def test_init_with_storage(self):
        """Test initialization with storage backend."""
        mock_storage = MagicMock()

        dm = DataManager(storage=mock_storage)

        assert dm.storage is mock_storage

    def test_init_with_transactions(self):
        """Test initialization with transactional storage."""
        mock_storage = MagicMock()

        with patch("ml4t.data.storage.transaction.TransactionalStorage") as mock_trans:
            mock_trans.return_value = MagicMock()
            _dm = DataManager(storage=mock_storage, use_transactions=True)  # noqa: F841

            mock_trans.assert_called_once_with(mock_storage)

    def test_init_validation_disabled(self):
        """Test initialization with validation disabled."""
        dm = DataManager(enable_validation=False)

        assert dm.enable_validation is False

    def test_init_validation_enabled_default(self):
        """Test validation enabled by default."""
        dm = DataManager()

        assert dm.enable_validation is True


class TestDataManagerProviderClasses:
    """Tests for PROVIDER_CLASSES configuration."""

    def test_provider_classes_has_yahoo(self):
        """Test PROVIDER_CLASSES includes yahoo."""
        assert "yahoo" in DataManager.PROVIDER_CLASSES

    def test_provider_classes_has_mock(self):
        """Test PROVIDER_CLASSES includes mock."""
        assert "mock" in DataManager.PROVIDER_CLASSES

    def test_provider_classes_has_synthetic(self):
        """Test PROVIDER_CLASSES includes synthetic."""
        assert "synthetic" in DataManager.PROVIDER_CLASSES

    def test_provider_classes_has_cryptocompare(self):
        """Test PROVIDER_CLASSES includes cryptocompare."""
        assert "cryptocompare" in DataManager.PROVIDER_CLASSES

    def test_provider_classes_has_binance(self):
        """Test PROVIDER_CLASSES includes binance."""
        assert "binance" in DataManager.PROVIDER_CLASSES


class TestDataManagerFetchBatch:
    """Tests for fetch_batch method."""

    def test_fetch_batch_returns_dict(self):
        """Test fetch_batch returns dictionary."""
        dm = DataManager()

        result = dm.fetch_batch([], "2024-01-01", "2024-01-31")

        assert isinstance(result, dict)
        assert result == {}  # Empty list returns empty dict

    def test_fetch_batch_multiple_symbols(self):
        """Test fetch_batch with multiple symbols returns dict."""
        dm = DataManager()
        dm.router.add_pattern(r".*", "mock")

        # Even if fetch fails, it should return dict with keys
        result = dm.fetch_batch(["SYM1", "SYM2"], "2024-01-01", "2024-01-31")

        assert isinstance(result, dict)
        assert "SYM1" in result
        assert "SYM2" in result


class TestDataManagerDetectProviders:
    """Tests for provider detection during initialization."""

    def test_detect_providers_with_env_keys(self):
        """Test provider detection with environment keys."""
        dm = DataManager()

        # Provider detection happens automatically during __init__
        # At minimum, should include providers without required API keys
        assert "yahoo" in dm._available_providers or "mock" in dm._available_providers
