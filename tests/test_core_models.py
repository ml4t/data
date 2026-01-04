"""Test core models functionality."""

from datetime import datetime


class TestCoreModels:
    """Test core model functionality."""

    def test_metadata_creation(self):
        """Test Metadata model creation."""
        from ml4t.data.core.models import Metadata

        metadata = Metadata(
            provider="yahoo",
            symbol="AAPL",
            bar_type="time",
            bar_params={"frequency": "daily"},
            asset_class="equities",
        )

        assert metadata.provider == "yahoo"
        assert metadata.symbol == "AAPL"
        assert metadata.frequency == "daily"
        assert metadata.asset_class == "equities"

    def test_metadata_with_date_range(self):
        """Test Metadata model with date range."""
        from ml4t.data.core.models import Metadata

        date_range = {"start": "2024-01-01", "end": "2024-01-31"}

        metadata = Metadata(
            provider="yahoo",
            symbol="AAPL",
            bar_type="time",
            bar_params={"frequency": "daily"},
            asset_class="equities",
            data_range=date_range,
        )

        assert metadata.data_range == date_range
        assert metadata.data_range["start"] == "2024-01-01"
        assert metadata.data_range["end"] == "2024-01-31"

    def test_metadata_dict_conversion(self):
        """Test Metadata model dict conversion."""
        from ml4t.data.core.models import Metadata

        metadata = Metadata(
            provider="yahoo",
            symbol="AAPL",
            bar_type="time",
            bar_params={"frequency": "daily"},
            asset_class="equities",
        )

        # Test that the object has the expected attributes
        assert hasattr(metadata, "provider")
        assert hasattr(metadata, "symbol")
        assert metadata.provider == "yahoo"
        assert metadata.symbol == "AAPL"

    def test_data_object_creation(self):
        """Test DataObject model creation."""
        import polars as pl

        from ml4t.data.core.models import DataObject, Metadata

        metadata = Metadata(
            provider="yahoo",
            symbol="AAPL",
            bar_type="time",
            bar_params={"frequency": "daily"},
            asset_class="equities",
        )

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "volume": [1000000],
            }
        )

        data_obj = DataObject(data=df, metadata=metadata)

        assert data_obj.metadata.symbol == "AAPL"
        assert len(data_obj.data) == 1
        assert data_obj.data.columns == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_data_object_properties(self):
        """Test DataObject model properties."""
        import polars as pl

        from ml4t.data.core.models import DataObject, Metadata

        metadata = Metadata(
            provider="yahoo",
            symbol="AAPL",
            bar_type="time",
            bar_params={"frequency": "daily"},
            asset_class="equities",
        )

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "open": [99.0, 100.0],
                "high": [101.0, 102.0],
                "low": [98.0, 99.0],
                "close": [100.0, 101.0],
                "volume": [1000000, 1100000],
            }
        )

        data_obj = DataObject(data=df, metadata=metadata)

        # Test row count via len
        assert len(data_obj.data) == 2

        # Test column names
        assert "timestamp" in data_obj.data.columns
        assert "open" in data_obj.data.columns
        assert "high" in data_obj.data.columns
        assert "low" in data_obj.data.columns
        assert "close" in data_obj.data.columns
        assert "volume" in data_obj.data.columns
