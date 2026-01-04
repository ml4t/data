"""Comprehensive tests for metadata tracker functionality."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ml4t.data.storage.metadata_tracker import (
    DatasetMetadata,
    MetadataTracker,
    UpdateRecord,
)


class TestUpdateRecord:
    """Tests for UpdateRecord dataclass."""

    def test_create_update_record(self) -> None:
        """Test creating an update record with all fields."""
        record = UpdateRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            update_type="incremental",
            rows_before=100,
            rows_after=150,
            rows_added=50,
            rows_updated=10,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 15),
            provider="yahoo",
            duration_seconds=5.5,
            gaps_filled=2,
            errors=["warning: missing data for 2024-01-05"],
        )

        assert record.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert record.update_type == "incremental"
        assert record.rows_before == 100
        assert record.rows_after == 150
        assert record.rows_added == 50
        assert record.rows_updated == 10
        assert record.provider == "yahoo"
        assert record.duration_seconds == 5.5
        assert record.gaps_filled == 2
        assert len(record.errors) == 1

    def test_update_record_default_errors(self) -> None:
        """Test that errors defaults to empty list."""
        record = UpdateRecord(
            timestamp=datetime(2024, 1, 15),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 15),
            provider="yahoo",
            duration_seconds=10.0,
        )

        assert record.errors == []

    def test_update_record_to_dict(self) -> None:
        """Test serialization to dictionary."""
        record = UpdateRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            update_type="incremental",
            rows_before=100,
            rows_after=150,
            rows_added=50,
            rows_updated=10,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 15),
            provider="yahoo",
            duration_seconds=5.5,
            gaps_filled=2,
        )

        result = record.to_dict()

        assert result["timestamp"] == "2024-01-15T10:30:00"
        assert result["update_type"] == "incremental"
        assert result["rows_before"] == 100
        assert result["rows_after"] == 150
        assert result["rows_added"] == 50
        assert result["rows_updated"] == 10
        assert result["start_date"] == "2024-01-01T00:00:00"
        assert result["end_date"] == "2024-01-15T00:00:00"
        assert result["provider"] == "yahoo"
        assert result["duration_seconds"] == 5.5
        assert result["gaps_filled"] == 2
        assert result["errors"] == []

    def test_update_record_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "timestamp": "2024-01-15T10:30:00",
            "update_type": "full",
            "rows_before": 0,
            "rows_after": 100,
            "rows_added": 100,
            "rows_updated": 0,
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-15T00:00:00",
            "provider": "binance",
            "duration_seconds": 15.0,
            "gaps_filled": 0,
            "errors": ["test error"],
        }

        record = UpdateRecord.from_dict(data)

        assert record.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert record.update_type == "full"
        assert record.rows_before == 0
        assert record.rows_after == 100
        assert record.provider == "binance"
        assert record.gaps_filled == 0
        assert record.errors == ["test error"]

    def test_update_record_from_dict_missing_optional_fields(self) -> None:
        """Test deserialization handles missing optional fields."""
        data = {
            "timestamp": "2024-01-15T10:30:00",
            "update_type": "full",
            "rows_before": 0,
            "rows_after": 100,
            "rows_added": 100,
            "rows_updated": 0,
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-15T00:00:00",
            "provider": "yahoo",
            "duration_seconds": 10.0,
            # gaps_filled and errors are missing
        }

        record = UpdateRecord.from_dict(data)

        assert record.gaps_filled == 0
        assert record.errors == []

    def test_update_record_roundtrip(self) -> None:
        """Test serialization/deserialization roundtrip."""
        original = UpdateRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            update_type="gap_fill",
            rows_before=100,
            rows_after=110,
            rows_added=10,
            rows_updated=5,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 15),
            provider="eodhd",
            duration_seconds=3.5,
            gaps_filled=10,
            errors=["gap at 2024-01-10"],
        )

        roundtrip = UpdateRecord.from_dict(original.to_dict())

        assert roundtrip.timestamp == original.timestamp
        assert roundtrip.update_type == original.update_type
        assert roundtrip.rows_before == original.rows_before
        assert roundtrip.rows_after == original.rows_after
        assert roundtrip.rows_added == original.rows_added
        assert roundtrip.rows_updated == original.rows_updated
        assert roundtrip.start_date == original.start_date
        assert roundtrip.end_date == original.end_date
        assert roundtrip.provider == original.provider
        assert roundtrip.duration_seconds == original.duration_seconds
        assert roundtrip.gaps_filled == original.gaps_filled
        assert roundtrip.errors == original.errors


class TestDatasetMetadata:
    """Tests for DatasetMetadata dataclass."""

    def test_create_dataset_metadata(self) -> None:
        """Test creating dataset metadata with all fields."""
        metadata = DatasetMetadata(
            symbol="AAPL",
            asset_class="equities",
            frequency="daily",
            provider="yahoo",
            first_update=datetime(2024, 1, 1),
            last_update=datetime(2024, 1, 15),
            total_rows=1000,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2024, 1, 15),
            update_count=5,
            last_check=datetime(2024, 1, 15, 10, 0, 0),
            health_status="healthy",
            health_message="",
        )

        assert metadata.symbol == "AAPL"
        assert metadata.asset_class == "equities"
        assert metadata.frequency == "daily"
        assert metadata.provider == "yahoo"
        assert metadata.total_rows == 1000
        assert metadata.update_count == 5
        assert metadata.health_status == "healthy"

    def test_dataset_metadata_defaults(self) -> None:
        """Test default values for optional fields."""
        metadata = DatasetMetadata(
            symbol="BTC",
            asset_class="crypto",
            frequency="hourly",
            provider="binance",
            first_update=datetime(2024, 1, 1),
            last_update=datetime(2024, 1, 1),
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 1),
            update_count=1,
        )

        assert metadata.last_check is None
        assert metadata.health_status == "healthy"
        assert metadata.health_message == ""

    def test_dataset_metadata_to_dict(self) -> None:
        """Test serialization to dictionary."""
        metadata = DatasetMetadata(
            symbol="AAPL",
            asset_class="equities",
            frequency="daily",
            provider="yahoo",
            first_update=datetime(2024, 1, 1),
            last_update=datetime(2024, 1, 15),
            total_rows=1000,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2024, 1, 15),
            update_count=5,
            last_check=datetime(2024, 1, 15, 10, 0, 0),
            health_status="stale",
            health_message="Data is 10 days old",
        )

        result = metadata.to_dict()

        assert result["symbol"] == "AAPL"
        assert result["asset_class"] == "equities"
        assert result["frequency"] == "daily"
        assert result["provider"] == "yahoo"
        assert result["first_update"] == "2024-01-01T00:00:00"
        assert result["last_update"] == "2024-01-15T00:00:00"
        assert result["total_rows"] == 1000
        assert result["date_range_start"] == "2023-01-01T00:00:00"
        assert result["date_range_end"] == "2024-01-15T00:00:00"
        assert result["update_count"] == 5
        assert result["last_check"] == "2024-01-15T10:00:00"
        assert result["health_status"] == "stale"
        assert result["health_message"] == "Data is 10 days old"

    def test_dataset_metadata_to_dict_null_last_check(self) -> None:
        """Test serialization handles None last_check."""
        metadata = DatasetMetadata(
            symbol="AAPL",
            asset_class="equities",
            frequency="daily",
            provider="yahoo",
            first_update=datetime(2024, 1, 1),
            last_update=datetime(2024, 1, 15),
            total_rows=1000,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2024, 1, 15),
            update_count=5,
            last_check=None,
        )

        result = metadata.to_dict()

        assert result["last_check"] is None

    def test_dataset_metadata_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "symbol": "GOOGL",
            "asset_class": "equities",
            "frequency": "daily",
            "provider": "polygon",
            "first_update": "2024-01-01T00:00:00",
            "last_update": "2024-01-15T00:00:00",
            "total_rows": 500,
            "date_range_start": "2023-06-01T00:00:00",
            "date_range_end": "2024-01-15T00:00:00",
            "update_count": 10,
            "last_check": "2024-01-15T12:00:00",
            "health_status": "error",
            "health_message": "API error",
        }

        metadata = DatasetMetadata.from_dict(data)

        assert metadata.symbol == "GOOGL"
        assert metadata.asset_class == "equities"
        assert metadata.provider == "polygon"
        assert metadata.total_rows == 500
        assert metadata.update_count == 10
        assert metadata.last_check == datetime(2024, 1, 15, 12, 0, 0)
        assert metadata.health_status == "error"
        assert metadata.health_message == "API error"

    def test_dataset_metadata_from_dict_missing_optional_fields(self) -> None:
        """Test deserialization handles missing optional fields."""
        data = {
            "symbol": "ETH",
            "asset_class": "crypto",
            "frequency": "hourly",
            "provider": "binance",
            "first_update": "2024-01-01T00:00:00",
            "last_update": "2024-01-15T00:00:00",
            "total_rows": 100,
            "date_range_start": "2024-01-01T00:00:00",
            "date_range_end": "2024-01-15T00:00:00",
            "update_count": 1,
            # last_check, health_status, health_message missing
        }

        metadata = DatasetMetadata.from_dict(data)

        assert metadata.last_check is None
        assert metadata.health_status == "healthy"
        assert metadata.health_message == ""

    def test_dataset_metadata_roundtrip(self) -> None:
        """Test serialization/deserialization roundtrip."""
        original = DatasetMetadata(
            symbol="MSFT",
            asset_class="equities",
            frequency="minute",
            provider="databento",
            first_update=datetime(2024, 1, 1, 9, 30, 0),
            last_update=datetime(2024, 1, 15, 16, 0, 0),
            total_rows=100000,
            date_range_start=datetime(2024, 1, 1, 9, 30, 0),
            date_range_end=datetime(2024, 1, 15, 16, 0, 0),
            update_count=100,
            last_check=datetime(2024, 1, 15, 16, 5, 0),
            health_status="healthy",
            health_message="All systems operational",
        )

        roundtrip = DatasetMetadata.from_dict(original.to_dict())

        assert roundtrip.symbol == original.symbol
        assert roundtrip.asset_class == original.asset_class
        assert roundtrip.frequency == original.frequency
        assert roundtrip.provider == original.provider
        assert roundtrip.first_update == original.first_update
        assert roundtrip.last_update == original.last_update
        assert roundtrip.total_rows == original.total_rows
        assert roundtrip.update_count == original.update_count
        assert roundtrip.last_check == original.last_check
        assert roundtrip.health_status == original.health_status
        assert roundtrip.health_message == original.health_message


class TestMetadataTracker:
    """Tests for MetadataTracker class."""

    @pytest.fixture
    def tracker(self, tmp_path: Path) -> MetadataTracker:
        """Create a MetadataTracker instance with temporary storage."""
        return MetadataTracker(tmp_path)

    @pytest.fixture
    def sample_update_record(self) -> UpdateRecord:
        """Create a sample update record for testing."""
        return UpdateRecord(
            timestamp=datetime.now(),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 15),
            provider="yahoo",
            duration_seconds=5.0,
        )

    def test_init_creates_metadata_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates metadata directory."""
        tracker = MetadataTracker(tmp_path)

        assert tracker.metadata_dir.exists()
        assert tracker.metadata_dir == tmp_path / ".metadata"

    def test_get_metadata_path(self, tracker: MetadataTracker) -> None:
        """Test metadata path generation."""
        path = tracker._get_metadata_path("equities/daily/AAPL")

        assert path.name == "equities_daily_AAPL_metadata.json"

    def test_get_history_path(self, tracker: MetadataTracker) -> None:
        """Test history path generation."""
        path = tracker._get_history_path("crypto/hourly/BTC")

        assert path.name == "crypto_hourly_BTC_history.json"

    def test_get_metadata_nonexistent(self, tracker: MetadataTracker) -> None:
        """Test getting metadata for nonexistent dataset returns None."""
        result = tracker.get_metadata("nonexistent/dataset/KEY")

        assert result is None

    def test_update_metadata_creates_new(
        self,
        tracker: MetadataTracker,
        sample_update_record: UpdateRecord,
    ) -> None:
        """Test updating metadata creates new record."""
        key = "equities/daily/AAPL"

        metadata = tracker.update_metadata(
            key=key,
            update_record=sample_update_record,
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 15),
        )

        assert metadata.symbol == "AAPL"
        assert metadata.asset_class == "equities"
        assert metadata.frequency == "daily"
        assert metadata.provider == "yahoo"
        assert metadata.total_rows == 100
        assert metadata.update_count == 1
        assert metadata.health_status == "healthy"

    def test_update_metadata_updates_existing(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test updating existing metadata increments count."""
        key = "equities/daily/AAPL"

        # First update
        record1 = UpdateRecord(
            timestamp=datetime(2024, 1, 10),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            provider="yahoo",
            duration_seconds=5.0,
        )

        tracker.update_metadata(
            key=key,
            update_record=record1,
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 10),
        )

        # Second update
        record2 = UpdateRecord(
            timestamp=datetime(2024, 1, 15),
            update_type="incremental",
            rows_before=100,
            rows_after=150,
            rows_added=50,
            rows_updated=0,
            start_date=datetime(2024, 1, 10),
            end_date=datetime(2024, 1, 15),
            provider="yahoo",
            duration_seconds=3.0,
        )

        metadata = tracker.update_metadata(
            key=key,
            update_record=record2,
            total_rows=150,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 15),
        )

        assert metadata.update_count == 2
        assert metadata.total_rows == 150
        assert metadata.last_update == record2.timestamp

    def test_update_metadata_simple_key(
        self,
        tracker: MetadataTracker,
        sample_update_record: UpdateRecord,
    ) -> None:
        """Test updating metadata with simple key (no slashes)."""
        key = "simple_key"

        metadata = tracker.update_metadata(
            key=key,
            update_record=sample_update_record,
            total_rows=50,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 15),
        )

        # With simple key, asset_class and frequency should be empty
        assert metadata.symbol == "simple_key"
        assert metadata.asset_class == ""
        assert metadata.frequency == ""

    def test_get_metadata_after_update(
        self,
        tracker: MetadataTracker,
        sample_update_record: UpdateRecord,
    ) -> None:
        """Test getting metadata after update."""
        key = "equities/daily/MSFT"

        tracker.update_metadata(
            key=key,
            update_record=sample_update_record,
            total_rows=200,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 15),
        )

        result = tracker.get_metadata(key)

        assert result is not None
        assert result.symbol == "MSFT"
        assert result.total_rows == 200

    def test_add_update_record(
        self,
        tracker: MetadataTracker,
        sample_update_record: UpdateRecord,
    ) -> None:
        """Test adding update record to history."""
        key = "equities/daily/AAPL"

        tracker.add_update_record(key, sample_update_record)

        history = tracker.get_update_history(key)

        assert len(history) == 1
        assert history[0].update_type == sample_update_record.update_type

    def test_add_multiple_update_records(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test adding multiple update records."""
        key = "equities/daily/AAPL"

        for i in range(5):
            record = UpdateRecord(
                timestamp=datetime(2024, 1, i + 1),
                update_type="incremental",
                rows_before=i * 10,
                rows_after=(i + 1) * 10,
                rows_added=10,
                rows_updated=0,
                start_date=datetime(2024, 1, i + 1),
                end_date=datetime(2024, 1, i + 1),
                provider="yahoo",
                duration_seconds=1.0,
            )
            tracker.add_update_record(key, record)

        history = tracker.get_update_history(key, limit=10)

        assert len(history) == 5
        # Most recent first
        assert history[0].timestamp == datetime(2024, 1, 5)

    def test_update_history_limit(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test history respects limit parameter."""
        key = "equities/daily/AAPL"

        for i in range(10):
            record = UpdateRecord(
                timestamp=datetime(2024, 1, i + 1),
                update_type="incremental",
                rows_before=0,
                rows_after=10,
                rows_added=10,
                rows_updated=0,
                start_date=datetime(2024, 1, i + 1),
                end_date=datetime(2024, 1, i + 1),
                provider="yahoo",
                duration_seconds=1.0,
            )
            tracker.add_update_record(key, record)

        history = tracker.get_update_history(key, limit=3)

        assert len(history) == 3

    def test_update_history_trims_to_100(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test history is trimmed to 100 records max."""
        key = "equities/daily/AAPL"

        # Add 120 records
        for i in range(120):
            record = UpdateRecord(
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                update_type="incremental",
                rows_before=0,
                rows_after=10,
                rows_added=10,
                rows_updated=0,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1),
                provider="yahoo",
                duration_seconds=1.0,
            )
            tracker.add_update_record(key, record)

        history = tracker.get_update_history(key, limit=200)

        # Should be capped at 100
        assert len(history) == 100

    def test_get_update_history_empty(self, tracker: MetadataTracker) -> None:
        """Test getting history for dataset with no history."""
        result = tracker.get_update_history("nonexistent/key")

        assert result == []

    def test_check_health_no_metadata(self, tracker: MetadataTracker) -> None:
        """Test health check for nonexistent dataset."""
        status, message = tracker.check_health("nonexistent/key")

        assert status == "error"
        assert message == "No metadata found"

    def test_check_health_healthy(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test health check for healthy dataset."""
        key = "equities/daily/AAPL"
        record = UpdateRecord(
            timestamp=datetime.now(),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime.now() - timedelta(days=1),
            provider="yahoo",
            duration_seconds=5.0,
        )

        tracker.update_metadata(
            key=key,
            update_record=record,
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime.now() - timedelta(days=1),
        )

        status, message = tracker.check_health(key)

        assert status == "healthy"
        assert "up to date" in message

    def test_check_health_stale_update(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test health check for stale dataset (old last_update)."""
        key = "equities/daily/AAPL"
        record = UpdateRecord(
            timestamp=datetime.now() - timedelta(days=10),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 15),
            provider="yahoo",
            duration_seconds=5.0,
        )

        tracker.update_metadata(
            key=key,
            update_record=record,
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 15),
        )

        status, message = tracker.check_health(key, stale_days=7)

        assert status == "stale"
        assert "days old" in message

    def test_check_health_stale_data_range(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test health check for dataset with stale data range."""
        key = "equities/daily/AAPL"
        record = UpdateRecord(
            timestamp=datetime.now(),  # Recent update
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 15),  # Old end date
            provider="yahoo",
            duration_seconds=5.0,
        )

        tracker.update_metadata(
            key=key,
            update_record=record,
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 15),
        )

        status, message = tracker.check_health(key)

        assert status == "stale"
        assert "days behind" in message

    def test_check_health_with_errors(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test health check for dataset with recent errors."""
        key = "equities/daily/AAPL"

        # First create metadata (this adds one record without errors)
        record = UpdateRecord(
            timestamp=datetime.now(),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime.now() - timedelta(days=1),
            provider="yahoo",
            duration_seconds=5.0,
        )

        tracker.update_metadata(
            key=key,
            update_record=record,
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime.now() - timedelta(days=1),
        )

        # Add 4 more records with errors (now 4 of last 5 have errors)
        for i in range(4):
            error_record = UpdateRecord(
                timestamp=datetime.now() + timedelta(seconds=i + 1),
                update_type="incremental",
                rows_before=100,
                rows_after=100,
                rows_added=0,
                rows_updated=0,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now(),
                provider="yahoo",
                duration_seconds=1.0,
                errors=["API error"],  # All have errors
            )
            tracker.add_update_record(key, error_record)

        status, message = tracker.check_health(key)

        assert status == "error"
        assert "errors" in message

    def test_get_summary_empty(self, tracker: MetadataTracker) -> None:
        """Test summary for empty tracker."""
        summary = tracker.get_summary()

        assert summary["total_datasets"] == 0
        assert summary["healthy"] == 0
        assert summary["stale"] == 0
        assert summary["error"] == 0
        assert summary["total_rows"] == 0
        assert summary["total_updates"] == 0
        assert summary["by_asset_class"] == {}

    def test_get_summary_with_datasets(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test summary with multiple datasets."""
        # Add equity
        record1 = UpdateRecord(
            timestamp=datetime.now(),
            update_type="full",
            rows_before=0,
            rows_after=100,
            rows_added=100,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime.now() - timedelta(days=1),
            provider="yahoo",
            duration_seconds=5.0,
        )
        tracker.update_metadata(
            key="equities/daily/AAPL",
            update_record=record1,
            total_rows=100,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime.now() - timedelta(days=1),
        )

        # Add crypto
        record2 = UpdateRecord(
            timestamp=datetime.now(),
            update_type="full",
            rows_before=0,
            rows_after=200,
            rows_added=200,
            rows_updated=0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime.now() - timedelta(days=1),
            provider="binance",
            duration_seconds=3.0,
        )
        tracker.update_metadata(
            key="crypto/hourly/BTC",
            update_record=record2,
            total_rows=200,
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime.now() - timedelta(days=1),
        )

        summary = tracker.get_summary()

        assert summary["total_datasets"] == 2
        assert summary["total_rows"] == 300
        assert summary["total_updates"] == 2
        assert summary["by_asset_class"]["equities"] == 1
        assert summary["by_asset_class"]["crypto"] == 1

    def test_concurrent_updates(
        self,
        tracker: MetadataTracker,
    ) -> None:
        """Test that concurrent updates don't corrupt data."""
        key = "equities/daily/AAPL"

        # Simulate concurrent updates
        for i in range(10):
            record = UpdateRecord(
                timestamp=datetime.now() + timedelta(seconds=i),
                update_type="incremental",
                rows_before=i * 10,
                rows_after=(i + 1) * 10,
                rows_added=10,
                rows_updated=0,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1),
                provider="yahoo",
                duration_seconds=1.0,
            )
            tracker.update_metadata(
                key=key,
                update_record=record,
                total_rows=(i + 1) * 10,
                date_range_start=datetime(2024, 1, 1),
                date_range_end=datetime(2024, 1, 1),
            )

        metadata = tracker.get_metadata(key)

        assert metadata is not None
        assert metadata.update_count == 10
        assert metadata.total_rows == 100
