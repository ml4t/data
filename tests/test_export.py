"""Tests for export functionality."""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from ml4t.data.core.models import DataObject, Metadata
from ml4t.data.export.formats import CSVExporter, ExcelExporter, ExportConfig, JSONExporter
from ml4t.data.export.manager import ExportManager
from ml4t.data.storage.filesystem import FileSystemBackend


class TestCSVExporter:
    """Test CSV export functionality."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data."""
        return pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [104.0, 105.0, 106.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

    def test_export_csv(self, sample_data: pl.DataFrame) -> None:
        """Test basic CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="csv",
            )

            exporter = CSVExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success
            assert result.rows_exported == 3
            assert result.output_path.exists()
            assert result.output_path.name == "TEST.csv"

            # Read back and verify
            df = pl.read_csv(result.output_path)
            assert len(df) == 3
            assert df["close"][0] == 104.0

    def test_export_csv_compressed(self, sample_data: pl.DataFrame) -> None:
        """Test compressed CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="csv",
                compression="gzip",
            )

            exporter = CSVExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success
            assert result.output_path.name == "TEST.csv.gz"
            assert result.output_path.exists()

    def test_export_csv_with_transformations(self, sample_data: pl.DataFrame) -> None:
        """Test CSV export with transformations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="csv",
                columns=["timestamp", "close"],
                add_returns=True,
            )

            exporter = CSVExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success

            # Read back and verify
            df = pl.read_csv(result.output_path)
            assert "close" in df.columns
            assert "returns" in df.columns
            assert len(df.columns) == 3  # timestamp, close, returns

    def test_export_batch_csv(self, sample_data: pl.DataFrame) -> None:
        """Test batch CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="csv",
            )

            datasets = {
                "AAPL": sample_data,
                "GOOGL": sample_data.clone(),
            }

            exporter = CSVExporter(config)
            results = exporter.export_batch(datasets)

            assert len(results) == 2
            assert all(r.success for r in results)
            assert (Path(tmpdir) / "AAPL.csv").exists()
            assert (Path(tmpdir) / "GOOGL.csv").exists()


class TestJSONExporter:
    """Test JSON export functionality."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data."""
        return pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [104.0, 105.0, 106.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

    def test_export_json(self, sample_data: pl.DataFrame) -> None:
        """Test basic JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="json",
            )

            exporter = JSONExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success
            assert result.rows_exported == 3
            assert result.output_path.exists()
            assert result.output_path.name == "TEST.json"

            # Read back and verify
            with open(result.output_path) as f:
                data = json.load(f)

            assert data["symbol"] == "TEST"
            assert len(data["data"]) == 3
            assert data["data"][0]["close"] == 104.0

    def test_export_json_with_metadata(self, sample_data: pl.DataFrame) -> None:
        """Test JSON export with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="json",
                include_metadata=True,
            )

            exporter = JSONExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success

            # Read back and verify
            with open(result.output_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert data["metadata"]["rows"] == 3
            assert "exported_at" in data["metadata"]

    def test_export_batch_json(self, sample_data: pl.DataFrame) -> None:
        """Test batch JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="json",
            )

            datasets = {
                "AAPL": sample_data,
                "GOOGL": sample_data.clone(),
            }

            exporter = JSONExporter(config)
            results = exporter.export_batch(datasets)

            assert len(results) == 1  # Single file for batch
            assert results[0].success

            # Read back and verify
            with open(results[0].output_path) as f:
                data = json.load(f)

            assert "AAPL" in data
            assert "GOOGL" in data
            assert len(data["AAPL"]) == 3
            assert len(data["GOOGL"]) == 3


class TestExcelExporter:
    """Test Excel export functionality."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data."""
        return pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [104.0, 105.0, 106.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

    def test_export_excel_basic(self, sample_data: pl.DataFrame) -> None:
        """Test basic Excel export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="excel",
            )

            exporter = ExcelExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success
            assert result.rows_exported == 3
            assert result.output_path.exists()
            assert result.output_path.suffix == ".xlsx"

    def test_export_excel_with_metadata(self, sample_data: pl.DataFrame) -> None:
        """Test Excel export with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="excel",
                include_metadata=True,
            )

            exporter = ExcelExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success
            assert result.output_path.exists()

    def test_export_excel_batch(self, sample_data: pl.DataFrame) -> None:
        """Test Excel batch export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir) / "batch_export.xlsx",
                format="excel",
            )

            datasets = {
                "AAPL": sample_data,
                "GOOGL": sample_data.clone(),
            }

            exporter = ExcelExporter(config)
            results = exporter.export_batch(datasets)

            assert len(results) == 1
            assert results[0].success
            assert results[0].output_path.exists()

    def test_export_excel_compression(self, sample_data: pl.DataFrame) -> None:
        """Test Excel export with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                output_path=Path(tmpdir),
                format="excel",
                compression="gzip",
            )

            exporter = ExcelExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success
            assert result.output_path.exists()
            # Compression adds .gz extension
            assert result.output_path.suffix == ".gz"

    def test_export_excel_custom_filename(self, sample_data: pl.DataFrame) -> None:
        """Test Excel export with custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "custom_name.xlsx"
            config = ExportConfig(
                output_path=output_file,
                format="excel",
            )

            exporter = ExcelExporter(config)
            result = exporter.export(sample_data, "TEST")

            assert result.success
            assert result.output_path == output_file
            assert result.output_path.exists()


class TestExportManager:
    """Test export manager functionality."""

    @pytest.fixture
    def sample_storage(self) -> FileSystemBackend:
        """Create storage with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemBackend(data_root=Path(tmpdir))

            # Create sample data
            from datetime import datetime

            df = pl.DataFrame(
                {
                    "timestamp": [
                        datetime(2024, 1, 1),
                        datetime(2024, 1, 2),
                        datetime(2024, 1, 3),
                    ],
                    "open": [100.0, 101.0, 102.0],
                    "high": [105.0, 106.0, 107.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [104.0, 105.0, 106.0],
                    "volume": [1000000, 1100000, 1200000],
                }
            )

            metadata = Metadata(
                provider="test",
                symbol="AAPL",
                bar_type="time",
                bar_params={"frequency": "daily"},
                asset_class="equities",
                start="2024-01-01",
                end="2024-01-03",
            )

            data_obj = DataObject(data=df, metadata=metadata)

            # Write AAPL
            storage.write(data_obj)

            # Write GOOGL with different symbol
            metadata_googl = Metadata(
                provider="test",
                symbol="GOOGL",
                bar_type="time",
                bar_params={"frequency": "daily"},
                asset_class="equities",
                start="2024-01-01",
                end="2024-01-03",
            )
            data_obj_googl = DataObject(data=df, metadata=metadata_googl)
            storage.write(data_obj_googl)

            yield storage

    def test_export_single_dataset(self, sample_storage: FileSystemBackend) -> None:
        """Test exporting a single dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExportManager(storage=sample_storage)

            result = manager.export(
                key="equities/daily/AAPL",
                output_path=tmpdir,
                format_type="csv",
            )

            assert result.success
            assert result.rows_exported == 3
            assert (Path(tmpdir) / "AAPL.csv").exists()

    def test_export_batch_datasets(self, sample_storage: FileSystemBackend) -> None:
        """Test exporting multiple datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExportManager(storage=sample_storage)

            results = manager.export_batch(
                keys=["equities/daily/AAPL", "equities/daily/GOOGL"],
                output_path=tmpdir,
                format_type="json",
            )

            assert len(results) == 1  # Single JSON file
            assert results[0].success

    def test_export_pattern(self, sample_storage: FileSystemBackend) -> None:
        """Test exporting datasets matching a pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExportManager(storage=sample_storage)

            results = manager.export_pattern(
                pattern="equities/daily/*",
                output_path=tmpdir,
                format_type="csv",
            )

            # CSV creates separate files
            assert len(results) == 2
            assert all(r.success for r in results)

    def test_list_formats(self) -> None:
        """Test listing available formats."""
        formats = ExportManager.list_formats()

        assert "csv" in formats
        assert "json" in formats
        assert "excel" in formats
        assert "xlsx" in formats
