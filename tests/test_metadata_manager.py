"""Tests for metadata manager normalization and fallback behavior."""

from __future__ import annotations

import json
from pathlib import Path

from ml4t.data.managers.metadata_manager import MetadataManager
from ml4t.data.storage.key_codec import encode_storage_key


class _StorageWithGetMetadata:
    """Simple storage stub exposing get_metadata only."""

    def __init__(self, metadata: dict) -> None:
        self._metadata = metadata

    def get_metadata(self, _key: str) -> dict:
        return self._metadata


class _StorageWithMetadataDir:
    """Simple storage stub exposing metadata_dir only."""

    def __init__(self, metadata_dir: Path) -> None:
        self.metadata_dir = metadata_dir


def test_get_metadata_for_key_normalizes_manifest_shape() -> None:
    """MetadataManager should expose canonical fields from manifest custom payload."""
    manager = MetadataManager(
        _StorageWithGetMetadata(
            {
                "row_count": 25,
                "custom": {
                    "provider": "yahoo",
                    "symbol": "AAPL",
                    "asset_class": "equities",
                    "frequency": "daily",
                    "start_date": "2024-01-01T00:00:00",
                    "end_date": "2024-01-31T00:00:00",
                },
            }
        )
    )

    metadata = manager.get_metadata_for_key("equities/daily/AAPL")
    assert metadata is not None
    assert metadata["provider"] == "yahoo"
    assert metadata["symbol"] == "AAPL"
    assert metadata["asset_class"] == "equities"
    assert metadata["frequency"] == "daily"
    assert metadata["total_rows"] == 25
    assert metadata["data_range"] == {
        "start": "2024-01-01T00:00:00",
        "end": "2024-01-31T00:00:00",
    }


def test_get_metadata_for_key_reads_encoded_metadata_file(tmp_path: Path) -> None:
    """MetadataManager should read encoded metadata filenames when storage API is absent."""
    metadata_dir = tmp_path / ".metadata"
    metadata_dir.mkdir()

    key = "crypto/hourly/BTCUSD"
    metadata_file = metadata_dir / f"{encode_storage_key(key)}.json"
    metadata_file.write_text(
        json.dumps(
            {
                "row_count": 10,
                "custom": {
                    "provider": "cryptocompare",
                    "symbol": "BTCUSD",
                    "asset_class": "crypto",
                    "frequency": "hourly",
                },
            }
        )
    )

    manager = MetadataManager(_StorageWithMetadataDir(metadata_dir))
    metadata = manager.get_metadata_for_key(key)

    assert metadata is not None
    assert metadata["provider"] == "cryptocompare"
    assert metadata["symbol"] == "BTCUSD"
    assert metadata["asset_class"] == "crypto"
    assert metadata["frequency"] == "hourly"
    assert metadata["total_rows"] == 10
