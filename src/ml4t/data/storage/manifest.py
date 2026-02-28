"""Canonical manifest contract helpers for storage metadata."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ml4t.data.core.keys import is_storage_key, parse_storage_key

MANIFEST_VERSION = "1.0"


def _parse_key(key: str) -> tuple[str, str, str]:
    """Parse canonical storage key into parts, with simple-key fallback."""
    if is_storage_key(key):
        return parse_storage_key(key)
    return "", "", key


def _parse_datetime(value: Any) -> datetime | None:
    """Parse datetime values from common metadata payload formats."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_iso(value: Any) -> str:
    """Convert datetime-like values to ISO string, returning empty string when missing."""
    parsed = _parse_datetime(value)
    if parsed is None:
        return ""
    return parsed.isoformat()


def _coerce_int(value: Any, default: int = 0) -> int:
    """Coerce integer-like values while keeping invalid inputs safe."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


class ManifestV1(BaseModel):
    """Canonical storage metadata manifest."""

    manifest_version: str = MANIFEST_VERSION
    key: str
    provider: str = ""
    symbol: str = ""
    asset_class: str = ""
    frequency: str = ""
    exchange: str = ""
    bar_type: str = ""
    row_count: int = 0
    total_rows: int = 0
    data_range: dict[str, str] = Field(default_factory=dict)
    last_updated: str = ""
    last_update: str = ""
    first_update: str = ""
    update_history: list[dict[str, Any]] = Field(default_factory=list)
    schema_columns: list[str] = Field(default_factory=list, alias="schema")
    partitions: list[str] = Field(default_factory=list)
    file_path: str = ""
    file_size_mb: float = 0.0
    custom: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow", "populate_by_name": True}

    @classmethod
    def from_payload(
        cls,
        key: str,
        payload: dict[str, Any] | None,
    ) -> ManifestV1:
        """Normalize arbitrary manifest payload into canonical contract."""
        raw = payload or {}
        custom = raw.get("custom")
        if not isinstance(custom, dict):
            custom = {}

        key_asset_class, key_frequency, key_symbol = _parse_key(key)
        provider = raw.get("provider") or custom.get("provider") or ""
        symbol = raw.get("symbol") or custom.get("symbol") or key_symbol
        asset_class = raw.get("asset_class") or custom.get("asset_class") or key_asset_class
        frequency = raw.get("frequency") or custom.get("frequency") or key_frequency
        exchange = raw.get("exchange") or custom.get("exchange") or ""
        bar_type = raw.get("bar_type") or custom.get("bar_type") or ""

        raw_data_range = raw.get("data_range")
        if not isinstance(raw_data_range, dict):
            raw_data_range = {}

        start = (
            raw_data_range.get("start")
            or raw_data_range.get("start_date")
            or custom.get("start_date")
            or custom.get("start")
        )
        end = (
            raw_data_range.get("end")
            or raw_data_range.get("end_date")
            or custom.get("end_date")
            or custom.get("end")
        )

        last_updated = (
            raw.get("last_updated")
            or raw.get("last_update")
            or raw.get("download_utc_timestamp")
            or custom.get("last_updated")
            or custom.get("last_update")
            or ""
        )
        first_update = raw.get("first_update") or raw.get("download_utc_timestamp") or last_updated
        row_count = _coerce_int(raw.get("row_count", raw.get("total_rows", 0)))

        update_history = raw.get("update_history")
        if not isinstance(update_history, list):
            update_history = []

        schema = raw.get("schema")
        if not isinstance(schema, list):
            schema = []

        partitions = raw.get("partitions")
        if not isinstance(partitions, list):
            partitions = []

        return cls(
            manifest_version=str(raw.get("manifest_version", MANIFEST_VERSION)),
            key=key,
            provider=provider,
            symbol=symbol,
            asset_class=asset_class,
            frequency=frequency,
            exchange=exchange,
            bar_type=bar_type,
            row_count=row_count,
            total_rows=row_count,
            data_range={"start": _to_iso(start), "end": _to_iso(end)},
            last_updated=_to_iso(last_updated),
            last_update=_to_iso(last_updated),
            first_update=_to_iso(first_update),
            update_history=update_history,
            schema_columns=schema,
            partitions=partitions,
            file_path=str(raw.get("file_path", "")),
            file_size_mb=float(raw.get("file_size_mb", 0.0) or 0.0),
            custom=custom,
        )


def manifest_from_write(
    key: str,
    row_count: int,
    schema: list[str],
    custom: dict[str, Any] | None = None,
    *,
    range_start: Any = None,
    range_end: Any = None,
    partitions: list[str] | None = None,
    file_path: str | None = None,
    file_size_mb: float | None = None,
) -> dict[str, Any]:
    """Create canonical manifest payload for write operations."""
    now = datetime.now().isoformat()
    base = ManifestV1.from_payload(key, {"custom": custom or {}})
    payload = base.model_dump(by_alias=True)
    payload["manifest_version"] = MANIFEST_VERSION
    payload["row_count"] = row_count
    payload["total_rows"] = row_count
    payload["schema"] = schema
    payload["last_updated"] = now
    payload["last_update"] = now
    payload["first_update"] = payload.get("first_update") or now

    data_range = {
        "start": _to_iso(range_start) or payload["data_range"].get("start", ""),
        "end": _to_iso(range_end) or payload["data_range"].get("end", ""),
    }
    payload["data_range"] = data_range

    if partitions is not None:
        payload["partitions"] = partitions
    if file_path is not None:
        payload["file_path"] = file_path
    if file_size_mb is not None:
        payload["file_size_mb"] = file_size_mb

    return payload


def upgrade_manifest_payload(
    key: str,
    payload: dict[str, Any] | None,
    *,
    emit_deprecation_warning: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Upgrade legacy manifests to the canonical v1 contract."""
    raw = payload or {}
    current_version = raw.get("manifest_version")
    normalized = ManifestV1.from_payload(key, raw).model_dump(by_alias=True)
    normalized["manifest_version"] = MANIFEST_VERSION

    changed = normalized != raw

    if emit_deprecation_warning and current_version != MANIFEST_VERSION:
        if current_version is None:
            warnings.warn(
                "Legacy manifest without 'manifest_version' detected; upgraded to 1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Legacy manifest version '{current_version}' detected; upgraded to 1.0.",
                DeprecationWarning,
                stacklevel=2,
            )

    return normalized, changed


def manifest_with_incremental_update(
    key: str,
    payload: dict[str, Any] | None,
    *,
    last_update: datetime,
    records_added: int,
    chunk_file: str,
    max_history: int = 100,
) -> dict[str, Any]:
    """Apply incremental update metadata onto manifest payload."""
    normalized, _ = upgrade_manifest_payload(key, payload)

    normalized["manifest_version"] = MANIFEST_VERSION
    normalized["last_updated"] = last_update.isoformat()
    normalized["last_update"] = last_update.isoformat()
    if not normalized.get("first_update"):
        normalized["first_update"] = last_update.isoformat()

    history = normalized.get("update_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "timestamp": last_update.isoformat(),
            "records_added": records_added,
            "chunk_file": chunk_file,
        }
    )
    normalized["update_history"] = history[-max_history:]

    return normalized


def normalize_manifest_payload(key: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize payload and expose compatibility aliases used across the codebase."""
    normalized, _ = upgrade_manifest_payload(key, payload)
    update_history = normalized.get("update_history")
    update_count = len(update_history) if isinstance(update_history, list) else 0
    normalized["update_count"] = update_count
    normalized["attributes"] = {"last_update": normalized.get("last_updated", "")}
    return normalized
