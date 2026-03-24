"""Shared base specifications for persisted ML4T artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactKind(str, Enum):
    """Kinds of persisted artifacts shared across ML4T workflows."""

    MARKET_DATA = "market_data"
    LABELS = "labels"
    FEATURES = "features"
    PREDICTIONS = "predictions"


@dataclass(frozen=True, slots=True)
class ArtifactStorage:
    """Storage location and serialization hints for an artifact."""

    path: str | Path = ""
    format: str = "parquet"
    partition_by: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> ArtifactStorage:
        if mapping is None:
            return cls()
        partition_by = mapping.get("partition_by", ())
        if isinstance(partition_by, str):
            partition_by = (partition_by,)
        return cls(
            path=mapping.get("path", ""),
            format=str(mapping.get("format", "parquet")),
            partition_by=tuple(str(item) for item in partition_by),
        )


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Upstream lineage and content fingerprinting for an artifact."""

    source_artifacts: tuple[str, ...] = ()
    content_hash: str | None = None
    created_by: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> ArtifactProvenance:
        if mapping is None:
            return cls()
        source_artifacts = mapping.get("source_artifacts", ())
        if isinstance(source_artifacts, str):
            source_artifacts = (source_artifacts,)
        return cls(
            source_artifacts=tuple(str(item) for item in source_artifacts),
            content_hash=_optional_str(mapping.get("content_hash")),
            created_by=_optional_str(mapping.get("created_by")),
        )


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Base metadata shared by all persisted artifact specifications."""

    artifact_id: str
    kind: ArtifactKind
    version: int = 1
    storage: ArtifactStorage = field(default_factory=ArtifactStorage)
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


def optional_str(value: Any) -> str | None:
    """Coerce a value to a non-empty string."""
    return _optional_str(value)


def serialize_artifact_value(value: Any) -> Any:
    """Serialize enums and pathlib objects for YAML/JSON output."""
    return _serialize(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


__all__ = [
    "ArtifactKind",
    "ArtifactProvenance",
    "ArtifactSpec",
    "ArtifactStorage",
    "optional_str",
    "serialize_artifact_value",
]
