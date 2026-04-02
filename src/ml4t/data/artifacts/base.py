"""Backward-compatible re-exports for ML4T artifact base specs."""

from ml4t.specs.base import (
    ArtifactKind,
    ArtifactProvenance,
    ArtifactSpec,
    ArtifactStorage,
    optional_str,
    serialize_artifact_value,
)

__all__ = [
    "ArtifactKind",
    "ArtifactProvenance",
    "ArtifactSpec",
    "ArtifactStorage",
    "optional_str",
    "serialize_artifact_value",
]
