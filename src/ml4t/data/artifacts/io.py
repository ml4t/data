"""Backward-compatible re-exports for ML4T artifact spec I/O."""

from ml4t.specs.io import read_spec_payload, write_spec_payload

__all__ = ["read_spec_payload", "write_spec_payload"]
