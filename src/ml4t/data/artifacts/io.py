"""Low-level payload I/O for ML4T artifact specifications."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def read_spec_payload(path_or_mapping: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a spec payload from YAML/JSON or return a copied mapping."""
    if isinstance(path_or_mapping, Mapping):
        return dict(path_or_mapping)

    path = Path(path_or_mapping)
    with path.open() as f:
        data = json.load(f) if path.suffix.lower() == ".json" else yaml.safe_load(f)
    return dict(data or {})


def write_spec_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a spec payload to YAML or JSON."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        if dest.suffix.lower() == ".json":
            json.dump(dict(payload), f, indent=2)
            f.write("\n")
        else:
            yaml.safe_dump(dict(payload), f, sort_keys=False)
    return dest


__all__ = ["read_spec_payload", "write_spec_payload"]
