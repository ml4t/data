"""Secure configuration serialization helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Atomically write YAML with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        temporary_path.chmod(0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
            yaml.safe_dump(
                data,
                temporary_file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
