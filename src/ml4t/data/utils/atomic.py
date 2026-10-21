"""Atomic single-file publication for provider and compatibility caches."""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Publish only a complete flushed file; remove an interrupted staging file."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(directory)
            except OSError:
                pass
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
