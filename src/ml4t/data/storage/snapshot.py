"""Portable, verified copies of explicitly declared input tables."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import polars as pl
from filelock import FileLock

from ml4t.data.sessions import SessionAssigner
from ml4t.data.storage.backend import StorageBackend
from ml4t.data.storage.keys import validate_path_component
from ml4t.data.utils.atomic import atomic_write
from ml4t.specs import UNSET_CALENDAR, ArtifactSpec, CalendarUnset, FeedSpec

_DECLARATIONS = frozenset(
    {
        "source_revision",
        "identity_namespace",
        "units",
        "as_of",
        "price_basis",
        "share_basis",
        "revisions",
    }
)
_PRIVATE_SUFFIXES = (
    "apikey",
    "password",
    "secret",
    "token",
    "credential",
    "credentials",
    "authorization",
    "privatekey",
    "signingkey",
    "accesskeyid",
)


@dataclass(frozen=True)
class SnapshotTable:
    """One frame or storage key, its role, and explicit source declarations."""

    source: pl.DataFrame | pl.LazyFrame | StorageBackend
    role: str
    metadata: Mapping[str, Any]
    key: str | None = None
    feed_spec: FeedSpec | Mapping[str, Any] | None = None
    artifact: ArtifactSpec | None = None


@dataclass(frozen=True)
class InputSnapshot:
    """Eager tables returned only after every declared file has been verified."""

    snapshot_id: str
    tables: dict[str, pl.DataFrame]
    manifest: dict[str, Any]


def _portable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Snapshot declarations must contain finite JSON values")
        return value
    if isinstance(value, str):
        if value.startswith(("/", "~", "file://")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ValueError("Snapshot declarations must not contain machine paths")
        if "://" in value:
            url = urlsplit(value)
            if (
                url.username
                or url.password
                or any(
                    re.sub(r"[^a-z]", "", name.lower()).endswith(_PRIVATE_SUFFIXES)
                    for name, _ in parse_qsl(url.query)
                )
            ):
                raise ValueError("Snapshot declarations must not contain credentials")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Snapshot declaration keys must be strings")
            if re.sub(r"[^a-z]", "", key.lower()).endswith(_PRIVATE_SUFFIXES):
                raise ValueError("Snapshot declarations must not contain credentials")
            result[key] = _portable(item)
        return result
    if isinstance(value, list | tuple):
        return [_portable(item) for item in value]
    raise TypeError(f"Snapshot declarations require JSON values, not {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _member(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid snapshot member path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or relative != "/".join(parts):
        raise ValueError("Snapshot member paths must be canonical relative paths")
    for part in parts:
        validate_path_component(part)
    path = root
    if path.is_symlink():
        raise ValueError("Snapshot symlinks are forbidden")
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Snapshot symlinks are forbidden")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Snapshot member escapes its directory")
    return path


def _schema(frame: pl.DataFrame) -> list[list[str]]:
    return [[name, str(dtype)] for name, dtype in frame.schema.items()]


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Snapshot metadata must be a mapping")
    result = _portable(value)
    missing = _DECLARATIONS - result.keys()
    if missing:
        raise ValueError(f"Missing required snapshot declarations: {sorted(missing)}")
    if not isinstance(result["source_revision"], str) or not result["source_revision"].strip():
        raise ValueError("source_revision must identify the captured source")
    if not isinstance(result["units"], dict) or not isinstance(result["revisions"], dict):
        raise TypeError("units and revisions must be mappings")
    for name in ("identity_namespace", "as_of", "price_basis", "share_basis"):
        if result[name] is not None and (
            not isinstance(result[name], str) or not result[name].strip()
        ):
            raise ValueError(f"{name} must be a non-empty string or explicit null")
    return result


def _clock(
    feed_spec: FeedSpec | Mapping[str, Any] | None,
    metadata: dict[str, Any],
    calendar: str | None | CalendarUnset,
    columns: Sequence[str],
) -> dict[str, Any] | None:
    if (
        feed_spec is None
        and not any(name in metadata for name in ("timestamp_col", "feed_spec"))
        and "timestamp" not in columns
    ):
        return None
    declaration = feed_spec if feed_spec is not None else metadata.get("feed_spec", metadata)
    if isinstance(declaration, Mapping):
        declaration = {"time_contract_version": 2, **declaration}
    feed = FeedSpec.from_any(declaration)
    assigner = SessionAssigner(feed_spec=feed, calendar=calendar)
    clock = {
        "feed_spec": feed.to_dict(),
        "resolved_calendar": assigner.calendar_name,
        "schedule_provider": "pandas-market-calendars" if assigner.calendar is not None else None,
        "schedule_provider_version": version("pandas-market-calendars")
        if assigner.calendar is not None
        else None,
    }
    if calendar is not UNSET_CALENDAR:
        clock["calendar_override"] = calendar
    return clock


def _capture(
    source: SnapshotTable, destination: Path, calendar: str | None | CalendarUnset
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not isinstance(source.role, str) or not source.role.strip():
        raise ValueError("Every snapshot table needs a role")
    destination.mkdir()
    capture = None
    if isinstance(source.source, StorageBackend):
        if source.key is None:
            raise ValueError("A storage snapshot table requires its logical key")
        backend = source.source
        with backend._key_lock(source.key):
            commit = backend._current_commit(source.key)
            metadata = _metadata({**commit.metadata.get("custom", {}), **source.metadata})
            declared = commit.metadata.get("partitions")
            if declared is None:
                declared = (
                    ["data.parquet"]
                    if "file_path" in commit.metadata
                    else [
                        path.relative_to(commit.generation_path).as_posix()
                        for path in sorted(commit.generation_path.rglob("*.parquet"))
                    ]
                )
            if not declared or len(set(declared)) != len(declared):
                raise ValueError("A captured storage generation contains no Parquet files")
            relative_files = []
            for relative in declared:
                path = _member(commit.generation_path, relative)
                if path.is_dir():
                    relative += "/data.parquet"
                    path = _member(commit.generation_path, relative)
                target = _member(destination, relative)
                target.parent.mkdir(parents=True, exist_ok=True)

                def copy_file(temporary: Path, *, source_path: Path = path) -> None:
                    shutil.copyfile(source_path, temporary)

                atomic_write(target, copy_file)
                relative_files.append(relative)
            capture = {
                "backend": type(backend).__name__,
                "key": source.key,
                "commit": commit.commit_id,
                "generation": commit.generation_id,
            }
    else:
        if source.key is not None:
            raise ValueError("key only applies to a storage-backed table")
        metadata = _metadata(source.metadata)
        frame = (
            source.source.collect() if isinstance(source.source, pl.LazyFrame) else source.source
        )
        atomic_write(destination / "data.parquet", frame.write_parquet)
        relative_files = ["data.parquet"]
    frame = pl.concat([pl.read_parquet(destination / name) for name in relative_files])
    clock = _clock(source.feed_spec, metadata, calendar, frame.columns)
    if clock is not None and clock["feed_spec"]["timestamp_col"] not in frame.columns:
        raise ValueError("The declared snapshot timestamp column is missing")
    files = [
        {
            "path": ("tables/" + destination.name + "/" + name),
            "sha256": _digest((destination / name).read_bytes()),
        }
        for name in relative_files
    ]
    artifact = source.artifact.to_dict() if source.artifact is not None else None
    if artifact is not None:
        artifact["storage"]["path"] = "tables/" + destination.name
        artifact = _portable(artifact)
    return {
        "role": source.role,
        "metadata": metadata,
        "clock": clock,
        "artifact": artifact,
        "schema": _schema(frame),
        "row_count": len(frame),
        "files": files,
    }, capture


def export_snapshot(
    path: str | Path,
    tables: Mapping[str, SnapshotTable],
    *,
    effective_config: Mapping[str, Any],
    artifacts: Mapping[str, str | None],
    required_tables: Sequence[str] | None = None,
    unavailable: Mapping[str, str] | None = None,
    calendar: str | None | CalendarUnset = UNSET_CALENDAR,
) -> InputSnapshot:
    """Copy an immutable input set; existing different snapshots are never overwritten.

    Per-table declarations require source_revision, identity_namespace, units, as_of,
    price_basis, share_basis and revisions. Explicit null retains an unknown optional fact.
    Consumer policies and artifact hashes come from the caller, never from guessed defaults.
    """
    path = Path(path)
    validate_path_component(path.name)
    if (
        not isinstance(tables, Mapping)
        or not isinstance(effective_config, Mapping)
        or not isinstance(artifacts, Mapping)
    ):
        raise TypeError("tables, effective_config and artifacts must be mappings")
    if isinstance(required_tables, str | bytes):
        raise TypeError("required_tables must be a sequence of table names")
    if not tables:
        raise ValueError("A snapshot requires at least one table")
    for name in tables:
        validate_path_component(name)
    required = sorted(tables if required_tables is None else required_tables)
    if len(set(required)) != len(required) or any(name not in tables for name in required):
        raise ValueError("Required snapshot tables must be distinct captured table names")
    configuration = _portable(effective_config)
    artifacts = _portable(artifacts)
    for fingerprint in artifacts.values():
        if fingerprint is not None and (
            not isinstance(fingerprint, str) or re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None
        ):
            raise ValueError("Artifact hashes must be lowercase SHA256 or explicit null")
    missing = _portable(unavailable or {})
    if set(missing) & set(tables) or any(
        not isinstance(reason, str) or not reason for reason in missing.values()
    ):
        raise ValueError("Unavailable tables need a reason and cannot also be captured")
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(path.parent / f".{path.name}.snapshot.lock")),
        tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.staging-") as name,
    ):
        staging = Path(name)
        (staging / "tables").mkdir()
        entries, captures = {}, {}
        for name, source in sorted(tables.items()):
            entries[name], capture = _capture(source, staging / "tables" / name, calendar)
            if capture is not None:
                captures[name] = capture
        content = {
            "format_version": 1,
            "tables": entries,
            "required_tables": required,
            "unavailable": missing,
            "effective_config": configuration,
            "artifacts": artifacts,
            "software": {
                "ml4t-data": version("ml4t-data"),
                "ml4t-specs": version("ml4t-specs"),
                "python": sys.version.split()[0],
                "polars": pl.__version__,
            },
        }
        manifest = {
            "snapshot_id": _digest(_canonical(content)),
            "content": content,
            "captured_at": datetime.now(UTC).isoformat(),
            "captures": captures,
        }
        manifest["manifest_sha256"] = _digest(_canonical(manifest))

        def write_manifest(temporary: Path) -> None:
            temporary.write_bytes(_canonical(manifest))

        atomic_write(staging / "manifest.json", write_manifest)
        result = read_snapshot(staging)
        directories = [staging, *(path for path in staging.rglob("*") if path.is_dir())]
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            StorageBackend._fsync_directory(directory)
        if path.exists() or path.is_symlink():
            existing = read_snapshot(path)
            if existing.snapshot_id != result.snapshot_id:
                raise FileExistsError("Snapshot paths are immutable; use a new destination")
            return existing
        staging.replace(path)
        StorageBackend._fsync_directory(path.parent)
        return result


def read_snapshot(path: str | Path) -> InputSnapshot:
    """Validate manifest, every file hash and schema before returning any input tables."""
    root = Path(path)
    manifest = json.loads(_member(root, "manifest.json").read_bytes())
    checksum = manifest.get("manifest_sha256")
    if checksum != _digest(
        _canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    ):
        raise ValueError("Snapshot manifest checksum mismatch")
    content = manifest["content"]
    if type(content.get("format_version")) is not int or content["format_version"] != 1:
        raise ValueError("Unsupported snapshot format version")
    if manifest.get("snapshot_id") != _digest(_canonical(content)):
        raise ValueError("Snapshot manifest identity mismatch")
    _portable(manifest)
    entries = content["tables"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Snapshot table declarations are missing")
    if any(name not in entries for name in content["required_tables"]):
        raise ValueError("A required snapshot table is missing")
    buffers, seen = {}, set()
    for name, entry in entries.items():
        validate_path_component(name)
        _metadata(entry["metadata"])
        clock = entry["clock"]
        if clock is not None:
            feed = FeedSpec.from_mapping(clock["feed_spec"])
            if (
                feed.resolve_calendar(calendar=clock.get("calendar_override", UNSET_CALENDAR))
                != clock["resolved_calendar"]
            ):
                raise ValueError(
                    "Snapshot calendar resolution differs from its captured declaration"
                )
        if not entry["files"]:
            raise ValueError("A snapshot table has no files")
        buffers[name] = []
        for file in entry["files"]:
            relative = file["path"]
            if not relative.startswith(f"tables/{name}/") or relative in seen:
                raise ValueError("Snapshot members must belong to exactly one table")
            seen.add(relative)
            data = _member(root, relative).read_bytes()
            if _digest(data) != file["sha256"]:
                raise ValueError(f"Snapshot file hash mismatch: {relative}")
            buffers[name].append(data)
    tables = {}
    for name, members in buffers.items():
        frame = pl.concat([pl.read_parquet(io.BytesIO(data)) for data in members])
        if _schema(frame) != entries[name]["schema"] or len(frame) != entries[name]["row_count"]:
            raise ValueError(f"Snapshot table schema/count mismatch: {name}")
        tables[name] = frame
    return InputSnapshot(manifest["snapshot_id"], tables, manifest)
