from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ML4T_DATA_ENV_VAR = "ML4T_DATA_PATH"
ML4T_DATA_CONFIG_ENV_VAR = "ML4T_DATA_CONFIG"
ML4T_DATA_HOME_CONFIG_PATH = Path.home() / ".config" / "ml4t-data" / "config.yaml"


def expand_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    resolved = Path(path).expanduser()
    if base_dir is not None and not resolved.is_absolute():
        resolved = Path(base_dir).expanduser() / resolved
    return resolved


def get_ml4t_data_config_search_paths(cwd: str | Path | None = None) -> list[Path]:
    search_root = Path(cwd) if cwd is not None else Path.cwd()
    return [
        search_root / "ml4t.data.yaml",
        search_root / "ml4t.data.yml",
        search_root / ".ml4t-data.yaml",
        search_root / ".ml4t-data.yml",
        search_root / "config" / "ml4t.data.yaml",
        search_root / "config" / "ml4t.data.yml",
        ML4T_DATA_HOME_CONFIG_PATH,
    ]


def find_ml4t_data_config_path(cwd: str | Path | None = None) -> Path | None:
    explicit_path = os.getenv(ML4T_DATA_CONFIG_ENV_VAR)
    if explicit_path:
        return expand_path(explicit_path)

    for path in get_ml4t_data_config_search_paths(cwd):
        if path.exists():
            return path
    return None


def _get_nested_value(config: Mapping[str, Any] | Any, keys: tuple[str, ...]) -> Any:
    current: Mapping[str, Any] | Any = config
    for key in keys:
        current = current.get(key) if isinstance(current, Mapping) else getattr(current, key, None)
        if current is None:
            return None
    return current


def get_ml4t_data_root(
    config: Mapping[str, Any] | Any | None = None,
    *,
    config_dir: str | Path | None = None,
) -> Path | None:
    if config is not None:
        for keys in (("storage", "base_path"), ("base_dir",), ("data_root",)):
            value = _get_nested_value(config, keys)
            if value:
                return expand_path(value, base_dir=config_dir)

    value = os.getenv(ML4T_DATA_ENV_VAR)
    if value:
        return expand_path(value)

    return None


def default_ml4t_data_path(relative_path: str | Path = ".") -> Path:
    root = Path.cwd() / "data"
    relative = Path(relative_path)
    if str(relative) in {"", "."}:
        return root
    return root / relative


def resolve_ml4t_data_path(
    relative_path: str | Path,
    default_path: str | Path,
    *,
    configured_path: str | Path | None = None,
    config: Mapping[str, Any] | Any | None = None,
    config_dir: str | Path | None = None,
) -> Path:
    if configured_path is not None:
        return expand_path(configured_path, base_dir=config_dir)

    root = get_ml4t_data_root(config, config_dir=config_dir)
    if root is not None:
        return root / Path(relative_path)
    return expand_path(default_path)


def ml4t_data_path_candidates(
    relative_paths: Iterable[str | Path],
    default_paths: Iterable[str | Path],
    *,
    config: Mapping[str, Any] | Any | None = None,
    config_dir: str | Path | None = None,
) -> list[Path]:
    root = get_ml4t_data_root(config, config_dir=config_dir)
    if root is not None:
        return [root / Path(relative_path) for relative_path in relative_paths]
    return [expand_path(path) for path in default_paths]
