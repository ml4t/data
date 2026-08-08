"""Canonical logical-key and filesystem-component handling."""

from __future__ import annotations

import base64
import re
from pathlib import Path, PureWindowsPath

KEY_ENCODING_PREFIX = "k1_"
MAX_KEY_BYTES = 160

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_ENCODED_KEY_PATTERN = re.compile(rf"^{KEY_ENCODING_PREFIX}[A-Za-z0-9_-]+$")


def validate_storage_key(key: str) -> str:
    """Validate a logical storage key and return it unchanged."""
    if not isinstance(key, str) or not key:
        raise ValueError("Storage key must be a non-empty string")
    if len(key.encode("utf-8")) > MAX_KEY_BYTES:
        raise ValueError(f"Storage key exceeds the {MAX_KEY_BYTES}-byte limit")
    if key.startswith("/") or key.endswith("/"):
        raise ValueError("Storage key cannot start or end with a separator")
    if "\\" in key:
        raise ValueError("Storage key cannot contain a backslash")
    if any(ord(character) < 32 or ord(character) == 127 for character in key):
        raise ValueError("Storage key cannot contain control characters")

    components = key.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("Storage key contains an empty or relative component")
    return key


def encode_storage_key(key: str) -> str:
    """Encode a logical storage key as one portable filename component."""
    validated = validate_storage_key(key)
    encoded = base64.urlsafe_b64encode(validated.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{KEY_ENCODING_PREFIX}{encoded}"


def decode_storage_key(encoded_key: str) -> str:
    """Decode and validate a canonical storage filename component."""
    if not _ENCODED_KEY_PATTERN.fullmatch(encoded_key):
        raise ValueError(f"Unsupported storage key encoding: {encoded_key}")
    payload = encoded_key.removeprefix(KEY_ENCODING_PREFIX)
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.b64decode(payload + padding, altchars=b"-_", validate=True).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Invalid storage key encoding: {encoded_key}") from error
    validate_storage_key(decoded)
    if encode_storage_key(decoded) != encoded_key:
        raise ValueError(f"Non-canonical storage key encoding: {encoded_key}")
    return decoded


def storage_key_path(root: Path, key: str, suffix: str = "") -> Path:
    """Return the contained physical path for a logical storage key."""
    return contained_path(root, f"{encode_storage_key(key)}{suffix}")


def contained_path(root: Path, *components: str) -> Path:
    """Join trusted filename components and reject symlink escapes."""
    resolved_root = Path(root).resolve()
    candidate = resolved_root.joinpath(*components)
    if not candidate.resolve().is_relative_to(resolved_root):
        raise ValueError(f"Resolved path escapes configured root: {candidate}")
    return candidate


def validate_path_component(component: str, name: str = "path component") -> str:
    """Validate a user-controlled component that will remain human-readable."""
    if not isinstance(component, str) or not component:
        raise ValueError(f"{name} must be a non-empty string")
    windows_path = PureWindowsPath(component)
    if Path(component).is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"{name} cannot be absolute")
    if component in {".", ".."} or "/" in component or "\\" in component:
        raise ValueError(f"{name} cannot contain path separators or relative components")
    if ":" in component:
        raise ValueError(f"{name} cannot contain a Windows stream separator")
    if component.endswith((".", " ")):
        raise ValueError(f"{name} cannot end with a dot or space")
    if any(ord(character) < 32 or ord(character) == 127 for character in component):
        raise ValueError(f"{name} cannot contain control characters")
    if len(component.encode("utf-8")) > 200:
        raise ValueError(f"{name} exceeds the 200-byte limit")
    if component.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{name} uses a reserved Windows filename")
    return component


def encode_path_component(component: str, name: str = "path component") -> str:
    """Validate and encode one user-controlled directory component."""
    if not isinstance(component, str) or not component:
        raise ValueError(f"{name} must be a non-empty string")
    if component in {".", ".."} or "/" in component or "\\" in component:
        raise ValueError(f"{name} cannot contain path separators or relative components")
    if any(ord(character) < 32 or ord(character) == 127 for character in component):
        raise ValueError(f"{name} cannot contain control characters")
    if len(component.encode("utf-8")) > MAX_KEY_BYTES:
        raise ValueError(f"{name} exceeds the {MAX_KEY_BYTES}-byte limit")
    return encode_storage_key(component)
