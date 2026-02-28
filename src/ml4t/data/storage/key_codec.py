"""Storage key encoding helpers.

This module provides a reversible encoding for storage keys so filesystem
names remain unambiguous. It also supports decoding legacy names used before
encoded keys were introduced.
"""

from __future__ import annotations

import base64

ENCODED_PREFIX = "k__"


def encode_storage_key(key: str) -> str:
    """Encode a storage key to a filesystem-safe directory/file name."""
    raw = key.encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{ENCODED_PREFIX}{encoded}"


def is_encoded_storage_key(value: str) -> bool:
    """Return True if the value uses the encoded key format."""
    return value.startswith(ENCODED_PREFIX)


def decode_storage_key(value: str) -> str:
    """Decode either encoded key names or legacy slash-to-underscore names."""
    if not is_encoded_storage_key(value):
        return value.replace("_", "/")

    payload = value[len(ENCODED_PREFIX) :]
    padding = "=" * (-len(payload) % 4)
    raw = base64.urlsafe_b64decode(payload + padding)
    return raw.decode("utf-8")

