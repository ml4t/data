"""Tests for canonical storage-key encoding and path containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml4t.data.storage.keys import (
    MAX_KEY_BYTES,
    decode_storage_key,
    encode_storage_key,
    storage_key_path,
    validate_path_component,
    validate_storage_key,
)


@pytest.mark.parametrize(
    "key",
    [
        "a/b_c",
        "a_b/c",
        "equities/tick/BRK_B",
        "futures/daily/ES.v.0",
        "indices/daily/日経225",
        "x" * MAX_KEY_BYTES,
    ],
)
def test_storage_key_encoding_round_trips(key: str) -> None:
    encoded = encode_storage_key(key)

    assert "/" not in encoded
    assert "\\" not in encoded
    assert decode_storage_key(encoded) == key


def test_distinct_keys_have_distinct_encodings() -> None:
    assert encode_storage_key("a/b_c") != encode_storage_key("a_b/c")


@pytest.mark.parametrize(
    "key",
    ["", "/absolute", "trailing/", "a//b", "a/../b", "a/./b", "a\\..\\b", "a/\x00/b"],
)
def test_storage_key_rejects_ambiguous_or_unsafe_values(key: str) -> None:
    with pytest.raises(ValueError):
        validate_storage_key(key)


def test_storage_key_rejects_values_above_maximum() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        validate_storage_key("x" * (MAX_KEY_BYTES + 1))


@pytest.mark.parametrize(
    "component",
    ["../backup", "..\\backup", "/absolute", "C:\\absolute", "NUL", "name.", "name ", "a:b"],
)
def test_human_readable_path_component_is_portable(component: str) -> None:
    with pytest.raises(ValueError):
        validate_path_component(component)


def test_storage_key_path_rejects_existing_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    encoded = encode_storage_key("equities/daily/AAPL")
    try:
        (root / encoded).symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(ValueError, match="escapes configured root"):
        storage_key_path(root, "equities/daily/AAPL")
