"""Tests for storage key codec helpers."""

from ml4t.data.storage.key_codec import (
    decode_storage_key,
    encode_storage_key,
    is_encoded_storage_key,
)


def test_encode_decode_roundtrip() -> None:
    """Encoding should be reversible for storage keys."""
    key = "equities/daily/BRK_B"
    encoded = encode_storage_key(key)

    assert is_encoded_storage_key(encoded)
    assert decode_storage_key(encoded) == key


def test_decode_legacy_underscore_name() -> None:
    """Legacy underscore names should still decode to slash-separated keys."""
    assert decode_storage_key("provider_symbol") == "provider/symbol"


def test_encoded_prefix() -> None:
    """Encoded values should be prefixed for easy detection."""
    assert encode_storage_key("x/y").startswith("k__")
