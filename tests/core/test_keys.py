"""Tests for canonical storage key helpers."""

from __future__ import annotations

import pytest

from ml4t.data.core.keys import (
    build_storage_key,
    is_storage_key,
    normalize_storage_key,
    parse_storage_key,
)


def test_build_storage_key_defaults() -> None:
    assert build_storage_key("AAPL") == "equities/daily/AAPL"


def test_build_storage_key_custom_parts() -> None:
    assert build_storage_key("BTC", asset_class="crypto", frequency="hourly") == "crypto/hourly/BTC"


def test_build_storage_key_rejects_empty_parts() -> None:
    with pytest.raises(ValueError, match="symbol cannot be empty"):
        build_storage_key("")


def test_is_storage_key() -> None:
    assert is_storage_key("equities/daily/AAPL") is True
    assert is_storage_key("AAPL") is False
    assert is_storage_key("equities/daily/") is False


def test_parse_storage_key() -> None:
    assert parse_storage_key("equities/daily/AAPL") == ("equities", "daily", "AAPL")


def test_parse_storage_key_rejects_invalid_format() -> None:
    with pytest.raises(ValueError, match="invalid storage key format"):
        parse_storage_key("AAPL")


def test_normalize_storage_key_from_symbol() -> None:
    assert normalize_storage_key("AAPL", asset_class="equities", frequency="daily") == (
        "equities/daily/AAPL"
    )


def test_normalize_storage_key_from_key() -> None:
    assert normalize_storage_key("crypto/hourly/BTC") == "crypto/hourly/BTC"
