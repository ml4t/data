"""Integration tests for FXMacroData provider.

These tests make live API calls and require FXMACRODATA_API_KEY. They are
skipped by default when the key is absent.
"""

from __future__ import annotations

import os

import polars as pl
import pytest

from ml4t.data.providers.fxmacrodata import FXMacroDataProvider

FXMACRODATA_API_KEY = os.getenv("FXMACRODATA_API_KEY")

pytestmark = pytest.mark.skipif(
    not FXMACRODATA_API_KEY,
    reason="FXMACRODATA_API_KEY not set",
)


@pytest.fixture
def provider():
    provider = FXMacroDataProvider(api_key=FXMACRODATA_API_KEY, rate_limit=(1000, 1.0))
    yield provider
    provider.close()


def test_fetch_usd_announcements_with_metadata(provider):
    frame, metadata = provider.fetch_announcements(
        "usd",
        "inflation",
        limit=2,
        include_metadata=True,
    )

    assert isinstance(frame, pl.DataFrame)
    assert len(frame) >= 1
    assert "announcement_datetime" in frame.columns
    assert "data_quality" in metadata


def test_fetch_catalogue_forex_and_cot(provider):
    catalogue = provider.fetch_catalogue("usd")
    eurusd = provider.fetch_forex("eur", "usd", limit=2)
    cot = provider.fetch_cot("eur", limit=2)

    assert len(catalogue) >= 1
    assert "indicator" in catalogue.columns
    assert len(eurusd) >= 1
    assert len(cot) >= 1
