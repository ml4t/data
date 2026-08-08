"""Opt-in free live contract test for the Databento provider.

The test process does not load ``.env`` files. Export both required variables explicitly.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("databento")

from ml4t.data.providers.databento import DataBentoProvider


@pytest.fixture
def provider() -> DataBentoProvider:
    """Create a provider only after an explicit live-test opt in."""
    if os.getenv("ML4T_RUN_DATABENTO_LIVE") != "1":
        pytest.skip("export ML4T_RUN_DATABENTO_LIVE=1 to authorize the live contract request")

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        pytest.skip("export DATABENTO_API_KEY for the Databento live contract test")
    return DataBentoProvider(api_key=api_key)


@pytest.mark.integration
def test_databento_metadata_contract(provider: DataBentoProvider) -> None:
    """Authenticate and discover a dataset and its schemas without metered data."""
    datasets = provider.get_available_datasets()
    schemas = provider.get_available_schemas("GLBX.MDP3")

    assert "GLBX.MDP3" in datasets
    assert "ohlcv-1d" in schemas
