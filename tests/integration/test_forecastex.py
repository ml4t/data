"""Live smoke test for the ForecastEx daily files (public S3 bucket, no credentials).

API calls: four CSV downloads.
"""

import polars as pl
import pytest

from ml4t.data.providers.forecastex import ForecastExProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def provider():
    provider = ForecastExProvider()
    yield provider
    provider.close()


def test_resolved_fed_decision_contracts(provider):
    markets = provider.fetch_markets("2026-09-15", "2026-09-16", product="FFDEC")

    resolved = markets.filter(pl.col("status") == "resolved")
    assert not resolved.is_empty()
    by_ticker = {row["ticker"]: row for row in resolved.iter_rows(named=True)}
    # FOMC 2026-09-16 raised rates by 25 bp.
    assert by_ticker["FFDEC_091626_E25"]["result"] == "yes"
    assert by_ticker["FFDEC_091626_E0"]["result"] == "no"

    trades = provider.fetch_trades("2026-09-16", "2026-09-16", product="FFDEC")
    assert not trades.is_empty()
    print(resolved.select("ticker", "result", "settlement_value", "volume"))
    print(f"{len(trades)} FFDEC pairs on 2026-09-16")
