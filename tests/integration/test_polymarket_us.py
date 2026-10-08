"""Live smoke test for the Polymarket US public gateway (no credentials).

API calls: one or two market pages, one settlement, one price history.
"""

import polars as pl
import pytest

from ml4t.data.providers.polymarket_us import PolymarketUSProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def provider():
    provider = PolymarketUSProvider()
    yield provider
    provider.close()


def test_resolved_macro_markets_with_outcomes(provider):
    markets = provider.fetch_markets(closed=True, categories=["macro"], max_pages=1)

    resolved = markets.filter(pl.col("result").is_in(["yes", "no"]))
    assert not resolved.is_empty()

    # FOMC 2026-04-29 held rates; the gateway lists this market's outcome labels reversed.
    held = provider.fetch_markets(slugs=["rdc-usfed-fomc-2026-04-29-maintains"])
    assert held["result"].to_list() == ["yes"]
    assert provider.get_settlement("rdc-usfed-fomc-2026-04-29-maintains") == 1.0

    history = provider.fetch_price_history(resolved["ticker"][0])
    assert not history.is_empty()
    print(resolved.select("ticker", "result", "settlement_value", "winning_outcome").head(5))
    print(f"{len(resolved)} resolved macro markets; {len(history)} price points")
