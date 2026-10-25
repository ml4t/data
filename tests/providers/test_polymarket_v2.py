"""Data API v2 regressions using responses captured from the public service."""

import copy
import json
from pathlib import Path

import httpx
import pytest

from ml4t.data.core.exceptions import DataValidationError
from ml4t.data.providers.polymarket import POLYMARKET_TRADE_SCHEMA, PolymarketProvider

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/polymarket/data_api_v2_trades.json").read_text()
)
CONDITION = CAPTURE["condition"]


@pytest.fixture
def provider():
    with PolymarketProvider(rate_limit=(10000, 1)) as provider:
        yield provider


def test_v2_captured_pages_preserve_trades_and_cursor_filters(provider, use_transport):
    """Prevent retired v1 requests, dropped short pages, and lost token/wallet identities."""
    pages = copy.deepcopy(CAPTURE["pages"])
    pages[-1]["pagination"].update(has_more=False, next_cursor=None)
    requests = []

    def serve(request):
        requests.append(request)
        if request.url.path != "/v2/trades" or "market" in request.url.params:
            return httpx.Response(400, json={"error": "v1 is retired; use condition and cursor"})
        assert "offset" not in request.url.params
        assert request.url.params["condition"] == CONDITION
        assert request.url.params["start"] == "1"
        assert request.url.params["end"] == "1792786373"
        index = len(requests) - 1
        assert request.url.params.get("cursor") == (
            None if index == 0 else pages[index - 1]["pagination"]["next_cursor"]
        )
        return httpx.Response(200, json=pages[index])

    use_transport(provider, serve)
    result = provider.fetch_trades(CONDITION, start=1, end=1792786373, page_size=10000)
    rows = [row for page in pages for row in page["data"]]
    assert result.height == len(rows) == 12
    assert result.schema == POLYMARKET_TRADE_SCHEMA
    assert set(result["transaction_hash"]) == {row["transaction_hash"] for row in rows}
    assert set(result["asset"]) == {row["token_id"] for row in rows}
    assert set(result["proxy_wallet"]) == {row["proxy_wallet"] for row in rows}
    assert result["count"].sum() == pytest.approx(sum(row["size"] for row in rows))
    assert result["timestamp"].is_sorted()
    assert len(requests) == 3
    assert requests[0].url.params["limit"] == "1000"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"data": [], "pagination": {"has_more": True, "next_cursor": None}},
        {"data": [], "pagination": {"has_more": False, "next_cursor": "stale"}},
        {"data": [], "pagination": {"has_more": "false", "next_cursor": None}},
        {"data": [], "pagination": {}},
    ],
)
def test_invalid_v2_envelope_does_not_silently_truncate(provider, use_transport, payload):
    use_transport(provider, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(DataValidationError):
        provider.fetch_trades(CONDITION)


def test_empty_page_with_cursor_does_not_end_walk(provider, use_transport):
    pages = [
        {"data": [], "pagination": {"has_more": True, "next_cursor": "continue"}},
        {
            "data": CAPTURE["pages"][0]["data"],
            "pagination": {"has_more": False, "next_cursor": None},
        },
    ]
    requests = use_transport(provider, lambda _: httpx.Response(200, json=pages.pop(0)))
    assert provider.fetch_trades(CONDITION).height == 4
    assert len(requests) == 2


def test_repeated_cursor_fails_instead_of_requesting_forever(provider, use_transport):
    payload = copy.deepcopy(CAPTURE["pages"][0])
    use_transport(provider, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(DataValidationError, match="cursor repeated"):
        provider.fetch_trades(CONDITION)


def test_max_pages_stops_after_one_request(provider, use_transport):
    requests = use_transport(provider, lambda _: httpx.Response(200, json=CAPTURE["pages"][0]))
    assert provider.fetch_trades(CONDITION, max_pages=1).height == 4
    assert len(requests) == 1


def test_trade_ids_remain_stable_across_v1_v2_casing(provider):
    row = CAPTURE["pages"][0]["data"][0]
    v1 = dict(row)
    for old, new in (
        ("transactionHash", "transaction_hash"),
        ("asset", "token_id"),
        ("proxyWallet", "proxy_wallet"),
    ):
        v1[old] = v1.pop(new)
    assert provider._trade_key(row) == provider._trade_key(v1)


def test_v2_whole_quantities_preserve_published_v1_trade_ids(provider, use_transport):
    """Prevent re-downloaded trades becoming duplicates when v2 encodes 38 as 38.0."""
    pages = copy.deepcopy(CAPTURE["pages"])
    pages[-1]["pagination"].update(has_more=False, next_cursor=None)
    use_transport(provider, lambda _: httpx.Response(200, json=pages.pop(0)))

    trades = provider.fetch_trades(CONDITION, page_size=4)

    assert (
        dict(zip(trades["transaction_hash"], trades["trade_id"]))
        == CAPTURE["legacy_trade_ids_by_transaction"]
    )
