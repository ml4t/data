"""Offline tests for the ForecastEx daily-file provider.

CSV fixtures copy the headers and value formats of the public files probed on 2026-10-08
(``prices/daily_prices_YYYYMMDD.csv``, ``pairs/pairs_YYYYMMDD.csv``,
``daily_summary/summary_YYYYMMDD.csv``). A missing S3 key answers 404 with an XML body.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import polars as pl
import pytest

from ml4t.data.core.exceptions import DataValidationError, NetworkError, RateLimitError
from ml4t.data.providers.forecastex import (
    FORECASTEX_TRADE_SCHEMA,
    MARKET_SCHEMA,
    PRICES_SCHEMA,
    PRODUCTS_SCHEMA,
    ForecastExProvider,
)

PRICES_HEADER = (
    "event_contract,subtype,expiration_date,date,start_price,high_price,low_price,"
    "end_price,settlement_price,pair_quantity,open_interest,vwap"
)
PAIRS_HEADER = "pair_id,event_contract,expiration_date,quantity,yes_price,no_price,pair_time"
FED_EXP = "2026-09-16T13:00:00-05:00"
LATER_EXP = "2026-12-09T13:00:00-06:00"
HOURLY_EXP = "2026-09-16T15:00:00-05:00"
NOT_FOUND = (
    '<?xml version="1.0" encoding="UTF-8"?><Error><Code>NoSuchKey</Code>'
    "<Message>The specified key does not exist.</Message></Error>"
)


def _price_rows(contract: str, exp: str, day: str, yes_settle: str, no_settle: str) -> list[str]:
    return [
        f"{contract},YES,{exp},{day},0.86,0.90,0.82,0.85,{yes_settle},94393,0,0.87",
        f"{contract},NO,{exp},{day},0.14,0.18,0.10,0.15,{no_settle},94393,0,0.13",
    ]


def _csv(header: str, rows: list[str]) -> str:
    return "\n".join([header, *rows]) + "\n"


PRICES = {
    "20260915": _csv(
        PRICES_HEADER,
        [
            *_price_rows("FFDEC_091626_E25", FED_EXP, "2026-09-15", "0.86", "0.14"),
            *_price_rows("FFDEC_091626_E0", FED_EXP, "2026-09-15", "0.13", "0.87"),
            # Delisted after 09-15 with a final payout of 0.00.
            *_price_rows("FFDEC_091626_H50", FED_EXP, "2026-09-15", "0.00", "1.00"),
            *_price_rows("FFDEC_120926_E0", LATER_EXP, "2026-09-15", "0.40", "0.60"),
        ],
    ),
    "20260916": _csv(
        PRICES_HEADER,
        [
            *_price_rows("FFDEC_091626_E25", FED_EXP, "2026-09-16", "1.00", "0.00"),
            *_price_rows("FFDEC_091626_E0", FED_EXP, "2026-09-16", "0.00", "1.00"),
            # Intraday contract: drops out with its daily mark, not its payout.
            *_price_rows("HRULAS_09162615_89", HOURLY_EXP, "2026-09-16", "0.90", "0.10"),
            *_price_rows("FFDEC_120926_E0", LATER_EXP, "2026-09-16", "0.41", "0.59"),
            '"FEDRO_1128_Senate-R,House-R",YES,2029-01-08T15:00:00-06:00,2026-09-16,'
            "0.00,0.00,0.00,0.00,,0,0,0.00",
        ],
    ),
    "20260917": _csv(
        PRICES_HEADER,
        [
            *_price_rows("FFDEC_120926_E0", LATER_EXP, "2026-09-17", "0.42", "0.58"),
            '"FEDRO_1128_Senate-R,House-R",YES,2029-01-08T15:00:00-06:00,2026-09-17,'
            "0.00,0.00,0.00,0.00,,0,0,0.00",
        ],
    ),
}

PAIRS = {
    "20260916": _csv(
        PAIRS_HEADER,
        [
            "CGXVN0KXM07Q,FFDEC_091626_E0,2026-09-16T13:00:00-05:00,25,0.12,0.88,"
            "2026-09-15T16:18:18.905911443-05:00",
            "CGXTRZKZY07Q,YXHBT_123126_100000,2026-12-31T23:59:00-06:00,11,0.19,0.81,"
            "2026-09-15T16:15:26.12331110-05:00",
            "CGXVN0KXM07Q,FFDEC_091626_E0,2026-09-16T13:00:00-05:00,25,0.12,0.88,"
            "2026-09-15T16:18:18.905911443-05:00",
        ],
    ),
    "20260917": _csv(
        PAIRS_HEADER,
        [
            "CGXX6TSBG07Q,FFDEC_091626_E25,2026-09-16T13:00:00-05:00,100,0.87,0.13,"
            "2026-09-16T09:26:10.944740-05:00",
        ],
    ),
}

SUMMARY = _csv(
    "product_id,product_name,product_category,total_pairs",
    ["FFDEC,Fed Decision,Financial Markets,1200", "AXXME,Maine Senate,Elections,50"],
)


class S3Fake:
    """In-process S3 bucket serving a dict of object keys."""

    def __init__(self, objects: dict[str, str]) -> None:
        self.objects = dict(objects)
        self.overrides: dict[str, list[httpx.Response]] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = request.url.path.lstrip("/")
        queued = self.overrides.get(key)
        if queued:
            return queued.pop(0)
        if key in self.objects:
            return httpx.Response(200, text=self.objects[key])
        return httpx.Response(404, text=NOT_FOUND, headers={"content-type": "application/xml"})


def _objects(prices: dict[str, str] = PRICES, pairs: dict[str, str] = PAIRS) -> dict[str, str]:
    objects = {f"prices/daily_prices_{day}.csv": text for day, text in prices.items()}
    objects.update({f"pairs/pairs_{day}.csv": text for day, text in pairs.items()})
    objects["daily_summary/summary_20260916.csv"] = SUMMARY
    return objects


@pytest.fixture
def provider():
    provider = ForecastExProvider(rate_limit=(10_000, 1.0))
    yield provider
    provider.close()


@pytest.fixture
def bucket(provider, use_transport):
    fake = S3Fake(_objects())
    fake.requests = use_transport(provider, fake)
    return fake


class TestResolutionOutcome:
    def test_expired_contracts_resolve_from_final_settlement_price(self, provider, bucket):
        markets = provider.fetch_markets("2026-09-15", "2026-09-16")

        assert markets.schema == pl.Schema(MARKET_SCHEMA)
        rows = {row["ticker"]: row for row in markets.iter_rows(named=True)}
        assert rows["FFDEC_091626_E25"]["status"] == "resolved"
        assert rows["FFDEC_091626_E25"]["result"] == "yes"
        assert rows["FFDEC_091626_E25"]["settlement_value"] == 1.0
        assert rows["FFDEC_091626_E25"]["settlement_ts"] == datetime(2026, 9, 16, 18, 0, tzinfo=UTC)
        assert rows["FFDEC_091626_E0"]["result"] == "no"
        assert rows["FFDEC_091626_E0"]["settlement_value"] == 0.0
        assert rows["FFDEC_091626_H50"]["result"] == "no"
        assert rows["FFDEC_091626_H50"]["last_date"] == date(2026, 9, 15)
        # Still listed on 09-17: open, whatever its daily mark.
        assert rows["FFDEC_120926_E0"]["status"] == "open"
        assert rows["FFDEC_120926_E0"]["result"] is None
        assert rows["FFDEC_120926_E0"]["settlement_value"] is None
        # Dropped out with a 0.90 mark: expired, but the files do not give the outcome.
        assert rows["HRULAS_09162615_89"]["status"] == "expired"
        assert rows["HRULAS_09162615_89"]["result"] is None
        assert rows["FEDRO_1128_Senate-R,House-R"]["status"] == "open"
        assert bucket.requests[-1].url.path == "/prices/daily_prices_20260917.csv"

    def test_unpublished_next_day_leaves_final_day_contracts_open(self, provider, use_transport):
        prices = {day: text for day, text in PRICES.items() if day != "20260917"}
        fake = S3Fake(_objects(prices=prices))
        use_transport(provider, fake)

        markets = provider.fetch_markets("2026-09-15", "2026-09-16")

        rows = {row["ticker"]: row for row in markets.iter_rows(named=True)}
        assert rows["FFDEC_091626_E25"]["status"] == "open"
        assert rows["FFDEC_091626_E25"]["result"] is None
        # Absent from a later file inside the range: still resolved.
        assert rows["FFDEC_091626_H50"]["result"] == "no"

    def test_product_filter(self, provider, bucket):
        markets = provider.fetch_markets("2026-09-15", "2026-09-16", product="ffdec")

        assert set(markets["product"].to_list()) == {"FFDEC"}
        assert len(markets) == 4


class TestPricesAndTrades:
    def test_prices_are_typed_and_quoted_contracts_survive(self, provider, bucket):
        prices = provider.fetch_prices("2026-09-16", "2026-09-16")

        assert prices.schema == pl.Schema(PRICES_SCHEMA)
        fedro = prices.filter(pl.col("ticker") == "FEDRO_1128_Senate-R,House-R").row(0, named=True)
        assert fedro["product"] == "FEDRO"
        assert fedro["settlement_price"] is None
        e25 = prices.filter(
            (pl.col("ticker") == "FFDEC_091626_E25") & (pl.col("side") == "YES")
        ).row(0, named=True)
        assert e25["expiration"] == datetime(2026, 9, 16, 18, 0, tzinfo=UTC)
        assert (e25["open"], e25["high"], e25["low"], e25["close"]) == (0.86, 0.90, 0.82, 0.85)
        assert e25["volume"] == 94393.0
        assert e25["vwap"] == 0.87

    def test_trades_parse_nanosecond_offsets_and_deduplicate(self, provider, bucket):
        trades = provider.fetch_trades("2026-09-16", "2026-09-17")

        assert trades.schema == pl.Schema(FORECASTEX_TRADE_SCHEMA)
        assert trades["trade_id"].to_list() == ["CGXTRZKZY07Q", "CGXVN0KXM07Q", "CGXX6TSBG07Q"]
        first = trades.row(1, named=True)
        assert first["timestamp"] == datetime(2026, 9, 15, 21, 18, 18, 905911, tzinfo=UTC)
        assert first["price"] == 0.12
        assert first["no_price"] == 0.88
        assert first["count"] == 25.0
        assert first["taker_side"] is None
        assert first["is_block_trade"] is False

    def test_trade_filters(self, provider, bucket):
        by_product = provider.fetch_trades("2026-09-16", "2026-09-17", product="FFDEC")
        by_ticker = provider.fetch_trades("2026-09-16", "2026-09-17", ticker="FFDEC_091626_E25")

        assert set(by_product["product"].to_list()) == {"FFDEC"}
        assert by_ticker["trade_id"].to_list() == ["CGXX6TSBG07Q"]

    def test_products(self, provider, bucket):
        products = provider.fetch_products("2026-09-16")

        assert products.schema == pl.Schema(PRODUCTS_SCHEMA)
        assert products["product"].to_list() == ["FFDEC", "AXXME"]
        assert provider.fetch_products("2024-01-01").is_empty()


class TestEmptyAndFailing:
    def test_missing_files_give_typed_empty_frames(self, provider, bucket):
        assert provider.fetch_prices("2025-01-01", "2025-01-02").schema == pl.Schema(PRICES_SCHEMA)
        assert provider.fetch_trades("2025-01-01", "2025-01-02").is_empty()
        markets = provider.fetch_markets("2025-01-01", "2025-01-02")
        assert markets.is_empty()
        assert markets.schema == pl.Schema(MARKET_SCHEMA)

    def test_dates_before_first_file_make_no_requests(self, provider, bucket):
        assert provider.fetch_prices("2024-01-01", "2024-07-31").is_empty()
        assert bucket.requests == []

    def test_reversed_range_is_rejected(self, provider, bucket):
        with pytest.raises(DataValidationError, match="before start"):
            provider.fetch_prices("2026-09-17", "2026-09-16")

    def test_missing_column_is_rejected(self, provider, use_transport):
        broken = PRICES["20260916"].replace("settlement_price", "settle", 1)
        use_transport(provider, S3Fake(_objects(prices={"20260916": broken})))

        with pytest.raises(DataValidationError, match="lacks columns"):
            provider.fetch_prices("2026-09-16", "2026-09-16")

    def test_unparseable_value_is_rejected(self, provider, use_transport):
        broken = PRICES["20260916"].replace("0.86,0.90", "n/a,0.90", 1)
        use_transport(provider, S3Fake(_objects(prices={"20260916": broken})))

        with pytest.raises(DataValidationError, match="Unparseable"):
            provider.fetch_prices("2026-09-16", "2026-09-16")

    def test_empty_body_is_rejected(self, provider, use_transport):
        use_transport(provider, S3Fake(_objects(prices={"20260916": ""})))

        with pytest.raises(DataValidationError, match="Malformed CSV"):
            provider.fetch_prices("2026-09-16", "2026-09-16")

    def test_throttling_is_retried(self, provider, bucket, retry_sleeps):
        bucket.overrides["prices/daily_prices_20260916.csv"] = [
            httpx.Response(429, text="slow down", headers={"Retry-After": "2"})
        ]

        prices = provider.fetch_prices("2026-09-16", "2026-09-16")

        assert not prices.is_empty()
        retry_sleeps.assert_called_once_with(4.0)

    def test_persistent_throttling_raises(self, provider, bucket):
        bucket.overrides["prices/daily_prices_20260916.csv"] = [
            httpx.Response(429) for _ in range(3)
        ]

        with pytest.raises(RateLimitError):
            provider.fetch_prices("2026-09-16", "2026-09-16")

    def test_server_error_is_retried_then_raised(self, provider, bucket):
        bucket.overrides["pairs/pairs_20260916.csv"] = [httpx.Response(503) for _ in range(3)]

        with pytest.raises(NetworkError, match="HTTP 503"):
            provider.fetch_trades("2026-09-16", "2026-09-16")
