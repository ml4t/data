"""Tests for Alpaca OHLCV provider (requires ``alpaca-py`` / ``ml4t-data[alpaca]``)."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

pytest.importorskip("alpaca")

from ml4t.data.core.exceptions import AuthenticationError, DataValidationError
from ml4t.data.providers.alpaca import (
    AlpacaHistoricalProvider,
    AlpacaProvider,
    _bars_pandas_to_polars,
    _bars_pandas_to_polars_batch,
    _infer_kind,
    _timeframe_for,
)


class TestInferKind:
    """Tests for asset-class inference from symbol."""

    def test_crypto_slash(self):
        assert _infer_kind("BTC/USD") == "crypto"
        assert _infer_kind("ETH/USD") == "crypto"

    def test_option_oc(self):
        assert _infer_kind("AAPL241220C00300000") == "option"

    def test_stock(self):
        assert _infer_kind("AAPL") == "stock"
        assert _infer_kind("MSFT") == "stock"


class TestFrequencyMap:
    """FREQUENCY_MAP mirrors Yahoo-style aliases."""

    def test_keys_cover_aliases(self):
        p = AlpacaProvider()
        for key in (
            "daily",
            "1day",
            "minute",
            "5minute",
            "hourly",
            "monthly",
        ):
            assert key in p.FREQUENCY_MAP

    def test_map_normalizes_to_internal_names(self):
        p = AlpacaProvider()
        assert p.FREQUENCY_MAP["daily"] == "1day"
        assert p.FREQUENCY_MAP["minute"] == "1minute"
        assert p.FREQUENCY_MAP["5minute"] == "5minute"


class TestTimeframeFor:
    """Tests for Alpaca TimeFrame mapping."""

    def test_daily(self):
        from alpaca.data.timeframe import TimeFrameUnit

        tf = _timeframe_for("daily")
        assert tf.amount == 1
        assert tf.unit == TimeFrameUnit.Day

    def test_five_minute(self):
        from alpaca.data.timeframe import TimeFrameUnit

        tf = _timeframe_for("5minute")
        assert tf.amount == 5
        assert tf.unit == TimeFrameUnit.Minute

    def test_invalid_frequency_raises_data_validation(self):
        with pytest.raises(DataValidationError, match="Unsupported frequency"):
            _timeframe_for("not_a_real_freq")


class TestBarsPandasToPolars:
    """Pandas BarSet → Polars (single- and multi-symbol)."""

    def test_multiindex_symbol_timestamp(self):
        idx = pd.MultiIndex.from_tuples(
            [("BTC/USD", pd.Timestamp("2024-01-02", tz="UTC"))],
            names=["symbol", "timestamp"],
        )
        df_pd = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=idx,
        )
        end_inc = datetime(2024, 1, 3, 23, 59, 59, 999999, tzinfo=UTC)
        out = _bars_pandas_to_polars(df_pd, "BTC/USD", end_inc)
        assert len(out) == 1
        assert out.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert out["symbol"][0] == "BTC/USD"
        assert out["close"][0] == 1.5

    def test_end_filter_excludes_bar_after_range(self):
        idx = pd.MultiIndex.from_tuples(
            [
                ("BTC/USD", pd.Timestamp("2024-01-02", tz="UTC")),
                ("BTC/USD", pd.Timestamp("2024-01-05", tz="UTC")),
            ],
            names=["symbol", "timestamp"],
        )
        df_pd = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [2.0, 3.0],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [100.0, 200.0],
            },
            index=idx,
        )
        end_inc = datetime(2024, 1, 3, 23, 59, 59, 999999, tzinfo=UTC)
        out = _bars_pandas_to_polars(df_pd, "BTC/USD", end_inc)
        assert len(out) == 1

    def test_batch_two_symbols(self):
        idx = pd.MultiIndex.from_tuples(
            [
                ("BTC/USD", pd.Timestamp("2024-01-02", tz="UTC")),
                ("ETH/USD", pd.Timestamp("2024-01-02", tz="UTC")),
            ],
            names=["symbol", "timestamp"],
        )
        df_pd = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [2.0, 3.0],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [100.0, 200.0],
            },
            index=idx,
        )
        end_inc = datetime(2024, 1, 3, 23, 59, 59, 999999, tzinfo=UTC)
        out = _bars_pandas_to_polars_batch(df_pd, end_inc)
        assert len(out) == 2
        assert set(out["symbol"].to_list()) == {"BTC/USD", "ETH/USD"}


class TestCreateEmptyDataframe:
    """Empty OHLCV schema (BaseProvider)."""

    def test_empty_dataframe_columns(self):
        provider = AlpacaProvider()
        df = provider._create_empty_dataframe()
        assert list(df.columns) == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_empty_dataframe_length(self):
        provider = AlpacaProvider()
        assert len(provider._create_empty_dataframe()) == 0


class TestAuthentication:
    """Stock/option paths require credentials."""

    def test_stock_without_keys_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            p = AlpacaProvider()
            with pytest.raises(AuthenticationError, match="api_key and secret_key"):
                p.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05", "daily")

    def test_option_without_keys_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            p = AlpacaProvider()
            with pytest.raises(AuthenticationError, match="api_key and secret_key"):
                p.fetch_ohlcv("AAPL241220C00300000", "2024-12-01", "2024-12-05", "daily")

    def test_batch_stock_without_keys_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            p = AlpacaProvider()
            with pytest.raises(AuthenticationError, match="stock or option"):
                p.fetch_batch_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-01-05", "daily")

    def test_init_reads_env_keys(self):
        with patch.dict(
            "os.environ",
            {"ALPACA_API_KEY": "pk-x", "ALPACA_SECRET_KEY": "sec-y"},
        ):
            p = AlpacaProvider()
            assert p._api_key == "pk-x"
            assert p._secret_key == "sec-y"


class TestFetchCryptoMocked:
    """Offline tests with mocked Alpaca client."""

    def test_fetch_crypto_returns_canonical_schema(self):
        idx = pd.MultiIndex.from_tuples(
            [("BTC/USD", pd.Timestamp("2024-01-02", tz="UTC"))],
            names=["symbol", "timestamp"],
        )
        df_pd = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=idx,
        )
        barset = MagicMock()
        barset.df = df_pd

        with patch("ml4t.data.providers.alpaca.CryptoHistoricalDataClient") as client_cls:
            inst = MagicMock()
            client_cls.return_value = inst
            inst.get_crypto_bars.return_value = barset
            p = AlpacaProvider()
            out = p._fetch_and_transform_data("BTC/USD", "2024-01-01", "2024-01-03", "daily")
        assert len(out) == 1
        assert out["close"][0] == 1.5

    def test_fetch_batch_crypto_two_symbols_mocked(self):
        idx = pd.MultiIndex.from_tuples(
            [
                ("BTC/USD", pd.Timestamp("2024-01-02", tz="UTC")),
                ("ETH/USD", pd.Timestamp("2024-01-02", tz="UTC")),
            ],
            names=["symbol", "timestamp"],
        )
        df_pd = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [2.0, 3.0],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [100.0, 200.0],
            },
            index=idx,
        )
        barset = MagicMock()
        barset.df = df_pd

        with patch("ml4t.data.providers.alpaca.CryptoHistoricalDataClient") as client_cls:
            inst = MagicMock()
            client_cls.return_value = inst
            inst.get_crypto_bars.return_value = barset
            p = AlpacaProvider()
            out = p.fetch_batch_ohlcv(
                ["BTC/USD", "ETH/USD"], "2024-01-01", "2024-01-03", "daily", chunk_size=10
            )
        assert len(out) == 2
        assert out.columns == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_fetch_batch_empty_symbols(self):
        p = AlpacaProvider()
        out = p.fetch_batch_ohlcv([], "2024-01-01", "2024-01-03")
        assert len(out) == 0


class TestAlpacaProviderInit:
    """Naming and backward-compat alias."""

    def test_name(self):
        assert AlpacaProvider().name == "alpaca"

    def test_historical_alias(self):
        assert AlpacaHistoricalProvider is AlpacaProvider


class TestInvalidFrequencyInFetch:
    """Unsupported frequency → DataValidationError before auth."""

    def test_fetch_bad_frequency(self):
        p = AlpacaProvider()
        with pytest.raises(DataValidationError, match="Unsupported frequency"):
            p.fetch_ohlcv("BTC/USD", "2024-01-01", "2024-01-03", "bad_freq_xyz")

    def test_stock_bad_frequency_without_keys_is_validation_not_auth(self):
        with patch.dict("os.environ", {}, clear=True):
            p = AlpacaProvider()
            with pytest.raises(DataValidationError, match="Unsupported frequency"):
                p.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-03", "bad_freq_xyz")


class TestFetchBatchAsync:
    """Mirror Yahoo's asyncio.to_thread batch wrapper."""

    def test_fetch_batch_ohlcv_async_delegates(self):
        p = AlpacaProvider()
        with patch.object(p, "fetch_batch_ohlcv", return_value=pl.DataFrame()) as mock_fb:
            out = asyncio.run(
                p.fetch_batch_ohlcv_async(
                    ["BTC/USD"], "2024-01-01", "2024-01-02", frequency="daily"
                )
            )
        mock_fb.assert_called_once()
        assert len(out) == 0
