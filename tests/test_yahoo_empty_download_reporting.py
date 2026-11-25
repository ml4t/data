"""What an empty ``yf.download`` is allowed to be reported as.

``yf.download`` hands back an empty frame for every per-symbol failure it has: it calls
``Ticker.history`` without ``raise_errors``, catches whatever comes back inside
``_download_one``, records it on a context object local to that call, logs it, and returns
nothing a caller can read. The provider used to conclude "symbol not found" from that
emptiness alone, so **any** cause was reported as an invalid symbol. Rate limiting is the
common one and the one that reached CI, but it is an instance of the class, not the class.

Measured against yfinance 1.5.2. The behaviour these tests pin is upstream's, so
``test_ticker_history_still_takes_raise_errors`` fails loudly if the channel the fix depends
on disappears, rather than letting the probe silently stop raising.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError, YFTzMissingError

from ml4t.data.core.exceptions import (
    DataValidationError,
    RateLimitError,
    SymbolNotFoundError,
)
from ml4t.data.providers.yahoo import YahooFinanceProvider

FETCH = ("AAPL", "2024-01-01", "2024-02-01", "daily")


def _bars() -> pd.DataFrame:
    """A day of flat-column OHLCV, the shape ``Ticker.history`` returns."""
    return pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100]},
        index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
    )


@pytest.fixture
def provider() -> YahooFinanceProvider:
    return YahooFinanceProvider()


def _probing(side_effect=None, return_value=None) -> MagicMock:
    """Patch the second ask, the one that recovers why the download came back empty."""
    ticker = MagicMock()
    ticker.history = MagicMock(side_effect=side_effect, return_value=return_value)
    factory = MagicMock(return_value=ticker)
    return factory


class TestEmptyDownloadIsDiagnosed:
    def test_a_rate_limit_is_reported_as_a_rate_limit(self, provider):
        """The defect this file exists for: not 'Symbol AAPL not found or invalid'."""
        factory = _probing(side_effect=YFRateLimitError())
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
            pytest.raises(RateLimitError) as excinfo,
        ):
            provider._fetch_and_transform_data(*FETCH)

        assert "Rate limit" in str(excinfo.value)
        assert "not found or invalid" not in str(excinfo.value)

    def test_a_rate_limit_is_retryable_so_the_base_class_retries_it(self, provider):
        """Option 2 of the issue, delivered by the mapping rather than by new machinery.

        ``BaseProvider.fetch_ohlcv`` retries a ``NetworkError`` whose ``retryable`` is set.
        A ``SymbolNotFoundError`` is not a ``NetworkError``, so the old reporting made a 429
        fatal as well as misleading.
        """
        factory = _probing(side_effect=YFRateLimitError())
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
            pytest.raises(RateLimitError) as excinfo,
        ):
            provider._fetch_and_transform_data(*FETCH)

        assert excinfo.value.retryable is True

    def test_a_rate_limit_survives_the_catch_all(self, provider):
        """``_fetch_and_transform_data`` turns unnamed exceptions into DataValidationError.

        That conversion would undo the fix two lines after it was made: a
        ``DataValidationError`` is not a ``NetworkError`` and is therefore not retried.
        """
        factory = _probing(side_effect=YFRateLimitError())
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
        ):
            with pytest.raises(RateLimitError):
                provider._fetch_and_transform_data(*FETCH)
            with pytest.raises(DataValidationError):
                # ...while something genuinely unclassifiable still converts.
                with patch(
                    "ml4t.data.providers.yahoo.yf.download",
                    side_effect=ValueError("malformed"),
                ):
                    provider._fetch_and_transform_data(*FETCH)

    def test_a_named_yahoo_refusal_carries_its_reason(self, provider):
        """Still SymbolNotFoundError, but no longer a guess: the reason is in details."""
        factory = _probing(side_effect=YFPricesMissingError("AAPL", "no price data found"))
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
            pytest.raises(SymbolNotFoundError) as excinfo,
        ):
            provider._fetch_and_transform_data(*FETCH)

        assert "reason" in excinfo.value.details

    def test_a_genuinely_unknown_symbol_is_still_reported_as_one(self, provider):
        """The only case the old message was right about, and it must keep working."""
        factory = _probing(return_value=pd.DataFrame())
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
            pytest.raises(SymbolNotFoundError) as excinfo,
        ):
            provider._fetch_and_transform_data(*FETCH)

        assert "not found or invalid" in str(excinfo.value)
        assert "reason" not in excinfo.value.details

    def test_a_transient_empty_download_is_recovered_rather_than_raised(self, provider):
        """If the second ask answers, throwing that answer away to raise would be perverse."""
        factory = _probing(return_value=_bars())
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
        ):
            df = provider._fetch_and_transform_data(*FETCH)

        assert df.height == 1
        assert df["close"].to_list() == [1.5]


class TestTheHappyPathIsUntouched:
    def test_a_non_empty_download_never_asks_twice(self, provider):
        """The probe costs a request, so it must only run on a path that already failed."""
        factory = _probing(return_value=_bars())
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=_bars()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
        ):
            df = provider._fetch_and_transform_data(*FETCH)

        assert df.height == 1
        factory.assert_not_called()


class TestTheUpstreamChannelTheFixDependsOn:
    def test_ticker_history_still_takes_raise_errors(self):
        """``yf.download`` has no ``raise_errors``; the path it calls internally does.

        If a yfinance release removes it, the probe stops raising and every empty download
        silently reports "symbol not found" again - the defect, back and invisible.
        """
        import inspect

        import yfinance as yf
        from yfinance.scrapers.history import PriceHistory

        assert "raise_errors" in inspect.signature(PriceHistory.history).parameters
        assert "raise_errors" not in inspect.signature(yf.download).parameters

    def test_shared_errors_is_not_a_channel(self):
        """It looks like one. In 1.5.2 it is declared and never written.

        Recorded so the next reader does not spend the time finding that out again.
        """
        import yfinance.shared as shared

        assert shared._ERRORS == {}


WEEKEND = ("SPY", "2025-06-07", "2025-06-08", "daily")


def _known_symbol_without_bars_in_window() -> MagicMock:
    """What yfinance 1.5.2 does for SPY over Saturday and Sunday, measured live.

    The ranged ask resolves SPY's exchange timezone and then finds no bar, so it raises
    ``YFPricesMissingError``; asking for the recent sessions of the same symbol succeeds.
    """

    def history(**kwargs):
        if "period" in kwargs:
            return _bars()
        raise YFPricesMissingError("SPY", "(1d 2025-06-07 -> 2025-06-09)")

    ticker = MagicMock()
    ticker.history = MagicMock(side_effect=history)
    return MagicMock(return_value=ticker)


class TestAWindowWithoutSessionsIsNotAnUnknownSymbol:
    """ml4t/data#164: a valid symbol over a weekend or holiday was reported as not found.

    A caller acting on ``SymbolNotFoundError`` drops the symbol from its universe; one asking
    for yesterday on a Monday has nothing to do. Like the other adapters, Yahoo now answers an
    empty window for a known symbol with the canonical empty frame.
    """

    def test_a_weekend_returns_the_canonical_empty_frame(self, provider):
        """Prevents: SymbolNotFoundError for SPY over 2025-06-07..2025-06-08."""
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", _known_symbol_without_bars_in_window()),
        ):
            df = provider.fetch_ohlcv(*WEEKEND)

        assert df.is_empty()
        assert df.schema == provider._create_empty_dataframe().schema

    def test_empty_ranged_history_still_recognizes_a_symbol_with_recent_bars(self, provider):
        ticker = _probing(side_effect=[pd.DataFrame(), _bars()])
        with (
            patch("yfinance.download", return_value=pd.DataFrame()),
            patch("yfinance.Ticker", ticker),
        ):
            bars = provider.fetch_ohlcv(*WEEKEND)
        assert bars.is_empty()
        assert bars.schema == provider._create_empty_dataframe().schema
        assert ticker.return_value.history.call_args.kwargs["period"] == "1mo"

    def test_a_window_holding_only_the_current_session_placeholder_is_empty(self, provider):
        """Prevents: SymbolNotFoundError when Yahoo's only row is the null-price placeholder.

        Yahoo answered with a row for the symbol, so the symbol is known; the window has no bar.
        """
        placeholder = pd.DataFrame(
            {
                "Open": [float("nan")],
                "High": [float("nan")],
                "Low": [float("nan")],
                "Close": [float("nan")],
                "Volume": [1_000],
            },
            index=pd.DatetimeIndex(["2025-06-09"], name="Date"),
        )
        with patch("ml4t.data.providers.yahoo.yf.download", return_value=placeholder):
            df = provider.fetch_ohlcv("SPY", "2025-06-09", "2025-06-09", "daily")

        assert df.is_empty()
        assert df.schema == provider._create_empty_dataframe().schema

    def test_adjusted_close_over_a_weekend_returns_an_empty_typed_frame(self, provider):
        """Prevents: SymbolNotFoundError from ``fetch_adjusted_close`` over a weekend."""
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", _known_symbol_without_bars_in_window()),
        ):
            df = provider.fetch_adjusted_close("SPY", "2025-06-07", "2025-06-09")

        assert df.is_empty()
        assert df.columns == ["timestamp", "symbol", "close"]
        assert df.schema["timestamp"] == pl.Datetime("us", "UTC")
        assert provider.last_fetch_metadata["price_basis"] == "yahoo_auto_adjusted"

    def test_a_rate_limit_while_confirming_the_symbol_is_a_rate_limit(self, provider):
        """Prevents: a throttled confirmation being reported as an unknown symbol."""

        def history(**kwargs):
            if "period" in kwargs:
                raise YFRateLimitError()
            raise YFPricesMissingError("SPY", "(1d 2025-06-07 -> 2025-06-09)")

        ticker = MagicMock()
        ticker.history = MagicMock(side_effect=history)
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", MagicMock(return_value=ticker)),
            pytest.raises(RateLimitError),
        ):
            provider._fetch_and_transform_data(*WEEKEND)

    def test_an_unknown_symbol_over_a_weekend_is_still_not_found(self, provider):
        """Prevents the fix overreaching: no timezone means Yahoo does not know the symbol."""
        factory = _probing(side_effect=YFTzMissingError("NOPE_ZZZ"))
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
            pytest.raises(SymbolNotFoundError) as excinfo,
        ):
            provider.fetch_ohlcv("NOPE_ZZZ", "2025-06-07", "2025-06-08", "daily")

        assert "no timezone found" in excinfo.value.details["reason"]

    def test_a_symbol_without_recent_prices_is_still_not_found(self, provider):
        """Prevents the fix overreaching: a timezone alone does not make a symbol tradable.

        yfinance caches timezones, so a delisted symbol can still resolve one. Only a symbol
        with recent sessions turns an empty window into an empty frame.
        """
        factory = _probing(side_effect=YFPricesMissingError("GONE", "(period=1mo)"))
        with (
            patch("ml4t.data.providers.yahoo.yf.download", return_value=pd.DataFrame()),
            patch("ml4t.data.providers.yahoo.yf.Ticker", factory),
            pytest.raises(SymbolNotFoundError),
        ):
            provider.fetch_ohlcv("GONE", "2025-06-07", "2025-06-08", "daily")
