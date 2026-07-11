"""Tests for provider fundamentals helpers and endpoints."""

import inspect
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml4t.data.core.exceptions import DataValidationError
from ml4t.data.providers.eodhd import EODHDProvider
from ml4t.data.providers.finnhub import FinnhubProvider
from ml4t.data.providers.fundamentals import rows_to_company_metrics_frame
from ml4t.data.providers.polygon import MassiveProvider
from ml4t.data.providers.yahoo import YahooFinanceProvider


class TestFundamentalsHelpers:
    def test_rows_to_company_metrics_frame_infers_schema_beyond_first_100_rows(self):
        rows = [
            {
                "symbol": "AAPL",
                "provider": "test",
                "metric": f"metric_{index}",
                "value": float(index),
                "source": None,
            }
            for index in range(100)
        ]
        rows.append(
            {
                "symbol": "AAPL",
                "provider": "test",
                "metric": "metric_100",
                "value": 100.0,
                "source": "late_source",
            }
        )

        frame = rows_to_company_metrics_frame(rows)

        assert len(frame) == 101
        assert frame["source"][-1] == "late_source"


class TestYahooFundamentals:
    def test_fetch_financials_normalizes_wide_statement(self):
        provider = YahooFinanceProvider()
        statement = pd.DataFrame(
            {
                pd.Timestamp("2024-12-31"): [100.0, 25.0],
                pd.Timestamp("2023-12-31"): [90.0, 20.0],
            },
            index=["Total Revenue", "Net Income"],
        )
        ticker = MagicMock()
        ticker.get_income_stmt.return_value = statement

        with patch("ml4t.data.providers.yahoo.yf.Ticker", return_value=ticker):
            frame = provider.fetch_financials("aapl", statement="income", period="annual")

        assert len(frame) == 4
        assert frame["symbol"].unique().to_list() == ["AAPL"]
        assert set(frame["line_item"]) == {"Total Revenue", "Net Income"}
        ticker.get_income_stmt.assert_called_once_with(freq="yearly")

    def test_fetch_financials_rejects_ttm(self):
        provider = YahooFinanceProvider()

        with pytest.raises(DataValidationError, match="annual and quarterly"):
            provider.fetch_financials("AAPL", period="ttm")

    def test_fetch_company_metrics_keeps_numeric_values(self):
        provider = YahooFinanceProvider()
        ticker = MagicMock()
        ticker.get_info.return_value = {
            "marketCap": 3_000_000_000.0,
            "enterpriseValue": "3,100,000,000",
            "longName": "Apple Inc.",
            "trailingPE": 31.5,
            "isEsgPopulated": True,
        }

        with patch("ml4t.data.providers.yahoo.yf.Ticker", return_value=ticker):
            frame = provider.fetch_company_metrics("AAPL")

        assert len(frame) == 3
        assert set(frame["metric"]) == {"enterpriseValue", "marketCap", "trailingPE"}
        assert "isEsgPopulated" not in frame["metric"].to_list()


class TestFinnhubFundamentals:
    @pytest.fixture
    def provider(self):
        return FinnhubProvider(api_key="test_key")

    def test_fetch_financials(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "financials": [
                {
                    "period": "2024-12-31",
                    "year": 2024,
                    "quarter": 0,
                    "revenue": 100.0,
                    "netIncome": 20.0,
                }
            ]
        }

        with patch.object(provider.rate_limiter, "acquire"):
            with patch.object(provider.session, "get", return_value=response) as get:
                frame = provider.fetch_financials("aapl", statement="income", period="annual")

        assert len(frame) == 2
        assert set(frame["line_item"]) == {"revenue", "netIncome"}
        assert get.call_args.kwargs["params"]["statement"] == "ic"
        assert get.call_args.kwargs["params"]["freq"] == "annual"

    def test_fetch_company_metrics(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"metric": {"peBasicExclExtraTTM": 30.5, "name": "Apple"}}

        with patch.object(provider.rate_limiter, "acquire"):
            with patch.object(provider.session, "get", return_value=response):
                frame = provider.fetch_company_metrics("AAPL")

        assert len(frame) == 1
        assert frame["metric"][0] == "peBasicExclExtraTTM"

    def test_fetch_company_metrics_provider_options_are_keyword_only(self, provider):
        signature = inspect.signature(provider.fetch_company_metrics)

        assert signature.parameters["metric_group"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["metrics"].kind is inspect.Parameter.KEYWORD_ONLY


class TestEODHDFundamentals:
    @pytest.fixture
    def provider(self):
        return EODHDProvider(api_key="test_key", rate_limit=(100, 1.0))

    def test_fetch_financials(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "General": {"CurrencyCode": "USD"},
            "Financials": {
                "Income_Statement": {
                    "yearly": {
                        "2024-12-31": {
                            "date": "2024-12-31",
                            "totalRevenue": 100.0,
                            "netIncome": 20.0,
                        }
                    }
                }
            },
        }

        with patch.object(provider.rate_limiter, "acquire"):
            with patch.object(provider.session, "get", return_value=response) as get:
                frame = provider.fetch_financials("AAPL")

        assert len(frame) == 2
        assert set(frame["line_item"]) == {"totalRevenue", "netIncome"}
        assert "fundamentals/AAPL.US" in get.call_args.args[0]

    def test_fetch_financials_accepts_string_values(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "General": {"CurrencyCode": "USD"},
            "Financials": {
                "Income_Statement": {
                    "yearly": {
                        "2024-12-31": {
                            "date": "2024-12-31",
                            "totalRevenue": "100.5",
                            "netIncome": "20",
                            "isAudited": True,
                        }
                    }
                }
            },
        }

        with patch.object(provider.rate_limiter, "acquire"):
            with patch.object(provider.session, "get", return_value=response):
                frame = provider.fetch_financials("AAPL")

        assert len(frame) == 2
        assert set(frame["line_item"]) == {"totalRevenue", "netIncome"}
        assert frame.filter(frame["line_item"] == "totalRevenue")["value"][0] == 100.5

    def test_fetch_company_metrics(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "General": {"CurrencyCode": "USD"},
            "Highlights": {"MarketCapitalization": "3000.0", "Name": "Apple"},
            "Valuation": {"TrailingPE": 30.0},
        }

        with patch.object(provider.rate_limiter, "acquire"):
            with patch.object(provider.session, "get", return_value=response):
                frame = provider.fetch_company_metrics("AAPL")

        assert len(frame) == 2
        assert set(frame["metric"]) == {"Highlights.MarketCapitalization", "Valuation.TrailingPE"}

    def test_fetch_company_metrics_provider_options_are_keyword_only(self, provider):
        signature = inspect.signature(provider.fetch_company_metrics)

        assert signature.parameters["exchange"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["metrics"].kind is inspect.Parameter.KEYWORD_ONLY


class TestMassiveFundamentals:
    @pytest.fixture
    def provider(self):
        return MassiveProvider(api_key="test_key", rate_limit=(100, 1.0))

    def test_fetch_financials(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "results": [
                {
                    "end_date": "2024-12-31",
                    "filing_date": "2025-02-01",
                    "fiscal_period": "FY",
                    "fiscal_year": 2024,
                    "financials": {
                        "income_statement": {
                            "revenues": {"value": 100.0, "unit": "USD"},
                            "net_income_loss": {"value": 20.0, "unit": "USD"},
                        }
                    },
                }
            ]
        }

        with patch.object(provider.rate_limiter, "acquire"):
            with patch.object(provider.session, "get", return_value=response) as get:
                frame = provider.fetch_financials("AAPL")

        assert len(frame) == 2
        assert set(frame["line_item"]) == {"revenues", "net_income_loss"}
        assert get.call_args.kwargs["params"]["timeframe"] == "annual"

    def test_fetch_company_metrics(self, provider):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "results": [
                {
                    "ticker": "AAPL",
                    "end_date": "2024-12-31",
                    "fiscal_period": "FY",
                    "fiscal_year": 2024,
                    "valuation": {"price_to_earnings": 30.0},
                    "profitability": {"return_on_equity": 0.45},
                }
            ]
        }

        with patch.object(provider.rate_limiter, "acquire"):
            with patch.object(provider.session, "get", return_value=response):
                frame = provider.fetch_company_metrics("AAPL")

        assert len(frame) == 2
        assert set(frame["metric"]) == {
            "valuation.price_to_earnings",
            "profitability.return_on_equity",
        }

    def test_fetch_company_metrics_provider_options_are_keyword_only(self, provider):
        signature = inspect.signature(provider.fetch_company_metrics)

        assert signature.parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["metrics"].kind is inspect.Parameter.KEYWORD_ONLY
