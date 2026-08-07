"""Canonical metadata and lazy class loading for data providers."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class CredentialRequirement:
    """One credential that may come from configuration or the environment."""

    config_field: str
    environment: tuple[str, ...]

    def is_satisfied(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> bool:
        return _has_value(config.get(self.config_field)) or any(
            _has_value(environ.get(name)) for name in self.environment
        )


@dataclass(frozen=True)
class ProviderSpec:
    """Static provider contract used by configuration, discovery, and routing."""

    name: str
    module: str
    class_name: str
    description: str
    capabilities: frozenset[str]
    credentials: tuple[CredentialRequirement, ...] = ()
    optional_credential_environment: tuple[str, ...] = ()
    required_configuration: tuple[str, ...] = ()
    extra: str | None = None
    manager_compatible: bool = True
    advertised: bool = True
    deprecated: bool = False

    @property
    def credential_environment(self) -> tuple[str, ...]:
        """Return all required credential environment variable names."""
        return tuple(name for requirement in self.credentials for name in requirement.environment)

    @property
    def access_label(self) -> str:
        """Return a concise CLI label for access requirements."""
        if self.credentials:
            return "Yes"
        if self.required_configuration:
            return "Configuration"
        if self.optional_credential_environment:
            return "Optional"
        return "No"

    def is_configured(
        self,
        config: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether constructor requirements are available without network access."""
        provider_config = config or {}
        environment = environ or os.environ
        if provider_config.get("enabled") is False:
            return False
        return all(
            requirement.is_satisfied(provider_config, environment)
            for requirement in self.credentials
        ) and all(provider_config.get(field) is not None for field in self.required_configuration)

    def has_api_key(
        self,
        config: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether an API-key value is configured for this provider."""
        provider_config = config or {}
        environment = environ or os.environ
        api_key_environment = (
            tuple(
                name
                for requirement in self.credentials
                if requirement.config_field == "api_key"
                for name in requirement.environment
            )
            + self.optional_credential_environment
        )
        return _has_value(provider_config.get("api_key")) or any(
            _has_value(environment.get(name)) for name in api_key_environment
        )

    def load_class(self) -> type:
        """Import and return the provider class named by this specification."""
        module = importlib.import_module(self.module)
        provider_class = getattr(module, self.class_name)
        if not isinstance(provider_class, type):
            raise TypeError(f"Registered provider '{self.name}' does not resolve to a class")
        return provider_class


def _credential(config_field: str, *environment: str) -> CredentialRequirement:
    return CredentialRequirement(config_field, environment)


def _has_value(value: Any) -> bool:
    """Return whether a configured value is resolved and non-empty."""
    return bool(value) and not (
        isinstance(value, str) and value.startswith("${") and value.endswith("}")
    )


def _spec(
    name: str,
    module: str,
    class_name: str,
    description: str,
    *capabilities: str,
    credentials: tuple[CredentialRequirement, ...] = (),
    optional_credential_environment: tuple[str, ...] = (),
    required_configuration: tuple[str, ...] = (),
    extra: str | None = None,
    manager_compatible: bool = True,
    advertised: bool = True,
    deprecated: bool = False,
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        module=f"ml4t.data.providers.{module}",
        class_name=class_name,
        description=description,
        capabilities=frozenset(capabilities),
        credentials=credentials,
        optional_credential_environment=optional_credential_environment,
        required_configuration=required_configuration,
        extra=extra,
        manager_compatible=manager_compatible,
        advertised=advertised,
        deprecated=deprecated,
    )


_SPECS = (
    _spec(
        "yahoo", "yahoo", "YahooFinanceProvider", "Yahoo Finance equities", "ohlcv", extra="yahoo"
    ),
    _spec(
        "alpaca",
        "alpaca",
        "AlpacaDataProvider",
        "Alpaca equities and crypto",
        "ohlcv",
        credentials=(
            _credential("api_key", "APCA_API_KEY_ID", "ALPACA_API_KEY"),
            _credential("api_secret", "APCA_API_SECRET_KEY", "ALPACA_API_SECRET"),
        ),
    ),
    _spec(
        "tiingo",
        "tiingo",
        "TiingoProvider",
        "Tiingo equities",
        "ohlcv",
        credentials=(_credential("api_key", "TIINGO_API_KEY"),),
    ),
    _spec(
        "finnhub",
        "finnhub",
        "FinnhubProvider",
        "Finnhub multi-asset data",
        "ohlcv",
        credentials=(_credential("api_key", "FINNHUB_API_KEY"),),
    ),
    _spec(
        "eodhd",
        "eodhd",
        "EODHDProvider",
        "EODHD global equities",
        "ohlcv",
        credentials=(_credential("api_key", "EODHD_API_KEY"),),
    ),
    _spec(
        "fred",
        "fred",
        "FREDProvider",
        "Federal Reserve economic series",
        "ohlcv",
        "series",
        credentials=(_credential("api_key", "FRED_API_KEY"),),
    ),
    _spec(
        "fxmacrodata",
        "fxmacrodata",
        "FXMacroDataProvider",
        "FX macroeconomic data",
        "macro",
        optional_credential_environment=("FXMACRODATA_API_KEY", "FXMD_API_KEY"),
        manager_compatible=False,
    ),
    _spec(
        "aqr",
        "aqr",
        "AQRFactorProvider",
        "AQR research factors",
        "factors",
        manager_compatible=False,
    ),
    _spec(
        "fama_french",
        "fama_french",
        "FamaFrenchProvider",
        "Fama-French research factors",
        "factors",
        manager_compatible=False,
    ),
    _spec("kalshi", "kalshi", "KalshiProvider", "Kalshi prediction markets", "ohlcv", "events"),
    _spec(
        "polymarket",
        "polymarket",
        "PolymarketProvider",
        "Polymarket prediction markets",
        "ohlcv",
        "events",
    ),
    _spec(
        "coingecko",
        "coingecko",
        "CoinGeckoProvider",
        "CoinGecko crypto markets",
        "ohlcv",
        optional_credential_environment=("COINGECKO_API_KEY",),
    ),
    _spec("binance", "binance", "BinanceProvider", "Binance spot markets", "ohlcv"),
    _spec(
        "binance_public",
        "binance_public",
        "BinancePublicProvider",
        "Binance public archives",
        "ohlcv",
    ),
    _spec("okx", "okx", "OKXProvider", "OKX perpetual markets", "ohlcv"),
    _spec(
        "cryptocompare",
        "cryptocompare",
        "CryptoCompareProvider",
        "CryptoCompare crypto markets",
        "ohlcv",
        optional_credential_environment=("CRYPTOCOMPARE_API_KEY",),
    ),
    _spec(
        "oanda",
        "oanda",
        "OandaProvider",
        "OANDA forex and CFDs",
        "ohlcv",
        credentials=(_credential("api_key", "OANDA_API_KEY"),),
        extra="oanda",
    ),
    _spec(
        "massive",
        "polygon",
        "MassiveProvider",
        "Massive multi-asset markets",
        "ohlcv",
        credentials=(_credential("api_key", "POLYGON_API_KEY", "MASSIVE_API_KEY"),),
    ),
    _spec(
        "polygon",
        "polygon",
        "PolygonProvider",
        "Deprecated Polygon alias for Massive",
        "ohlcv",
        credentials=(_credential("api_key", "POLYGON_API_KEY"),),
        advertised=False,
        deprecated=True,
    ),
    _spec(
        "twelve_data",
        "twelve_data",
        "TwelveDataProvider",
        "Twelve Data multi-asset markets",
        "ohlcv",
        credentials=(_credential("api_key", "TWELVE_DATA_API_KEY"),),
    ),
    _spec(
        "databento",
        "databento",
        "DataBentoProvider",
        "Databento exchange data",
        "ohlcv",
        credentials=(_credential("api_key", "DATABENTO_API_KEY"),),
        extra="databento",
    ),
    _spec(
        "nasdaq_itch",
        "nasdaq_itch",
        "ITCHSampleProvider",
        "NASDAQ TotalView-ITCH samples",
        "tick",
        manager_compatible=False,
    ),
    _spec(
        "wiki_prices",
        "wiki_prices",
        "WikiPricesProvider",
        "Historical Wiki Prices dataset",
        "ohlcv",
        required_configuration=("parquet_path",),
    ),
    _spec("synthetic", "synthetic", "SyntheticProvider", "Generated OHLCV data", "ohlcv"),
    _spec(
        "learned_synthetic",
        "learned_synthetic",
        "LearnedSyntheticProvider",
        "Artifact-backed generated OHLCV data",
        "synthetic_artifact",
        required_configuration=("samples",),
        manager_compatible=False,
    ),
    _spec(
        "mock",
        "mock",
        "MockProvider",
        "Deterministic test provider",
        "ohlcv",
        advertised=False,
    ),
)

PROVIDER_REGISTRY: Mapping[str, ProviderSpec] = MappingProxyType(
    {spec.name: spec for spec in _SPECS}
)


def get_provider_spec(name: str) -> ProviderSpec:
    """Return the canonical specification for a provider name."""
    try:
        return PROVIDER_REGISTRY[name.lower()]
    except KeyError as error:
        choices = ", ".join(PROVIDER_REGISTRY)
        raise ValueError(f"Unknown provider '{name}'. Registered providers: {choices}") from error


def advertised_provider_specs() -> tuple[ProviderSpec, ...]:
    """Return stable public provider specifications in registry order."""
    return tuple(spec for spec in PROVIDER_REGISTRY.values() if spec.advertised)
