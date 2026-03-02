"""Data provider implementations.

This module provides unified access to multiple financial data providers.

Available Providers (20 live + 3 synthetic/testing):
    - BaseProvider: Abstract base class for all providers
    - YahooFinanceProvider: Yahoo Finance (free, no API key)
    - TiingoProvider: Tiingo stocks (free tier: 1000 req/day, 500 symbols/month)
    - FinnhubProvider: Finnhub multi-asset data (free tier: 60 req/min)
    - EODHDProvider: EODHD global equities (free tier: 500 req/day, 1 year depth)
    - FREDProvider: FRED economic data (free, 120 req/min)
    - AQRFactorProvider: AQR research factors (QMJ, BAB, VME)
    - FamaFrenchProvider: Fama-French factors (3-factor, 5-factor, momentum)
    - KalshiProvider: Kalshi prediction markets (no API key needed for public data)
    - PolymarketProvider: Polymarket prediction market history/order book snapshots
    - CoinGeckoProvider: CoinGecko crypto data (free, no API key)
    - BinanceProvider: Binance cryptocurrency exchange (live API, may have geo-restrictions)
    - BinancePublicProvider: Binance public data (bulk downloads, no geo-restrictions)
    - OKXProvider: OKX crypto perpetuals and funding rates (no geo-restrictions)
    - CryptoCompareProvider: CryptoCompare crypto data
    - TwelveDataProvider: Twelve Data multi-asset (stocks, forex, crypto)
    - PolygonProvider: Polygon multi-asset (stocks, options, crypto, forex)
    - OandaProvider: Oanda forex and CFDs
    - DataBentoProvider: DataBento market data
    - ITCHSampleProvider: NASDAQ TotalView-ITCH sample data (tick-level, free)
    - WikiPricesProvider: Quandl Wiki Prices (US equities 1962-2018, free)
    - SyntheticProvider: Synthetic data generator (no network required)
    - LearnedSyntheticProvider: ML-based synthetic data generator (no network required)
    - MockProvider: Mock provider for testing

Note: Updater classes have been removed to simplify the library.
If you need incremental update functionality, implement it separately using the provider's fetch_ohlcv() method.

Example:
    >>> from ml4t.data.providers import CoinGeckoProvider
    >>>
    >>> # Use provider directly
    >>> provider = CoinGeckoProvider()
    >>> data = provider.fetch_ohlcv("bitcoin", "2024-01-01", "2024-01-31")
"""

# Base classes
from ml4t.data.providers.base import BaseProvider, Provider

# Equity providers
try:
    from ml4t.data.providers.yahoo import YahooFinanceProvider
except ImportError:
    YahooFinanceProvider = None  # type: ignore

try:
    from ml4t.data.providers.tiingo import TiingoProvider
except ImportError:
    TiingoProvider = None  # type: ignore

try:
    from ml4t.data.providers.finnhub import FinnhubProvider
except ImportError:
    FinnhubProvider = None  # type: ignore

try:
    from ml4t.data.providers.eodhd import EODHDProvider
except ImportError:
    EODHDProvider = None  # type: ignore

# Economic data providers
try:
    from ml4t.data.providers.fred import FREDProvider
except ImportError:
    FREDProvider = None  # type: ignore

# Factor data providers
try:
    from ml4t.data.providers.aqr import AQRFactorProvider
except ImportError:
    AQRFactorProvider = None  # type: ignore

try:
    from ml4t.data.providers.fama_french import FamaFrenchProvider
except ImportError:
    FamaFrenchProvider = None  # type: ignore

# Prediction market providers
try:
    from ml4t.data.providers.kalshi import KalshiProvider
except ImportError:
    KalshiProvider = None  # type: ignore

try:
    from ml4t.data.providers.polymarket import PolymarketProvider
except ImportError:
    PolymarketProvider = None  # type: ignore

# Crypto providers
from ml4t.data.providers.coingecko import CoinGeckoProvider

try:
    from ml4t.data.providers.binance import BinanceProvider
except ImportError:
    BinanceProvider = None  # type: ignore

from ml4t.data.providers.binance_public import BinancePublicProvider

try:
    from ml4t.data.providers.okx import OKXProvider
except ImportError:
    OKXProvider = None  # type: ignore

try:
    from ml4t.data.providers.cryptocompare import CryptoCompareProvider
except ImportError:
    CryptoCompareProvider = None  # type: ignore

# Forex providers
try:
    from ml4t.data.providers.oanda import OandaProvider
except ImportError:
    OandaProvider = None  # type: ignore

# Multi-asset providers
from ml4t.data.providers.polygon import PolygonProvider
from ml4t.data.providers.twelve_data import TwelveDataProvider

# Market data providers
try:
    from ml4t.data.providers.databento import DataBentoProvider
except ImportError:
    DataBentoProvider = None  # type: ignore[assignment, misc]

# Tick data providers
from ml4t.data.providers.nasdaq_itch import ITCHSampleProvider

try:
    from ml4t.data.providers.wiki_prices import WikiPricesProvider
except ImportError:
    WikiPricesProvider = None  # type: ignore

# Synthetic data
from ml4t.data.providers.learned_synthetic import LearnedSyntheticProvider

# Testing
from ml4t.data.providers.mock import MockProvider
from ml4t.data.providers.synthetic import SyntheticProvider

__all__ = [
    # Base classes
    "BaseProvider",
    "Provider",
    # Equity providers
    "YahooFinanceProvider",
    "TiingoProvider",
    "FinnhubProvider",
    "EODHDProvider",
    # Economic data providers
    "FREDProvider",
    # Factor data providers
    "AQRFactorProvider",
    "FamaFrenchProvider",
    # Prediction market providers
    "KalshiProvider",
    "PolymarketProvider",
    # Crypto providers
    "CoinGeckoProvider",
    "BinanceProvider",
    "BinancePublicProvider",
    "OKXProvider",
    "CryptoCompareProvider",
    # Forex providers
    "OandaProvider",
    # Multi-asset providers
    "PolygonProvider",
    "TwelveDataProvider",
    # Market data providers
    "DataBentoProvider",
    # Tick data providers
    "ITCHSampleProvider",
    "WikiPricesProvider",
    # Synthetic data
    "SyntheticProvider",
    "LearnedSyntheticProvider",
    # Testing
    "MockProvider",
]
