"""Asset class definitions and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar


class AssetClass(str, Enum):
    """Supported asset classes.

    Canonical enum for all asset class references across ml4t-data.
    Plural aliases (EQUITIES, FUTURES, OPTIONS) are provided for backward
    compatibility with config files and serialized data.
    """

    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    FIXED_INCOME = "fixed_income"
    INDEX = "index"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"

    # Plural aliases for backward compatibility
    EQUITIES = "equities"
    FUTURES = "futures"
    OPTIONS = "options"


@dataclass
class AssetInfo:
    """Information about an asset."""

    symbol: str
    asset_class: AssetClass
    name: str | None = None
    exchange: str | None = None
    currency: str | None = None

    # Equity-specific
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None

    # Crypto-specific
    base_currency: str | None = None
    quote_currency: str | None = None
    is_stablecoin: bool = False
    blockchain: str | None = None

    # Derivatives-specific
    underlying: str | None = None
    expiry: str | None = None
    strike: float | None = None
    contract_size: float | None = None

    def __post_init__(self) -> None:
        """Validate asset info based on asset class."""
        if self.asset_class == AssetClass.CRYPTO:
            # Parse crypto pairs
            if "/" in self.symbol:
                parts = self.symbol.split("/")
                if len(parts) == 2:
                    self.base_currency = parts[0]
                    self.quote_currency = parts[1]
            elif "-" in self.symbol:
                parts = self.symbol.split("-")
                if len(parts) == 2:
                    self.base_currency = parts[0]
                    self.quote_currency = parts[1]

            # Check for stablecoins
            stablecoins = ["USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "USDD"]
            if self.base_currency in stablecoins or self.quote_currency in stablecoins:
                self.is_stablecoin = True

    @property
    def is_derivative(self) -> bool:
        """Check if asset is a derivative."""
        return self.asset_class in [AssetClass.OPTION, AssetClass.FUTURE]

    @property
    def is_crypto_pair(self) -> bool:
        """Check if asset is a crypto trading pair."""
        return (
            self.asset_class == AssetClass.CRYPTO
            and self.base_currency is not None
            and self.quote_currency is not None
        )

    @property
    def requires_24_7_calendar(self) -> bool:
        """Check if asset trades 24/7."""
        return self.asset_class in [AssetClass.CRYPTO, AssetClass.FOREX]


class MarketHours:
    """Market hours for different asset classes."""

    SCHEDULES: ClassVar[dict] = {
        AssetClass.EQUITY: {
            "NYSE": {
                "open": "09:30",
                "close": "16:00",
                "timezone": "America/New_York",
                "days": [0, 1, 2, 3, 4],  # Monday to Friday
            },
            "NASDAQ": {
                "open": "09:30",
                "close": "16:00",
                "timezone": "America/New_York",
                "days": [0, 1, 2, 3, 4],
            },
            "LSE": {
                "open": "08:00",
                "close": "16:30",
                "timezone": "Europe/London",
                "days": [0, 1, 2, 3, 4],
            },
        },
        AssetClass.CRYPTO: {
            "24/7": {
                "open": "00:00",
                "close": "23:59",
                "timezone": "UTC",
                "days": [0, 1, 2, 3, 4, 5, 6],  # All days
            },
        },
        AssetClass.FOREX: {
            "24/5": {
                "open": "17:00",  # Sunday 5PM EST
                "close": "17:00",  # Friday 5PM EST
                "timezone": "America/New_York",
                "days": [0, 1, 2, 3, 4],  # Continuous from Sun evening to Fri evening
            },
        },
    }

    @classmethod
    def get_schedule(cls, asset_class: AssetClass, exchange: str | None = None) -> dict[str, Any]:
        """
        Get market schedule for asset class and exchange.

        Args:
            asset_class: Type of asset
            exchange: Optional exchange name

        Returns:
            Market schedule dictionary
        """
        schedules = cls.SCHEDULES.get(asset_class, {})

        if exchange and exchange in schedules:
            return schedules[exchange]

        # Return default for asset class
        if schedules:
            return next(iter(schedules.values()))

        return None


def parse_crypto_symbol(symbol: str) -> tuple[str, str]:
    """
    Parse crypto trading pair symbol.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT", "BTC-USDT", "BTCUSDT")

    Returns:
        Tuple of (base_currency, quote_currency)
    """
    # Try different separators
    if "/" in symbol:
        parts = symbol.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]

    if "-" in symbol:
        parts = symbol.split("-")
        if len(parts) == 2:
            return parts[0], parts[1]

    # Try to parse without separator (e.g., "BTCUSDT")
    # Common quote currencies
    quote_currencies = ["USDT", "USDC", "BUSD", "USD", "EUR", "BTC", "ETH", "BNB"]

    for quote in quote_currencies:
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            if base:  # Ensure base is not empty
                return base, quote

    # Default: assume no pair
    return symbol, ""


def normalize_crypto_symbol(symbol: str, separator: str = "/") -> str:
    """
    Normalize crypto symbol to standard format.

    Args:
        symbol: Input symbol
        separator: Desired separator

    Returns:
        Normalized symbol
    """
    base, quote = parse_crypto_symbol(symbol)

    if quote:
        return f"{base}{separator}{quote}"
    return symbol


def get_asset_class_from_symbol(symbol: str) -> AssetClass:
    """
    Infer asset class from symbol.

    Args:
        symbol: Asset symbol

    Returns:
        Inferred asset class
    """
    # Check for crypto patterns
    crypto_patterns = ["/", "-"]
    crypto_suffixes = ["USDT", "USDC", "BUSD", "BTC", "ETH"]

    for pattern in crypto_patterns:
        if pattern in symbol:
            return AssetClass.CRYPTO

    for suffix in crypto_suffixes:
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            return AssetClass.CRYPTO

    # Check for forex patterns (6 characters, two 3-letter currencies)
    if len(symbol) == 6 and symbol.isalpha():
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
        if symbol[:3] in currencies and symbol[3:] in currencies:
            return AssetClass.FOREX

    # Default to equity
    return AssetClass.EQUITY
