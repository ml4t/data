"""Provider management for DataManager.

This module handles provider routing, caching, and lifecycle management.
It includes the ProviderRouter for symbol-to-provider mapping and the
ProviderManager for provider instance management.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class ProviderRouter:
    """Routes symbols to appropriate providers based on patterns.

    The router matches symbols against configured regex patterns to determine
    which provider should handle data requests for that symbol.

    Attributes:
        patterns: List of (compiled_pattern, provider_name) tuples
        _cache: Cache of symbol to provider routing decisions

    Example:
        >>> router = ProviderRouter()
        >>> router.add_pattern(r"^BTC", "binance")
        >>> router.add_pattern(r"^[A-Z]{4}$", "yahoo")
        >>> router.get_provider("BTCUSD")  # Returns "binance"
        >>> router.get_provider("AAPL")    # Returns "yahoo"
    """

    def __init__(self) -> None:
        """Initialize provider router."""
        self.patterns: list[tuple[re.Pattern, str]] = []
        self._cache: dict[str, str] = {}

    def add_pattern(self, pattern: str, provider: str) -> None:
        """Add a routing pattern.

        Args:
            pattern: Regular expression pattern to match symbols
            provider: Provider name to route to
        """
        compiled = re.compile(pattern)
        self.patterns.append((compiled, provider))
        # Clear cache when patterns change
        self._cache.clear()

    def get_provider(self, symbol: str, override: str | None = None) -> str | None:
        """Get provider for a symbol.

        Args:
            symbol: Symbol to route
            override: Optional provider override (takes precedence)

        Returns:
            Provider name or None if no match
        """
        if override:
            return override

        # Check cache first
        if symbol in self._cache:
            return self._cache[symbol]

        # Match against patterns
        for pattern, provider in self.patterns:
            if pattern.match(symbol):
                self._cache[symbol] = provider
                return provider

        return None

    def clear_cache(self) -> None:
        """Clear the routing cache."""
        self._cache.clear()

    def setup_default_patterns(self) -> None:
        """Setup default routing patterns if none configured.

        Default patterns:
        - Forex: EURUSD or EUR_USD format → oanda
        - Crypto: BTC/ETH/SOL prefixes → cryptocompare
        - Futures: Continuous (.v.) or contracts (ESZ4) → databento
        """
        if self.patterns:
            return  # Don't override existing patterns

        # Forex patterns
        self.add_pattern(r"^[A-Z]{6}$", "oanda")  # EURUSD format
        self.add_pattern(r"^[A-Z]{3}_[A-Z]{3}$", "oanda")  # EUR_USD format

        # Crypto patterns
        self.add_pattern(r"^(BTC|ETH|SOL|ADA|DOT|LINK|AVAX|MATIC)", "cryptocompare")
        self.add_pattern(r"^[A-Z]{3,5}(-USD)?$", "cryptocompare")

        # Futures patterns
        self.add_pattern(r"^[A-Z]+\.(v|V)\.[0-9]+$", "databento")  # Continuous
        self.add_pattern(r"^[A-Z]{2,3}[FGHJKMNQUVXZ][0-9]$", "databento")  # Contracts


class ProviderManager:
    """Manages provider instances and availability detection.

    This class handles:
    - Provider class registration
    - Provider instance caching (connection pooling)
    - Availability detection based on configuration
    - Provider lifecycle management

    Attributes:
        config: Configuration dictionary
        providers: Cached provider instances
        _provider_classes: Registered provider classes
        _available_providers: List of available provider names

    Example:
        >>> from ml4t.data.managers import ConfigManager, ProviderManager
        >>> config_mgr = ConfigManager()
        >>> provider_mgr = ProviderManager(config_mgr.config)
        >>> yahoo = provider_mgr.get_provider("yahoo")
        >>> df = yahoo.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")
    """

    # Provider class mapping - lazy loaded to avoid circular imports
    _PROVIDER_CLASSES: dict[str, type] | None = None

    # Providers that work without API keys
    FREE_PROVIDERS = frozenset(
        {
            "yahoo",
            "binance",
            "binance_public",
            "mock",
            "cryptocompare",
            "synthetic",
            "okx",
        }
    )

    # Providers that require API keys
    KEYED_PROVIDERS = frozenset(
        {
            "databento",
            "oanda",
        }
    )

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize ProviderManager.

        Args:
            config: Configuration dictionary (from ConfigManager)
        """
        self.config = config
        self.providers: dict[str, BaseProvider] = {}
        self._provider_classes = self._get_provider_classes()
        self._available_providers: list[str] = []
        self._detect_available_providers()

        logger.debug(
            "ProviderManager initialized",
            available_providers=self._available_providers,
        )

    @classmethod
    def _get_provider_classes(cls) -> dict[str, type]:
        """Get provider class mapping (lazy loaded).

        Returns:
            Dictionary mapping provider names to classes
        """
        if cls._PROVIDER_CLASSES is not None:
            return cls._PROVIDER_CLASSES

        # Import core providers (no external dependencies)
        from ml4t.data.providers.binance import BinanceProvider
        from ml4t.data.providers.binance_public import BinancePublicProvider
        from ml4t.data.providers.cryptocompare import CryptoCompareProvider
        from ml4t.data.providers.mock import MockProvider
        from ml4t.data.providers.okx import OKXProvider
        from ml4t.data.providers.synthetic import SyntheticProvider

        provider_classes: dict[str, type] = {
            "binance": BinanceProvider,
            "binance_public": BinancePublicProvider,
            "cryptocompare": CryptoCompareProvider,
            "mock": MockProvider,
            "okx": OKXProvider,
            "synthetic": SyntheticProvider,
        }

        # Optional providers with external dependencies
        try:
            from ml4t.data.providers.yahoo import YahooFinanceProvider

            provider_classes["yahoo"] = YahooFinanceProvider
        except ImportError:
            pass

        try:
            from ml4t.data.providers.databento import DataBentoProvider

            provider_classes["databento"] = DataBentoProvider
        except ImportError:
            pass

        try:
            from ml4t.data.providers.oanda import OandaProvider

            provider_classes["oanda"] = OandaProvider
        except ImportError:
            pass

        cls._PROVIDER_CLASSES = provider_classes
        return provider_classes

    def _detect_available_providers(self) -> None:
        """Detect which providers are available based on configuration."""
        self._available_providers = []
        providers_config = self.config.get("providers", {})

        # Check configured providers
        for provider_name, provider_config in providers_config.items():
            if (
                provider_name in self.FREE_PROVIDERS
                or provider_name in self.KEYED_PROVIDERS
                and provider_config.get("api_key")
            ):
                self._available_providers.append(provider_name)

        # Check environment for API keys not in config
        env_to_provider = {
            "CRYPTOCOMPARE_API_KEY": "cryptocompare",
            "DATABENTO_API_KEY": "databento",
            "OANDA_API_KEY": "oanda",
        }

        for env_var, provider_name in env_to_provider.items():
            if env_var in os.environ and provider_name not in self._available_providers:
                self._available_providers.append(provider_name)

        # Add free providers if not already detected
        for free_provider in self.FREE_PROVIDERS:
            if free_provider not in self._available_providers:
                self._available_providers.append(free_provider)

    @property
    def available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return self._available_providers.copy()

    def is_available(self, provider_name: str) -> bool:
        """Check if a provider is available.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider is available
        """
        return provider_name in self._available_providers

    def get_provider(self, provider_name: str) -> BaseProvider:
        """Get or create a provider instance.

        Provider instances are cached for connection reuse.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is not available or initialization fails
        """
        if provider_name not in self._available_providers:
            raise ValueError(
                f"Provider '{provider_name}' not available. "
                f"Available providers: {self._available_providers}"
            )

        # Return cached instance if exists
        if provider_name in self.providers:
            return self.providers[provider_name]

        # Get provider class
        provider_class = self._provider_classes.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Get provider configuration
        provider_config = self.config.get("providers", {}).get(provider_name, {})

        # Create provider instance
        try:
            provider = provider_class(**provider_config)
            self.providers[provider_name] = provider
            logger.info(f"Initialized {provider_name} provider")
            return provider
        except Exception as e:
            raise ValueError(f"Failed to initialize {provider_name}: {e}") from e

    def get_provider_info(self, provider_name: str) -> dict[str, Any]:
        """Get information about a provider.

        Args:
            provider_name: Provider name

        Returns:
            Provider information dictionary

        Raises:
            ValueError: If provider is not available
        """
        if provider_name not in self._available_providers:
            raise ValueError(f"Provider '{provider_name}' not available")

        config = self.config.get("providers", {}).get(provider_name, {})

        return {
            "name": provider_name,
            "configured": provider_name in self.config.get("providers", {}),
            "has_api_key": bool(config.get("api_key")),
            "is_free": provider_name in self.FREE_PROVIDERS,
            "config": {k: v for k, v in config.items() if k != "api_key"},
        }

    def close_all(self) -> None:
        """Close all provider connections and clear cache."""
        for provider in self.providers.values():
            if hasattr(provider, "close"):
                try:
                    provider.close()
                except Exception as e:
                    logger.warning(f"Error closing provider: {e}")

        self.providers.clear()
        logger.info("Closed all provider connections")

    def register_provider(self, name: str, provider_class: type) -> None:
        """Register a custom provider class.

        Args:
            name: Provider name
            provider_class: Provider class (must extend BaseProvider)
        """
        self._provider_classes[name] = provider_class
        if name not in self._available_providers:
            self._available_providers.append(name)
        logger.info(f"Registered custom provider: {name}")
