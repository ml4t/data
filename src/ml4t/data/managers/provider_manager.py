"""Provider management for DataManager.

This module handles provider routing, caching, and lifecycle management.
It includes the ProviderRouter for symbol-to-provider mapping and the
ProviderManager for provider instance management.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog

from ml4t.data.providers.registry import PROVIDER_REGISTRY, ProviderSpec, get_provider_spec

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
        self.patterns: list[tuple[re.Pattern[str], str]] = []
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
        """Set up routing only for symbol formats with one clear asset class.

        Default patterns:
        - Forex: ``EUR_USD`` format routes to OANDA
        - Futures: continuous ``ROOT.v.N`` symbols route to Databento

        Bare tickers and compact pairs are intentionally not inferred because their
        formats overlap across equities, crypto, forex, and futures. Delimited crypto
        pairs are also not inferred because several providers accept the same syntax.
        """
        if self.patterns:
            return  # Don't override existing patterns

        self.add_pattern(r"^[A-Z]{3}_[A-Z]{3}$", "oanda")
        self.add_pattern(r"^[A-Z]+\.(v|V)\.[0-9]+$", "databento")


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
        spec.name
        for spec in PROVIDER_REGISTRY.values()
        if spec.manager_compatible and not spec.credentials and not spec.required_configuration
    )

    # Providers that require API keys
    KEYED_PROVIDERS = frozenset(
        spec.name
        for spec in PROVIDER_REGISTRY.values()
        if spec.manager_compatible and spec.credentials
    )

    # Credential fields that must never leave get_provider_info's sanitized view
    SECRET_FIELDS = frozenset(
        requirement.config_field
        for spec in PROVIDER_REGISTRY.values()
        for requirement in spec.credentials
    ) | {"api_key", "api_secret"}

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize ProviderManager.

        Args:
            config: Configuration dictionary (from ConfigManager)
        """
        self.config = config
        self.providers: dict[str, BaseProvider] = {}
        self._provider_classes = dict(self._get_provider_classes())
        self._provider_specs = dict(PROVIDER_REGISTRY)
        self._register_configured_aliases()
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

        provider_classes: dict[str, type] = {}
        for spec in PROVIDER_REGISTRY.values():
            if not spec.manager_compatible:
                continue
            try:
                provider_classes[spec.name] = spec.load_class()
            except ImportError:
                continue

        cls._PROVIDER_CLASSES = provider_classes
        return provider_classes

    def _register_configured_aliases(self) -> None:
        """Bind configured instance names to their registered provider types."""
        for name, config in self.config.get("providers", {}).items():
            provider_type = config.get("type", name)
            if not isinstance(provider_type, str) or provider_type == name:
                continue
            spec = get_provider_spec(provider_type)
            if not spec.manager_compatible:
                raise ValueError(
                    f"Provider '{name}' uses type '{provider_type}', which is direct-API only"
                )
            provider_class = self._provider_classes.get(provider_type)
            if provider_class is None:
                logger.warning(
                    "Configured provider alias is unavailable",
                    provider=name,
                    provider_type=provider_type,
                    extra=spec.extra,
                )
                continue
            self._provider_classes[name] = provider_class
            self._provider_specs[name] = spec

    def _detect_available_providers(self) -> None:
        """Detect which providers are available based on configuration."""
        providers_config = self.config.get("providers", {})
        self._available_providers = [
            name
            for name, spec in self._provider_specs.items()
            if name in self._provider_classes and spec.is_configured(providers_config.get(name, {}))
        ]

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

    def get_provider(
        self,
        provider_name: str,
        *,
        required_capability: str | None = None,
    ) -> BaseProvider:
        """Get or create a provider instance.

        Provider instances are cached for connection reuse.

        Args:
            provider_name: Name of the provider
            required_capability: Capability the caller will invoke

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is not available or initialization fails
        """
        spec: ProviderSpec | None = self._provider_specs.get(provider_name)

        if required_capability is not None and spec is not None:
            if required_capability not in spec.capabilities:
                supported = ", ".join(sorted(spec.capabilities))
                raise ValueError(
                    f"Provider '{provider_name}' does not support '{required_capability}'. "
                    f"Supported capabilities: {supported}"
                )
            if not spec.manager_compatible:
                raise ValueError(
                    f"Provider '{provider_name}' is available only through its direct API"
                )

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
        provider_config = dict(self.config.get("providers", {}).get(provider_name, {}))
        provider_config.pop("enabled", None)
        provider_config.pop("name", None)
        provider_config.pop("type", None)
        extra = provider_config.pop("extra", None)
        if isinstance(extra, dict):
            provider_config = {**extra, **provider_config}

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
        spec = self._provider_specs.get(provider_name)
        if spec is None:
            raise ValueError(f"Provider '{provider_name}' not available: not registered")
        config = self.config.get("providers", {}).get(provider_name, {})
        sanitized_config = {
            key: (
                {
                    nested_key: nested_value
                    for nested_key, nested_value in value.items()
                    if nested_key not in self.SECRET_FIELDS
                }
                if key == "extra" and isinstance(value, dict)
                else value
            )
            for key, value in config.items()
            if key not in self.SECRET_FIELDS
        }

        return {
            "name": provider_name,
            "description": spec.description,
            "available": provider_name in self._available_providers,
            "configured": provider_name in self.config.get("providers", {}),
            "has_api_key": spec.has_api_key(config),
            "is_free": not spec.credentials and not spec.required_configuration,
            "capabilities": sorted(spec.capabilities),
            "credential_environment": list(spec.credential_environment),
            "extra": spec.extra,
            "deprecated": spec.deprecated,
            "config": sanitized_config,
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
