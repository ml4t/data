"""Pre-defined symbol universes for common market indices and asset groups.

This module provides pre-defined lists of symbols for common use cases:
- S&P 500 (US large-cap equities)
- NASDAQ 100 (US tech-heavy large-cap equities)
- Top 100 cryptocurrencies by market cap
- Major forex pairs

Universes can be accessed as class attributes or retrieved by name (case-insensitive).
"""

from __future__ import annotations

import importlib.resources
import json
from typing import ClassVar


def _load_universe(filename: str) -> list[str]:
    """Load a universe from a JSON data file."""
    data_dir = importlib.resources.files("ml4t.data.assets") / "data"
    return json.loads((data_dir / filename).read_text())


class Universe:
    """Pre-defined symbol lists for common market indices and asset groups.

    This class provides convenient access to commonly-used symbol universes,
    eliminating the need to manually maintain symbol lists for standard indices.

    Attributes:
        SP500: S&P 500 constituents (503 symbols including share classes)
        NASDAQ100: NASDAQ 100 constituents (100 symbols)
        CRYPTO_TOP_100: Top 100 cryptocurrencies by market cap
        FOREX_MAJORS: Major currency pairs (28 pairs)

    Examples:
        Access pre-defined universes:

        >>> sp500_symbols = Universe.SP500
        >>> len(sp500_symbols)
        503

        >>> nasdaq_symbols = Universe.NASDAQ100
        >>> len(nasdaq_symbols)
        100

        Case-insensitive retrieval:

        >>> symbols = Universe.get("sp500")
        >>> symbols == Universe.SP500
        True

        >>> symbols = Universe.get("NASDAQ100")
        >>> len(symbols)
        100

        List all available universes:

        >>> available = Universe.list_universes()
        >>> "SP500" in available
        True
        >>> "NASDAQ100" in available
        True
    """

    # Load built-in universes from JSON data files
    SP500: ClassVar[list[str]] = _load_universe("sp500.json")
    NASDAQ100: ClassVar[list[str]] = _load_universe("nasdaq100.json")
    CRYPTO_TOP_100: ClassVar[list[str]] = _load_universe("crypto_top100.json")
    FOREX_MAJORS: ClassVar[list[str]] = _load_universe("forex_majors.json")

    # Internal registry of all universes
    _UNIVERSES: ClassVar[dict[str, list[str]]] = {
        "SP500": SP500,
        "NASDAQ100": NASDAQ100,
        "CRYPTO_TOP_100": CRYPTO_TOP_100,
        "FOREX_MAJORS": FOREX_MAJORS,
    }

    @classmethod
    def get(cls, universe_name: str) -> list[str]:
        """Get a universe by name (case-insensitive).

        Args:
            universe_name: Name of the universe (e.g., "sp500", "NASDAQ100")

        Returns:
            List of symbols in the universe

        Raises:
            ValueError: If universe name is not recognized

        Examples:
            >>> symbols = Universe.get("sp500")
            >>> len(symbols)
            503

            >>> symbols = Universe.get("NASDAQ100")
            >>> len(symbols)
            100

            >>> symbols = Universe.get("crypto_top_100")
            >>> "BTC" in symbols
            True

            >>> Universe.get("invalid")
            Traceback (most recent call last):
                ...
            ValueError: Unknown universe 'invalid'. Available: SP500, NASDAQ100, ...
        """
        # Normalize to uppercase with underscores
        normalized = universe_name.upper().replace("-", "_").replace(" ", "_")

        # Try exact match first
        if normalized in cls._UNIVERSES:
            return cls._UNIVERSES[normalized].copy()

        # Try fuzzy match (remove underscores)
        normalized_no_underscore = normalized.replace("_", "")
        for key, value in cls._UNIVERSES.items():
            if key.replace("_", "") == normalized_no_underscore:
                return value.copy()

        # Not found
        available = ", ".join(sorted(cls._UNIVERSES.keys()))
        raise ValueError(f"Unknown universe '{universe_name}'. Available universes: {available}")

    @classmethod
    def list_universes(cls) -> list[str]:
        """List all available universe names.

        Returns:
            Sorted list of universe names

        Examples:
            >>> universes = Universe.list_universes()
            >>> "SP500" in universes
            True
            >>> "NASDAQ100" in universes
            True
            >>> len(universes) >= 4
            True
        """
        return sorted(cls._UNIVERSES.keys())

    @classmethod
    def add_custom(cls, name: str, symbols: list[str]) -> None:
        """Add a custom universe.

        This allows users to register their own symbol lists for convenience.

        Args:
            name: Universe name (will be converted to uppercase)
            symbols: List of symbols

        Raises:
            ValueError: If universe name already exists

        Examples:
            >>> Universe.add_custom("my_portfolio", ["AAPL", "MSFT", "GOOGL"])
            >>> symbols = Universe.get("my_portfolio")
            >>> len(symbols)
            3

            >>> Universe.add_custom("sp500", ["AAPL"])  # Duplicate
            Traceback (most recent call last):
                ...
            ValueError: Universe 'SP500' already exists
        """
        normalized = name.upper().replace("-", "_").replace(" ", "_")

        if normalized in cls._UNIVERSES:
            raise ValueError(
                f"Universe '{normalized}' already exists. Use a different name or remove it first."
            )

        cls._UNIVERSES[normalized] = symbols.copy()

    @classmethod
    def remove_custom(cls, name: str) -> None:
        """Remove a custom universe.

        Built-in universes (SP500, NASDAQ100, etc.) cannot be removed.

        Args:
            name: Universe name to remove

        Raises:
            ValueError: If universe doesn't exist or is a built-in universe

        Examples:
            >>> Universe.add_custom("temp", ["AAPL"])
            >>> Universe.remove_custom("temp")
            >>> Universe.get("temp")
            Traceback (most recent call last):
                ...
            ValueError: Unknown universe 'temp'...

            >>> Universe.remove_custom("SP500")  # Built-in
            Traceback (most recent call last):
                ...
            ValueError: Cannot remove built-in universe 'SP500'
        """
        normalized = name.upper().replace("-", "_").replace(" ", "_")

        # Prevent removal of built-in universes
        builtin = {"SP500", "NASDAQ100", "CRYPTO_TOP_100", "FOREX_MAJORS"}
        if normalized in builtin:
            raise ValueError(
                f"Cannot remove built-in universe '{normalized}'. Built-in universes are read-only."
            )

        if normalized not in cls._UNIVERSES:
            raise ValueError(f"Universe '{normalized}' does not exist")

        del cls._UNIVERSES[normalized]
