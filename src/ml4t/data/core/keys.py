"""Storage key helpers.

Canonical storage key format:
    <asset_class>/<frequency>/<symbol>
"""

from __future__ import annotations

DEFAULT_ASSET_CLASS = "equities"
DEFAULT_FREQUENCY = "daily"


def is_storage_key(value: str) -> bool:
    """Return True when value is already a canonical storage key."""
    parts = value.split("/")
    return len(parts) == 3 and all(part.strip() for part in parts)


def build_storage_key(
    symbol: str,
    asset_class: str = DEFAULT_ASSET_CLASS,
    frequency: str = DEFAULT_FREQUENCY,
) -> str:
    """Build canonical storage key from parts."""
    symbol_clean = symbol.strip()
    asset_class_clean = asset_class.strip()
    frequency_clean = frequency.strip()

    if not symbol_clean:
        raise ValueError("symbol cannot be empty")
    if not asset_class_clean:
        raise ValueError("asset_class cannot be empty")
    if not frequency_clean:
        raise ValueError("frequency cannot be empty")

    return f"{asset_class_clean}/{frequency_clean}/{symbol_clean}"


def parse_storage_key(key: str) -> tuple[str, str, str]:
    """Parse canonical key into (asset_class, frequency, symbol)."""
    if not is_storage_key(key):
        raise ValueError(f"invalid storage key format: {key}")
    asset_class, frequency, symbol = key.split("/", 2)
    return asset_class, frequency, symbol


def normalize_storage_key(
    symbol_or_key: str,
    asset_class: str = DEFAULT_ASSET_CLASS,
    frequency: str = DEFAULT_FREQUENCY,
) -> str:
    """Normalize symbol or key into canonical storage key format."""
    value = symbol_or_key.strip()
    if not value:
        raise ValueError("symbol_or_key cannot be empty")

    if is_storage_key(value):
        parsed_asset_class, parsed_frequency, parsed_symbol = parse_storage_key(value)
        return build_storage_key(parsed_symbol, parsed_asset_class, parsed_frequency)

    return build_storage_key(value, asset_class, frequency)
