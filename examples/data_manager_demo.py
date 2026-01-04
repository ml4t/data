#!/usr/bin/env python
"""Demonstration of DataManager unified API for ML4T Data.

This example shows how to use the DataManager class to fetch data from
multiple providers transparently using a single, consistent interface.
"""

from __future__ import annotations

import os

from ml4t_data import DataManager


def main():
    """Demonstrate DataManager functionality."""

    # Example 1: Initialize with default configuration
    print("=" * 60)
    print("Example 1: Basic initialization")
    print("=" * 60)

    dm = DataManager()
    print(f"Available providers: {dm.list_providers()}")
    print()

    # Example 2: Initialize with YAML configuration
    print("=" * 60)
    print("Example 2: Configuration from YAML")
    print("=" * 60)

    # Create sample config file
    config_yaml = r"""
providers:
  cryptocompare:
    api_key: ${CRYPTOCOMPARE_API_KEY}
    rate_limit: 10
  databento:
    api_key: ${DATABENTO_API_KEY}
    dataset: GLBX.MDP3
  oanda:
    api_key: ${OANDA_API_KEY}
    account_id: practice_account

routing:
  patterns:
    - pattern: '^[A-Z]{6}$'
      provider: oanda
    - pattern: '^(BTC|ETH|SOL)'
      provider: cryptocompare
    - pattern: '^[A-Z]+\.(v|V)\.[0-9]+$'
      provider: databento

defaults:
  output_format: polars
  frequency: daily
"""

    # Save config temporarily
    with open("demo_config.yaml", "w") as f:
        f.write(config_yaml)

    try:
        dm_configured = DataManager(config_path="demo_config.yaml")
        print("Configuration loaded successfully")

        # Show provider info
        if "cryptocompare" in dm_configured.list_providers():
            info = dm_configured.get_provider_info("cryptocompare")
            print(f"CryptoCompare configured: {info}")
    finally:
        # Cleanup
        if os.path.exists("demo_config.yaml"):
            os.remove("demo_config.yaml")
    print()

    # Example 3: Fetch data with automatic provider routing
    print("=" * 60)
    print("Example 3: Automatic provider routing")
    print("=" * 60)

    # Set up some API keys for demonstration (mock values)
    os.environ["CRYPTOCOMPARE_API_KEY"] = "demo_key"

    dm = DataManager()

    # The routing will automatically determine the provider based on symbol
    symbols_to_test = [
        ("BTC", "cryptocompare"),  # Crypto -> CryptoCompare
        ("EURUSD", "oanda"),  # Forex -> OANDA
        ("ES.v.0", "databento"),  # Futures -> Databento
    ]

    for symbol, expected_provider in symbols_to_test:
        provider = dm.router.get_provider(symbol)
        print(f"{symbol:10} -> {provider:15} (expected: {expected_provider})")
    print()

    # Example 4: Fetch with different output formats
    print("=" * 60)
    print("Example 4: Output format conversion")
    print("=" * 60)

    # Create DataManager instances with different output formats
    dm_polars = DataManager(output_format="polars")
    dm_pandas = DataManager(output_format="pandas")
    dm_lazy = DataManager(output_format="lazy")

    print(f"Polars format: {dm_polars.output_format}")
    print(f"Pandas format: {dm_pandas.output_format}")
    print(f"Lazy format: {dm_lazy.output_format}")
    print()

    # Example 5: Batch fetching
    print("=" * 60)
    print("Example 5: Batch fetching multiple symbols")
    print("=" * 60)

    dm = DataManager()

    # Define symbols to fetch
    symbols = ["BTC", "ETH", "EURUSD"]
    start_date = "2024-01-01"
    end_date = "2024-01-01"

    print(f"Fetching batch: {symbols}")
    print(f"Date range: {start_date} to {end_date}")

    # Note: This would actually fetch data if providers were configured
    # results = dm.fetch_batch(symbols, start_date, end_date)
    print("(Batch fetch would execute here with configured providers)")
    print()

    # Example 6: Error handling
    print("=" * 60)
    print("Example 6: Error handling")
    print("=" * 60)

    dm = DataManager()

    try:
        # This will raise an error because no provider matches
        dm.fetch("UNKNOWN_SYMBOL_XYZ", "2024-01-01", "2024-01-01")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    print()

    # Example 7: Provider override
    print("=" * 60)
    print("Example 7: Manual provider override")
    print("=" * 60)

    dm = DataManager()

    # Normally BTC would route to cryptocompare
    normal_provider = dm.router.get_provider("BTC")
    print(f"Normal routing for BTC: {normal_provider}")

    # But we can override it
    override_provider = dm.router.get_provider("BTC", override="oanda")
    print(f"Override routing for BTC: {override_provider}")
    print()

    # Example 8: Configuration hierarchy
    print("=" * 60)
    print("Example 8: Configuration hierarchy")
    print("=" * 60)

    print("Configuration precedence:")
    print("1. Function parameters (highest priority)")
    print("2. Environment variables")
    print("3. YAML configuration file")
    print("4. Default values (lowest priority)")
    print()

    # Clean up
    os.environ.pop("CRYPTOCOMPARE_API_KEY", None)

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
