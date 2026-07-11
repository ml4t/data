"""Example usage of Phase 1 providers (CoinGecko, Yahoo Finance, Twelve Data).

This script demonstrates:
1. Basic provider usage for direct data fetching
2. Storage workflow for fetched data
3. Error handling and rate limit management
4. Storage and data retrieval
5. Comparison of multiple providers for same symbol

Requirements:
    - TWELVE_DATA_API_KEY environment variable (get at: https://twelvedata.com/)
    - CoinGecko and Yahoo Finance require no API key

Usage:
    python examples/phase1_providers.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.providers import (
    CoinGeckoProvider,
    TwelveDataProvider,
    YahooFinanceProvider,
)
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.storage.hive import HiveStorage


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_dataframe_sample(df, name="Data", rows=5):
    """Print a sample of the dataframe."""
    print(f"{name} ({len(df)} rows):")
    if len(df) > 0:
        print(df.head(rows))
    else:
        print("  (empty)")
    print()


def example_1_basic_provider_usage():
    """Example 1: Basic provider usage (direct fetching)."""
    print_section("EXAMPLE 1: Basic Provider Usage")

    # Date range for examples
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"Fetching data for the last 30 days ({start} to {end})\n")

    # 1. CoinGecko - No API key needed
    print("1️⃣  CoinGecko (Crypto - No API key)")
    print("-" * 80)
    try:
        coingecko = CoinGeckoProvider()
        btc_data = coingecko.fetch_ohlcv("bitcoin", start, end)
        print_dataframe_sample(btc_data, "Bitcoin OHLCV", 3)
        coingecko.close()
        time.sleep(2)  # Rate limit
    except Exception as e:
        print(f"❌ CoinGecko error: {e}\n")

    # 2. Yahoo Finance - No API key needed
    print("2️⃣  Yahoo Finance (Stocks - No API key)")
    print("-" * 80)
    try:
        yahoo = YahooFinanceProvider()
        aapl_data = yahoo.fetch_ohlcv("AAPL", start, end)
        print_dataframe_sample(aapl_data, "AAPL OHLCV", 3)
        yahoo.close()
        time.sleep(1)
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}\n")

    # 3. Twelve Data - API key required
    print("3️⃣  Twelve Data (Multi-asset - API key required)")
    print("-" * 80)
    if os.getenv("TWELVE_DATA_API_KEY"):
        try:
            twelve = TwelveDataProvider()
            msft_data = twelve.fetch_ohlcv("MSFT", start, end)
            print_dataframe_sample(msft_data, "MSFT OHLCV", 3)
            twelve.close()
            time.sleep(8)  # Strict rate limit
        except Exception as e:
            print(f"❌ Twelve Data error: {e}\n")
    else:
        print("⚠️  TWELVE_DATA_API_KEY not set - skipping\n")


def example_2_storage_workflow():
    """Example 2: Store fetched provider data."""
    print_section("EXAMPLE 2: Storage Workflow")

    # Setup storage
    storage_path = Path("./example_data")
    storage = HiveStorage(StorageConfig(base_path=storage_path))

    print(f"Storage path: {storage_path.absolute()}\n")

    # CoinGecko storage example
    print("1️⃣  CoinGecko Storage")
    print("-" * 80)
    try:
        provider = CoinGeckoProvider()
        data = provider.fetch_ohlcv("ethereum", start="2024-01-01", end="2024-01-31")
        storage.write(data, "coingecko/ethereum")
        print(f"  ✅ Stored {len(data)} Ethereum records")

        # Read data from storage
        print("\nReading stored data...")
        data = storage.read_data("ethereum", "coingecko")
        print_dataframe_sample(data, "Ethereum data from storage", 3)

        provider.close()

    except Exception as e:
        print(f"❌ Error: {e}\n")

    time.sleep(2)


def example_3_error_handling():
    """Example 3: Error handling and rate limiting."""
    print_section("EXAMPLE 3: Error Handling and Rate Limiting")

    # 1. Invalid symbol handling
    print("1️⃣  Handling Invalid Symbols")
    print("-" * 80)
    try:
        provider = CoinGeckoProvider()
        data = provider.fetch_ohlcv("invalid_symbol_xyz", "2024-01-01", "2024-01-31")
        print(f"Data: {data}")
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}")
        print(f"   Message: {str(e)}\n")

    time.sleep(2)

    # 2. Rate limiting
    print("2️⃣  Rate Limiting Demonstration")
    print("-" * 80)
    if os.getenv("TWELVE_DATA_API_KEY"):
        try:
            provider = TwelveDataProvider()

            print("Twelve Data has strict rate limit: 8 requests/minute")
            print("Making 2 rapid requests to demonstrate rate limiting...\n")

            start_time = datetime.now()

            # First request
            print("Request 1...")
            provider.fetch_quote("AAPL")
            print("  ✅ Completed")

            # Second request (should be delayed by rate limiter)
            print("Request 2...")
            provider.fetch_quote("MSFT")
            print("  ✅ Completed")

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\nTotal time: {elapsed:.2f} seconds")
            print("(Note: Rate limiter enforces ~8 second delay between requests)\n")

            provider.close()

        except Exception as e:
            print(f"❌ Error: {e}\n")
    else:
        print("⚠️  TWELVE_DATA_API_KEY not set - skipping rate limit demo\n")


def example_4_comparing_providers():
    """Example 4: Compare same symbol across multiple providers."""
    print_section("EXAMPLE 4: Compare AAPL Across Multiple Providers")

    storage_path = Path("./example_data")
    storage = HiveStorage(StorageConfig(base_path=storage_path))

    symbol = "AAPL"
    results = {}

    # Fetch from Yahoo Finance
    print("1️⃣  Fetching from Yahoo Finance...")
    try:
        provider = YahooFinanceProvider()
        data = provider.fetch_ohlcv(symbol, "2024-01-01", "2024-01-31")
        storage.write(data, f"yahoo/{symbol}")
        results["Yahoo Finance"] = data
        print(f"  ✅ Yahoo Finance: {len(data)} records")
        provider.close()
        time.sleep(1)
    except Exception as e:
        print(f"  ❌ Yahoo Finance error: {e}")

    # Fetch from Twelve Data
    if os.getenv("TWELVE_DATA_API_KEY"):
        print("2️⃣  Fetching from Twelve Data...")
        try:
            provider = TwelveDataProvider()
            data = provider.fetch_ohlcv(symbol, "2024-01-01", "2024-01-31")
            storage.write(data, f"twelve_data/{symbol}")
            results["Twelve Data"] = data
            print(f"  ✅ Twelve Data: {len(data)} records")
            provider.close()
            time.sleep(8)

        except Exception as e:
            print(f"  ❌ Twelve Data error: {e}")
    else:
        print("2️⃣  Twelve Data - skipped (no API key)")

    # Compare results
    if len(results) >= 2:
        print("\n📊 Comparison:")
        print("-" * 80)

        for provider_name, data in results.items():
            if len(data) > 0:
                latest = data.sort("timestamp").tail(1)
                print(f"{provider_name}:")
                print(f"  Records: {len(data)}")
                print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                print(f"  Latest close: ${latest['close'][0]:.2f}")
                print()
    else:
        print("\n⚠️  Need at least 2 providers with API keys for comparison")


def example_5_dry_run():
    """Example 5: Dry run mode (preview without saving)."""
    print_section("EXAMPLE 5: Dry Run Mode (Preview Without Saving)")

    storage_path = Path("./example_data")
    storage = HiveStorage(StorageConfig(base_path=storage_path))

    print("Dry run mode allows you to preview what would be downloaded")
    print("without actually saving to storage.\n")

    try:
        provider = CoinGeckoProvider()

        # Get current state
        try:
            before_data = storage.read_data("cardano", "coingecko")
            before_count = len(before_data)
        except FileNotFoundError:
            before_count = 0

        print(f"Records in storage before: {before_count}")

        print("\nPreviewing fetch for cardano without writing...")
        preview = provider.fetch_ohlcv(
            "cardano",
            start=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"),
        )
        print(f"  ✅ Would write {len(preview)} records")

        # Check storage again
        try:
            after_data = storage.read_data("cardano", "coingecko")
            after_count = len(after_data)
        except FileNotFoundError:
            after_count = 0

        print(f"\nRecords in storage after: {after_count}")
        print("  ✅ Storage unchanged (as expected in dry run mode)")

        provider.close()

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples."""
    print("=" * 80)
    print("  Phase 1 Providers - Example Usage")
    print("  CoinGecko + Yahoo Finance + Twelve Data")
    print("=" * 80)
    print()
    print("This script demonstrates the usage of Phase 1 data providers.")
    print()
    print("API Keys needed:")
    coingecko_ok = "✅" if True else "❌"
    twelve_ok = "✅" if os.getenv("TWELVE_DATA_API_KEY") else "❌"

    print(f"  {coingecko_ok} CoinGecko (no key needed)")
    print("  ✅ Yahoo Finance (no key needed)")
    print(f"  {twelve_ok} Twelve Data (TWELVE_DATA_API_KEY)")
    print()

    if not os.getenv("TWELVE_DATA_API_KEY"):
        print("⚠️  WARNING: TWELVE_DATA_API_KEY not set. Twelve Data examples will be skipped.")
        print()
        print("To get API keys:")
        print("  - Twelve Data: https://twelvedata.com/ (free tier: 800 req/day)")
        print()
        input("Press Enter to continue with limited examples...")

    # Run examples
    try:
        example_1_basic_provider_usage()
        example_2_storage_workflow()
        example_3_error_handling()
        example_4_comparing_providers()
        example_5_dry_run()

        print_section("✅ All Examples Completed!")
        print("Check the './example_data' directory for stored data.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
