"""Example usage of Finnhub provider.

This script demonstrates:
1. Basic provider usage for fetching OHLCV data
2. Real-time quote fetching
3. Multiple frequency support (daily, weekly, monthly)
4. Error handling and rate limit management
5. Provider updater pattern for incremental updates

Requirements:
    - FINNHUB_API_KEY environment variable (get free key at: https://finnhub.io/register)
    - Free tier: 60 calls per minute, no daily limit

Usage:
    # Set API key
    export FINNHUB_API_KEY=your_key_here

    # Run example
    python examples/finnhub_example.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.providers import FinnhubProvider, FinnhubUpdater
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
        print()
        # Print summary stats
        if "close" in df.columns:
            print(f"  Close range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        if "volume" in df.columns:
            print(f"  Total volume: {df['volume'].sum():,.0f}")
    else:
        print("  (empty)")
    print()


def example_1_basic_ohlcv_fetching():
    """Example 1: Basic OHLCV data fetching."""
    print_section("EXAMPLE 1: Basic OHLCV Data Fetching")

    # Check for API key
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("❌ FINNHUB_API_KEY not set")
        print("   Get free key at: https://finnhub.io/register")
        print("   Then: export FINNHUB_API_KEY=your_key_here\n")
        return

    # Date range
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"Fetching AAPL data for the last 30 days ({start} to {end})\n")

    # Create provider
    provider = FinnhubProvider(api_key=api_key)

    try:
        # Fetch daily data
        print("📈 Fetching daily OHLCV data...")
        df = provider.fetch_ohlcv("AAPL", start, end, frequency="daily")
        print_dataframe_sample(df, "AAPL Daily OHLCV", 5)

        print("✅ Successfully fetched daily data\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    finally:
        provider.close()


def example_2_real_time_quotes():
    """Example 2: Real-time quote fetching."""
    print_section("EXAMPLE 2: Real-Time Quote Fetching")

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("❌ FINNHUB_API_KEY not set\n")
        return

    provider = FinnhubProvider(api_key=api_key)

    try:
        symbols = ["AAPL", "MSFT", "GOOGL"]
        print(f"Fetching real-time quotes for: {', '.join(symbols)}\n")

        for symbol in symbols:
            quote = provider.fetch_quote(symbol)
            if len(quote) > 0:
                current = quote["current"][0]
                change = quote["change"][0]
                pct_change = quote["percent_change"][0]
                day_high = quote["high"][0]
                day_low = quote["low"][0]

                print(f"📊 {symbol}")
                print(f"   Current: ${current:.2f}")
                print(f"   Change: ${change:+.2f} ({pct_change:+.2f}%)")
                print(f"   Day range: ${day_low:.2f} - ${day_high:.2f}")
                print()

            # Small delay to respect rate limits
            time.sleep(1)

        print("✅ Successfully fetched quotes\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    finally:
        provider.close()


def example_3_multiple_frequencies():
    """Example 3: Fetching data with different frequencies."""
    print_section("EXAMPLE 3: Multiple Frequency Support")

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("❌ FINNHUB_API_KEY not set\n")
        return

    provider = FinnhubProvider(api_key=api_key)

    try:
        symbol = "AAPL"

        # Daily data (30 days)
        print("1️⃣  Daily Data (30 days)")
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        df_daily = provider.fetch_ohlcv(symbol, start, end, frequency="daily")
        print(f"   Fetched {len(df_daily)} rows")
        print(f"   Date range: {df_daily['timestamp'].min()} to {df_daily['timestamp'].max()}")
        print()

        time.sleep(1)  # Rate limit

        # Weekly data (90 days)
        print("2️⃣  Weekly Data (90 days)")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        df_weekly = provider.fetch_ohlcv(symbol, start, end, frequency="weekly")
        print(f"   Fetched {len(df_weekly)} rows")
        print(f"   Date range: {df_weekly['timestamp'].min()} to {df_weekly['timestamp'].max()}")
        print()

        time.sleep(1)  # Rate limit

        # Monthly data (180 days)
        print("3️⃣  Monthly Data (180 days)")
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        df_monthly = provider.fetch_ohlcv(symbol, start, end, frequency="monthly")
        print(f"   Fetched {len(df_monthly)} rows")
        print(f"   Date range: {df_monthly['timestamp'].min()} to {df_monthly['timestamp'].max()}")
        print()

        print("✅ Successfully fetched multiple frequencies\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    finally:
        provider.close()


def example_4_error_handling():
    """Example 4: Error handling examples."""
    print_section("EXAMPLE 4: Error Handling")

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("❌ FINNHUB_API_KEY not set\n")
        return

    provider = FinnhubProvider(api_key=api_key)

    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Test invalid symbol
        print("1️⃣  Testing invalid symbol...")
        try:
            provider.fetch_ohlcv("INVALID_SYMBOL_12345", start, end)
            print("   ⚠️  Unexpected: Should have raised DataNotAvailableError")
        except Exception as e:
            print(f"   ✅ Correctly handled: {type(e).__name__}")
            print()

        time.sleep(1)

        # Test invalid frequency
        print("2️⃣  Testing invalid frequency...")
        try:
            provider.fetch_ohlcv("AAPL", start, end, frequency="invalid")
            print("   ⚠️  Unexpected: Should have raised DataValidationError")
        except Exception as e:
            print(f"   ✅ Correctly handled: {type(e).__name__}")
            print()

        print("✅ Error handling working correctly\n")

    except Exception as e:
        print(f"❌ Unexpected error: {e}\n")

    finally:
        provider.close()


def example_5_incremental_updates():
    """Example 5: Using the updater for incremental updates."""
    print_section("EXAMPLE 5: Incremental Updates with Storage")

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("❌ FINNHUB_API_KEY not set\n")
        return

    # Set up storage
    storage_path = Path(__file__).parent.parent / "data" / "examples" / "finnhub"
    storage_config = StorageConfig(base_path=str(storage_path))
    storage = HiveStorage(storage_config)

    print(f"Storage location: {storage_path}\n")

    # Create provider and updater
    provider = FinnhubProvider(api_key=api_key)
    updater = FinnhubUpdater(provider, storage)

    try:
        symbol = "AAPL"
        print(f"Updating {symbol} data...\n")

        # Update symbol (defaults to last 365 days)
        result = updater.update_symbol(symbol, frequency="daily")

        if result["success"]:
            print("✅ Update successful")
            print(f"   Records fetched: {result['records_fetched']}")
            print(f"   Date range: {result['start_date']} to {result['end_date']}")
            print()

            # Read back from storage
            print("Reading data back from storage...")
            stored_data = storage.read(symbol, "finnhub")
            print(f"   Stored records: {len(stored_data)}")
            print()

        else:
            print(f"❌ Update failed: {result.get('error')}\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    finally:
        provider.close()

    print("💡 Tip: Run this example multiple times to see incremental updates\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("  FINNHUB PROVIDER EXAMPLES")
    print("=" * 80)

    # Check for API key first
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("\n⚠️  FINNHUB_API_KEY environment variable not set")
        print("\n📝 To run these examples:")
        print("   1. Get a free API key at: https://finnhub.io/register")
        print("   2. Set environment variable: export FINNHUB_API_KEY=your_key_here")
        print("   3. Run this script again\n")
        return

    print("\n✅ API key found - running examples...\n")

    # Run examples
    example_1_basic_ohlcv_fetching()
    example_2_real_time_quotes()
    example_3_multiple_frequencies()
    example_4_error_handling()
    example_5_incremental_updates()

    print_section("🎉 All examples completed!")
    print("Next steps:")
    print("  - Try different symbols (stocks, ETFs)")
    print("  - Experiment with different frequencies")
    print("  - Integrate with your own analysis pipeline")
    print("  - See documentation: https://finnhub.io/docs/api")
    print()


if __name__ == "__main__":
    main()
