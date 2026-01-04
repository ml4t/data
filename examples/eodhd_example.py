"""Example usage of EODHD provider for global equities data.

This script demonstrates:
1. Basic provider usage for US stocks
2. Global exchange support (US, London, Frankfurt)
3. Multiple frequency support (daily, weekly, monthly)
4. Error handling and rate limit management
5. Provider updater pattern for incremental updates

Requirements:
    - EODHD_API_KEY environment variable (get free key at: https://eodhd.com/register)
    - Free tier: 500 calls per day, 1 year historical depth
    - Paid tier: €19.99/month for all-world EOD data

Usage:
    # Set API key
    export EODHD_API_KEY=your_key_here

    # Run example
    python examples/eodhd_example.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.providers import EODHDProvider, EODHDUpdater
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
        if "timestamp" in df.columns:
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print("  (empty)")
    print()


def example_1_us_stocks():
    """Example 1: Basic US stock data fetching."""
    print_section("EXAMPLE 1: US Stock Data Fetching")

    # Check for API key
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("❌ EODHD_API_KEY not set")
        print("   Get free key at: https://eodhd.com/register")
        print("   Then: export EODHD_API_KEY=your_key_here\n")
        return

    # Date range (last 30 days)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"Fetching AAPL data for the last 30 days ({start} to {end})\n")

    # Create provider with default US exchange
    provider = EODHDProvider(api_key=api_key, exchange="US")

    try:
        # Fetch daily data (exchange defaults to US)
        print("📈 Fetching daily OHLCV data for AAPL...")
        df = provider.fetch_ohlcv("AAPL", start, end, frequency="daily")
        print_dataframe_sample(df, "AAPL Daily OHLCV", 5)

        print("✅ Successfully fetched US stock data\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    finally:
        provider.close()


def example_2_global_exchanges():
    """Example 2: Fetching data from multiple global exchanges."""
    print_section("EXAMPLE 2: Global Exchange Support")

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("❌ EODHD_API_KEY not set\n")
        return

    # Date range
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    provider = EODHDProvider(api_key=api_key)

    # Stocks from different exchanges
    stocks = [
        ("AAPL", "US", "Apple Inc (US)"),
        ("VOD", "LSE", "Vodafone Group (London)"),
        ("BMW", "FRA", "BMW (Frankfurt)"),
    ]

    try:
        for symbol, exchange, name in stocks:
            print(f"📊 Fetching {name} from {exchange} exchange...")
            df = provider.fetch_ohlcv(
                symbol=symbol,
                start=start,
                end=end,
                frequency="daily",
                exchange=exchange,
            )

            if len(df) > 0:
                print(f"   Symbol: {symbol}.{exchange}")
                print(f"   Records: {len(df)}")
                print(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
                print(f"   Last close: {df['close'][-1]:.2f}")
                print()

            # Rate limiting: be conservative with free tier
            time.sleep(2)

        print("✅ Successfully fetched data from multiple exchanges\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    finally:
        provider.close()


def example_3_multiple_frequencies():
    """Example 3: Fetching data with different frequencies."""
    print_section("EXAMPLE 3: Multiple Frequency Support")

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("❌ EODHD_API_KEY not set\n")
        return

    provider = EODHDProvider(api_key=api_key)

    try:
        symbol = "AAPL"

        # Daily data (last 30 days)
        print(f"📈 Fetching {symbol} at different frequencies...\n")

        end = datetime.now().strftime("%Y-%m-%d")

        # Daily
        start_daily = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        df_daily = provider.fetch_ohlcv(symbol, start_daily, end, frequency="daily")
        print(f"Daily: {len(df_daily)} records")
        print(f"  Date range: {df_daily['timestamp'].min()} to {df_daily['timestamp'].max()}")

        time.sleep(2)  # Rate limit

        # Weekly (last 90 days)
        start_weekly = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        df_weekly = provider.fetch_ohlcv(symbol, start_weekly, end, frequency="weekly")
        print(f"\nWeekly: {len(df_weekly)} records")
        print(f"  Date range: {df_weekly['timestamp'].min()} to {df_weekly['timestamp'].max()}")

        time.sleep(2)  # Rate limit

        # Monthly (last 365 days)
        start_monthly = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        df_monthly = provider.fetch_ohlcv(symbol, start_monthly, end, frequency="monthly")
        print(f"\nMonthly: {len(df_monthly)} records")
        print(f"  Date range: {df_monthly['timestamp'].min()} to {df_monthly['timestamp'].max()}")

        print("\n✅ Successfully fetched data at multiple frequencies\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    finally:
        provider.close()


def example_4_error_handling():
    """Example 4: Error handling."""
    print_section("EXAMPLE 4: Error Handling")

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("❌ EODHD_API_KEY not set\n")
        return

    provider = EODHDProvider(api_key=api_key)

    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Test invalid symbol
        print("Testing error handling with invalid symbol...\n")
        try:
            _ = provider.fetch_ohlcv(
                symbol="INVALID_SYMBOL_XYZ",
                start=start,
                end=end,
                frequency="daily",
                exchange="US",
            )
            print("⚠️  Expected error but got data (unexpected)")
        except Exception as e:
            print(f"✅ Correctly handled invalid symbol: {type(e).__name__}")
            print(f"   Message: {str(e)[:80]}...")

        print()

        # Test invalid exchange
        print("Testing error handling with invalid exchange...\n")
        try:
            _df = provider.fetch_ohlcv(
                symbol="AAPL",
                start=start,
                end=end,
                frequency="daily",
                exchange="INVALID_EXCHANGE",
            )
            print("⚠️  Expected error but got data (unexpected)")
        except Exception as e:
            print(f"✅ Correctly handled invalid exchange: {type(e).__name__}")
            print(f"   Message: {str(e)[:80]}...")

        print()

    except Exception as e:
        print(f"❌ Unexpected error: {e}\n")

    finally:
        provider.close()


def example_5_incremental_updates():
    """Example 5: Incremental updates with storage."""
    print_section("EXAMPLE 5: Incremental Updates with Storage")

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("❌ EODHD_API_KEY not set\n")
        return

    # Create temporary storage
    storage_path = Path(__file__).parent / "temp_data" / "eodhd_example"
    storage_path.mkdir(parents=True, exist_ok=True)

    print(f"Using storage path: {storage_path}\n")

    # Create provider and storage
    provider = EODHDProvider(api_key=api_key)
    storage = HiveStorage(StorageConfig(base_path=str(storage_path)))

    # Create updater
    updater = EODHDUpdater(provider, storage)

    try:
        symbols = ["AAPL", "MSFT"]

        print(f"Updating {len(symbols)} symbols with incremental pattern...\n")

        for symbol in symbols:
            print(f"📈 Updating {symbol}...")

            # Update with last 30 days
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            result = updater.update_symbol(
                symbol=symbol,
                start_time=start,
                end_time=end,
                frequency="daily",
                exchange="US",
                dry_run=False,
            )

            if result["success"]:
                print(f"   ✅ Updated: {result['records_fetched']} records")
                print(f"   Date range: {result['start_date']} to {result['end_date']}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

            print()

            # Rate limiting
            time.sleep(2)

        print("✅ Successfully updated symbols with incremental pattern\n")
        print(f"Data stored at: {storage_path}")

    except Exception as e:
        print(f"❌ Error: {e}\n")

    finally:
        provider.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("  EODHD Provider Examples")
    print("  Global Equities Data - 60+ Exchanges, 150,000+ Tickers")
    print("=" * 80)

    # Check if API key is set
    if not os.getenv("EODHD_API_KEY"):
        print("\n⚠️  EODHD_API_KEY not set!")
        print("   Get free key at: https://eodhd.com/register")
        print("   Then: export EODHD_API_KEY=your_key_here\n")
        return

    # Run examples
    example_1_us_stocks()
    example_2_global_exchanges()
    example_3_multiple_frequencies()
    example_4_error_handling()
    example_5_incremental_updates()

    print("\n" + "=" * 80)
    print("  All Examples Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
