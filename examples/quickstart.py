"""ML4T Data 5-Minute Quickstart

This script demonstrates the simplest way to get started with ML4T Data.
No configuration needed - just run and get Bitcoin data!

Requirements:
    - ml4t-data installed: pip install ml4t-data
    - No API key needed (uses CoinGecko free tier)
    - Internet connection

Usage:
    python examples/quickstart.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.providers import CoinGeckoProvider


def main():
    """Run the 5-minute quickstart example."""
    print("=" * 80)
    print("  ML4T Data 5-Minute Quickstart")
    print("  Get Bitcoin historical data with 3 lines of code!")
    print("=" * 80)
    print()

    # Calculate date range (last 30 days)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"📈 Fetching Bitcoin data from {start_date} to {end_date}...")
    print()

    # THE QUICKSTART CODE (3 lines!)
    # ================================
    provider = CoinGeckoProvider()  # No API key needed!
    btc_data = provider.fetch_ohlcv("bitcoin", start_date, end_date)
    print(btc_data.head())
    # ================================

    print()
    print(f"✅ Successfully fetched {len(btc_data)} days of Bitcoin data!")
    print()

    # Show some basic statistics
    print("=" * 80)
    print("  Bitcoin Statistics (Last 30 Days)")
    print("=" * 80)
    print()
    print(f"  Opening Price:  ${btc_data['open'][0]:,.2f}")
    print(f"  Closing Price:  ${btc_data['close'][-1]:,.2f}")
    print(f"  Highest Price:  ${btc_data['high'].max():,.2f}")
    print(f"  Lowest Price:   ${btc_data['low'].min():,.2f}")
    print(f"  Total Volume:   {btc_data['volume'].sum():,.0f}")
    print()

    # Calculate price change
    price_change = btc_data["close"][-1] - btc_data["open"][0]
    price_change_pct = (price_change / btc_data["open"][0]) * 100

    if price_change > 0:
        emoji = "📈"
        direction = "UP"
    else:
        emoji = "📉"
        direction = "DOWN"

    print(
        f"  Price Change:   {emoji} ${price_change:,.2f} ({direction} {abs(price_change_pct):.2f}%)"
    )
    print()

    # Show what you can do next
    print("=" * 80)
    print("  What's Next?")
    print("=" * 80)
    print()
    print("  ✅ You just fetched real market data with 3 lines of code!")
    print()
    print("  Want more? Here are your next steps:")
    print()
    print("  1. Try other cryptocurrencies:")
    print("     provider.fetch_ohlcv('ethereum', start_date, end_date)")
    print("     provider.fetch_ohlcv('cardano', start_date, end_date)")
    print()
    print("  2. Explore other providers (stocks, forex, futures):")
    print("     See: README.md → 'Choose Your Provider' section")
    print()
    print("  3. Save data for later (incremental updates):")
    print("     See: README.md → 'Incremental Updates' section")
    print()
    print("  4. Build a trading strategy:")
    print("     See: docs/tutorials/ for step-by-step guides")
    print()

    # Clean up
    provider.close()

    print("=" * 80)
    print("  Done! Happy trading! 🚀")
    print("=" * 80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as err:
        print(f"\n\n❌ Error: {err}")
        print()
        print("Common issues:")
        print("  - Check internet connection")
        print("  - Verify ml4t-data is installed: pip install ml4t-data")
        print("  - Check CoinGecko API status: https://status.coingecko.com/")
        print()
        sys.exit(1)
