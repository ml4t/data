#!/usr/bin/env python3
"""
Daily Market Data Update Script

This script updates market data for a configured list of symbols.
It can be run manually or scheduled via cron for automated updates.

Usage:
    python daily_update.py
    python daily_update.py --symbols AAPL MSFT GOOGL
    python daily_update.py --config custom_config.yaml
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from ml4t_data import QLDM

from ml4t.data.validation import OHLCVValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("daily_update.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Default symbols to update
DEFAULT_SYMBOLS = [
    # US Stocks
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "JPM",
    "V",
    "JNJ",
    # Cryptocurrencies (if Binance is configured)
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
]


class DailyUpdater:
    """Handles daily market data updates."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize the updater.

        Args:
            config_path: Optional path to configuration file
        """
        self.qldm = QLDM(config_path=config_path) if config_path else QLDM()
        self.validator = OHLCVValidator()
        self.update_stats = {"success": 0, "failed": 0, "skipped": 0, "validation_failed": 0}

    def update_symbol(self, symbol: str) -> bool:
        """
        Update data for a single symbol.

        Args:
            symbol: Symbol to update

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Determine provider based on symbol format
            provider = "binance" if "/" in symbol else "yahoo"

            logger.info(f"Updating {symbol} from {provider}...")

            # Perform update
            result = self.qldm.update(symbol, provider=provider)

            if result:
                # Validate the updated data
                df = self.qldm.get(symbol)
                validation_result = self.validator.validate(df)

                if validation_result.passed:
                    logger.info(f"✅ {symbol}: Updated successfully, validation passed")
                    self.update_stats["success"] += 1
                    return True
                logger.warning(f"⚠️ {symbol}: Updated but validation failed")
                logger.warning(f"   Issues: {len(validation_result.issues)}")
                for issue in validation_result.issues[:3]:  # Show first 3 issues
                    logger.warning(f"   - {issue.severity}: {issue.message}")
                self.update_stats["validation_failed"] += 1
                return False
            logger.info(f"⏭️ {symbol}: Already up to date")
            self.update_stats["skipped"] += 1
            return True

        except Exception as e:
            logger.error(f"❌ {symbol}: Update failed - {e!s}")
            self.update_stats["failed"] += 1
            return False

    def update_all(self, symbols: list[str]) -> None:
        """
        Update all symbols in the list.

        Args:
            symbols: List of symbols to update
        """
        logger.info(f"Starting daily update for {len(symbols)} symbols")
        logger.info("=" * 60)

        start_time = datetime.now()

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}")
            self.update_symbol(symbol)
            logger.info("-" * 40)

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Print summary
        self.print_summary(duration)

    def print_summary(self, duration: float) -> None:
        """
        Print update summary.

        Args:
            duration: Update duration in seconds
        """
        logger.info("=" * 60)
        logger.info("UPDATE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successful: {self.update_stats['success']}")
        logger.info(f"⏭️ Skipped (up-to-date): {self.update_stats['skipped']}")
        logger.info(f"⚠️ Validation failed: {self.update_stats['validation_failed']}")
        logger.info(f"❌ Failed: {self.update_stats['failed']}")
        logger.info(f"⏱️ Duration: {duration:.1f} seconds")
        logger.info("=" * 60)

        # Set exit code based on results
        if self.update_stats["failed"] > 0:
            sys.exit(1)  # Exit with error if any updates failed


def validate_symbols(symbols: list[str]) -> list[str]:
    """
    Validate and clean symbol list.

    Args:
        symbols: List of symbols

    Returns:
        Cleaned list of symbols
    """
    # Remove duplicates and empty strings
    cleaned = list({s.strip().upper() for s in symbols if s.strip()})

    # Validate format
    valid_symbols = []
    for symbol in cleaned:
        if "/" in symbol:
            # Crypto pair format
            parts = symbol.split("/")
            if len(parts) == 2 and all(p.isalnum() for p in parts):
                valid_symbols.append(symbol)
            else:
                logger.warning(f"Invalid crypto pair format: {symbol}")
        else:
            # Stock symbol format
            if symbol.replace("-", "").replace(".", "").isalnum():
                valid_symbols.append(symbol)
            else:
                logger.warning(f"Invalid stock symbol format: {symbol}")

    return valid_symbols


def load_symbols_from_file(filepath: str) -> list[str]:
    """
    Load symbols from a text file (one per line).

    Args:
        filepath: Path to symbols file

    Returns:
        List of symbols
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Symbols file not found: {filepath}")

    with open(path) as f:
        symbols = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    return symbols


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update market data for configured symbols")
    parser.add_argument("--symbols", nargs="+", help="Symbols to update (overrides defaults)")
    parser.add_argument("--symbols-file", help="File containing symbols to update (one per line)")
    parser.add_argument("--config", help="Path to QLDM configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without actually updating",
    )

    args = parser.parse_args()

    # Determine symbols to update
    if args.symbols:
        symbols = validate_symbols(args.symbols)
    elif args.symbols_file:
        symbols = validate_symbols(load_symbols_from_file(args.symbols_file))
    else:
        symbols = DEFAULT_SYMBOLS

    if not symbols:
        logger.error("No valid symbols to update")
        sys.exit(1)

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual updates will be performed")
        logger.info(f"Would update {len(symbols)} symbols:")
        for symbol in symbols:
            logger.info(f"  - {symbol}")
        sys.exit(0)

    # Perform updates
    try:
        updater = DailyUpdater(config_path=args.config)
        updater.update_all(symbols)
    except KeyboardInterrupt:
        logger.info("\nUpdate interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
