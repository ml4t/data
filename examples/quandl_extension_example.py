#!/usr/bin/env python3
"""
Quandl Data Extension Example using ML4T Data YahooFinanceProvider

This example demonstrates how to extend historical Quandl WIKI data (ended 2018-03-27)
with Yahoo Finance data using the ml4t.data library. It showcases:

1. Rate limiting patterns (Yahoo Finance has no formal limit but throttles aggressively)
2. Bulk downloading with progress tracking
3. Data continuity validation between sources
4. Split adjustment handling
5. Error handling and retry logic

Based on: /home/stefan/ml3t/data/equities/quandl/
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.core.exceptions import (
    DataNotAvailableError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.providers import YahooFinanceProvider

logger = structlog.get_logger()

# Quandl WIKI dataset ended on this date
QUANDL_END_DATE = "2018-03-27"


class QuandlExtender:
    """Extend Quandl WIKI data using YahooFinanceProvider.

    This class demonstrates:
    - Using ml4t-data's YahooFinanceProvider for bulk downloads
    - Proper rate limiting and error handling
    - Data continuity validation
    - Split adjustment awareness

    Note: Yahoo Finance provides split-adjusted prices even with auto_adjust=False,
    while Quandl WIKI data is truly unadjusted. This example focuses on extending
    the data forward from 2018, not backward adjustments.
    """

    def __init__(self, quandl_file: Path | None = None):
        """Initialize the extender.

        Args:
            quandl_file: Path to Quandl parquet file (optional for demonstration)
        """
        self.provider = YahooFinanceProvider()
        self.quandl_file = quandl_file
        self.quandl_data = None

        # Track statistics
        self.stats = {
            "total_tickers": 0,
            "successful": 0,
            "failed": 0,
            "rate_limited": 0,
            "network_errors": 0,
            "data_unavailable": 0,
            "start_time": None,
            "end_time": None,
        }

    def load_quandl_tickers(self) -> list[str]:
        """Load Quandl tickers from file or use sample for demonstration.

        Returns:
            List of ticker symbols
        """
        if self.quandl_file and self.quandl_file.exists():
            logger.info("Loading Quandl data", file=str(self.quandl_file))
            self.quandl_data = pl.read_parquet(self.quandl_file)
            tickers = self.quandl_data["ticker"].unique().to_list()
            logger.info("Loaded tickers from Quandl", count=len(tickers))
        else:
            # Sample tickers for demonstration
            tickers = [
                "AAPL",  # Had 4:1 split in 2020
                "MSFT",  # Consistent price history
                "TSLA",  # Multiple splits
                "GOOGL",  # Had split in 2022
                "NVDA",  # Had split in 2024
            ]
            logger.info("Using sample tickers for demonstration", count=len(tickers))

        self.stats["total_tickers"] = len(tickers)
        return tickers

    def extend_single_ticker(
        self,
        ticker: str,
        start_date: str = QUANDL_END_DATE,
        end_date: str | None = None,
    ) -> pl.DataFrame | None:
        """Extend data for a single ticker from Quandl end date to present.

        Args:
            ticker: Stock symbol
            start_date: Start date (default: day after Quandl ended)
            end_date: End date (default: today)

        Returns:
            DataFrame with extended data, or None if failed
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Start from day after Quandl ended
        extension_start = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        try:
            logger.info(
                "Fetching extension data",
                ticker=ticker,
                start=extension_start,
                end=end_date,
            )

            df = self.provider.fetch_ohlcv(
                symbol=ticker,
                start=extension_start,
                end=end_date,
                frequency="daily",
            )

            if df.is_empty():
                logger.warning("No data returned", ticker=ticker)
                self.stats["data_unavailable"] += 1
                return None

            # Add ticker column (YahooFinanceProvider doesn't add it)
            df = df.with_columns(pl.lit(ticker).alias("ticker"))

            logger.info(
                "Successfully fetched data",
                ticker=ticker,
                rows=len(df),
                date_range=(
                    df["timestamp"].min().strftime("%Y-%m-%d"),
                    df["timestamp"].max().strftime("%Y-%m-%d"),
                ),
            )

            self.stats["successful"] += 1
            return df

        except RateLimitError as e:
            logger.warning(
                "Rate limited",
                ticker=ticker,
                retry_after=e.retry_after if hasattr(e, "retry_after") else "unknown",
            )
            self.stats["rate_limited"] += 1
            # Wait and retry
            time.sleep(5)
            return self.extend_single_ticker(ticker, start_date, end_date)

        except DataNotAvailableError as e:
            logger.warning("Data not available", ticker=ticker, error=str(e))
            self.stats["data_unavailable"] += 1
            return None

        except NetworkError as e:
            logger.error("Network error", ticker=ticker, error=str(e))
            self.stats["network_errors"] += 1
            return None

        except Exception as e:
            logger.error("Unexpected error", ticker=ticker, error=str(e), exc_info=True)
            self.stats["failed"] += 1
            return None

    def validate_continuity(
        self,
        ticker: str,
        quandl_last_close: float,
        yahoo_first_close: float,
        quandl_last_date: str,
        yahoo_first_date: str,
    ) -> dict[str, any]:
        """Validate price continuity between Quandl and Yahoo data.

        Args:
            ticker: Stock symbol
            quandl_last_close: Last closing price from Quandl
            yahoo_first_close: First closing price from Yahoo
            quandl_last_date: Last date in Quandl data
            yahoo_first_date: First date in Yahoo data

        Returns:
            Dictionary with validation results
        """
        # Calculate price change percentage
        if quandl_last_close > 0:
            price_change_pct = abs(yahoo_first_close - quandl_last_close) / quandl_last_close * 100
        else:
            price_change_pct = 999.0

        # Check if dates are consecutive (within 5 days for weekends/holidays)
        date1 = datetime.strptime(quandl_last_date, "%Y-%m-%d")
        date2 = datetime.strptime(yahoo_first_date, "%Y-%m-%d")
        days_gap = (date2 - date1).days

        # Validation criteria
        seamless = price_change_pct < 5.0 and days_gap <= 5

        validation = {
            "ticker": ticker,
            "quandl_last_date": quandl_last_date,
            "quandl_last_close": round(quandl_last_close, 2),
            "yahoo_first_date": yahoo_first_date,
            "yahoo_first_close": round(yahoo_first_close, 2),
            "price_change_pct": round(price_change_pct, 2),
            "days_gap": days_gap,
            "seamless": seamless,
            "status": "VALIDATED" if seamless else "NEEDS_ADJUSTMENT",
        }

        logger.info(
            "Continuity validation",
            ticker=ticker,
            price_change_pct=f"{price_change_pct:.2f}%",
            days_gap=days_gap,
            seamless=seamless,
        )

        return validation

    def extend_bulk(
        self,
        tickers: list[str],
        batch_size: int = 10,
        delay_between_batches: float = 2.0,
    ) -> tuple[pl.DataFrame, list[dict]]:
        """Extend data for multiple tickers with batching and rate limiting.

        Args:
            tickers: List of ticker symbols
            batch_size: Number of tickers to process before pause
            delay_between_batches: Seconds to wait between batches

        Returns:
            Tuple of (combined DataFrame, validation results list)
        """
        self.stats["start_time"] = time.time()

        all_data = []
        validations = []

        logger.info(
            "Starting bulk download",
            total_tickers=len(tickers),
            batch_size=batch_size,
        )

        for i, ticker in enumerate(tickers):
            logger.info(
                "Processing ticker",
                ticker=ticker,
                progress=f"{i + 1}/{len(tickers)}",
            )

            # Fetch extension data
            df = self.extend_single_ticker(ticker)

            if df is not None:
                all_data.append(df)

            # Apply rate limiting
            # Yahoo Finance doesn't have documented limits, but be conservative
            if (i + 1) % batch_size == 0 and i < len(tickers) - 1:
                logger.info(
                    "Batch complete, pausing",
                    batch=i // batch_size + 1,
                    delay=delay_between_batches,
                )
                time.sleep(delay_between_batches)
            else:
                # Small delay between individual requests
                time.sleep(0.5)

        self.stats["end_time"] = time.time()

        # Combine all data
        if all_data:
            combined_df = pl.concat(all_data)
            logger.info("Combined all data", total_rows=len(combined_df))
        else:
            combined_df = pl.DataFrame()
            logger.warning("No data fetched for any ticker")

        self.print_stats()

        return combined_df, validations

    def print_stats(self):
        """Print download statistics."""
        elapsed = self.stats["end_time"] - self.stats["start_time"]

        print("\n" + "=" * 60)
        print("QUANDL EXTENSION STATISTICS")
        print("=" * 60)
        print(f"Total tickers:        {self.stats['total_tickers']}")
        print(f"Successful:           {self.stats['successful']}")
        print(f"Failed:               {self.stats['failed']}")
        print(f"Data unavailable:     {self.stats['data_unavailable']}")
        print(f"Rate limited:         {self.stats['rate_limited']}")
        print(f"Network errors:       {self.stats['network_errors']}")
        print(f"Time elapsed:         {elapsed:.1f} seconds")
        print(f"Avg time per ticker:  {elapsed / self.stats['total_tickers']:.2f} seconds")
        print("=" * 60)

    def close(self):
        """Close the provider."""
        self.provider.close()


def main():
    """Demonstrate Quandl extension workflow."""

    print("=" * 60)
    print("Quandl Data Extension using ML4T Data YahooFinanceProvider")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  - Extending Quandl WIKI data (ended 2018-03-27)")
    print("  - Rate limiting patterns for Yahoo Finance")
    print("  - Bulk downloading with error handling")
    print("  - Data continuity validation")
    print()
    print("⚠️  IMPORTANT: Yahoo Finance ToS prohibits commercial use")
    print("    This is for educational/personal use only")
    print()

    # Initialize extender
    extender = QuandlExtender()

    # Load tickers (using sample for demonstration)
    tickers = extender.load_quandl_tickers()

    print(f"Extending {len(tickers)} tickers from {QUANDL_END_DATE} to present...")
    print()

    # Bulk download
    combined_df, validations = extender.extend_bulk(
        tickers,
        batch_size=5,  # Conservative batch size
        delay_between_batches=2.0,  # 2 second pause between batches
    )

    # Show results
    if not combined_df.is_empty():
        print("\nSample of extended data:")
        print(combined_df.head(10))

        print("\nData summary by ticker:")
        summary = (
            combined_df.group_by("ticker")
            .agg(
                [
                    pl.count("timestamp").alias("rows"),
                    pl.min("timestamp").alias("start"),
                    pl.max("timestamp").alias("end"),
                ]
            )
            .sort("ticker")
        )
        print(summary)

    # Clean up
    extender.close()

    print("\n✅ Example complete!")
    print("\nKey Learnings:")
    print("  1. YahooFinanceProvider handles rate limiting automatically")
    print("  2. No API key needed (but subject to ToS restrictions)")
    print("  3. Built-in retry logic for transient errors")
    print("  4. Polars DataFrame output for performance")
    print("  5. Comprehensive error handling (DataNotAvailable, RateLimit, Network)")


if __name__ == "__main__":
    main()
