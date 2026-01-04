#!/usr/bin/env python3
"""
Stress Test: ML4T Data vs Direct yfinance

Test with 200+ tickers to determine if the 17x slowdown provides real reliability
benefits or is just poor performance.

This will answer:
1. Does ML4T Data handle rate limiting better at scale?
2. What's the actual failure rate for each approach?
3. Is the performance hit worth it?
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.core.exceptions import (
    DataNotAvailableError,
    NetworkError,
    RateLimitError,
)
from ml4t.data.providers import YahooFinanceProvider

logger = structlog.get_logger()

# Load tickers from Quandl dataset
QUANDL_TICKERS_FILE = (
    "/home/stefan/ml3t/data/equities/quandl/FINAL_US_EQUITY_DATA/quandl_original/wiki_tickers.csv"
)


def load_quandl_tickers(limit: int = 200) -> list[str]:
    """Load first N tickers from Quandl dataset."""
    tickers = []
    with open(QUANDL_TICKERS_FILE) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(tickers) >= limit:
                break
            tickers.append(row[0])
    return tickers


def test_direct_yfinance(tickers: list[str], start: str, end: str) -> dict:
    """Test direct yfinance approach (ml3t style)."""
    stats = {
        "approach": "direct_yfinance",
        "total": len(tickers),
        "successful": 0,
        "failed": 0,
        "empty": 0,
        "errors": 0,
        "start_time": time.time(),
        "end_time": None,
        "failures": [],
    }

    logger.info("Starting direct yfinance test", tickers=len(tickers))

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info("Progress", completed=i + 1, total=len(tickers))

        try:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
            )

            if hist.empty:
                stats["empty"] += 1
                stats["failures"].append((ticker, "empty"))
            else:
                stats["successful"] += 1

            # ml3t uses 100ms delay
            time.sleep(0.1)

        except Exception as e:
            stats["errors"] += 1
            stats["failures"].append((ticker, str(e)[:50]))
            logger.warning("Error", ticker=ticker, error=str(e)[:100])

    stats["failed"] = stats["empty"] + stats["errors"]
    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    stats["avg_per_ticker"] = stats["duration"] / stats["total"]

    return stats


def test_ml4t_data_provider(tickers: list[str], start: str, end: str) -> dict:
    """Test ML4T Data YahooFinanceProvider approach."""
    stats = {
        "approach": "ml4t-data",
        "total": len(tickers),
        "successful": 0,
        "failed": 0,
        "data_unavailable": 0,
        "rate_limited": 0,
        "network_errors": 0,
        "start_time": time.time(),
        "end_time": None,
        "failures": [],
    }

    provider = YahooFinanceProvider(max_requests_per_second=0.5)

    logger.info("Starting ML4T Data test", tickers=len(tickers))

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info("Progress", completed=i + 1, total=len(tickers))

        try:
            df = provider.fetch_ohlcv(
                symbol=ticker,
                start=start,
                end=end,
                frequency="daily",
            )

            if df.is_empty():
                stats["data_unavailable"] += 1
                stats["failures"].append((ticker, "empty"))
            else:
                stats["successful"] += 1

        except DataNotAvailableError:
            stats["data_unavailable"] += 1
            stats["failures"].append((ticker, "data_unavailable"))

        except RateLimitError:
            stats["rate_limited"] += 1
            stats["failures"].append((ticker, "rate_limited"))
            logger.warning("Rate limited", ticker=ticker)

        except NetworkError:
            stats["network_errors"] += 1
            stats["failures"].append((ticker, "network_error"))
            logger.warning("Network error", ticker=ticker)

        except Exception as e:
            stats["failed"] += 1
            stats["failures"].append((ticker, str(e)[:50]))
            logger.error("Unexpected error", ticker=ticker, error=str(e)[:100])

    provider.close()

    stats["failed"] = stats["data_unavailable"] + stats["rate_limited"] + stats["network_errors"]
    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    stats["avg_per_ticker"] = stats["duration"] / stats["total"]

    return stats


def print_comparison(stats_yfinance: dict, stats_ml4t_data: dict):
    """Print side-by-side comparison."""
    print("\n" + "=" * 80)
    print("STRESS TEST RESULTS: 200 TICKERS")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Direct yfinance':<20} {'ML4T Data':<20}")
    print("-" * 70)

    print(f"{'Total tickers':<30} {stats_yfinance['total']:<20} {stats_ml4t_data['total']:<20}")
    print(
        f"{'Successful':<30} {stats_yfinance['successful']:<20} {stats_ml4t_data['successful']:<20}"
    )
    print(f"{'Failed':<30} {stats_yfinance['failed']:<20} {stats_ml4t_data['failed']:<20}")
    print(
        f"{'Success rate':<30} {stats_yfinance['successful'] / stats_yfinance['total'] * 100:.1f}%{'':<15} {stats_ml4t_data['successful'] / stats_ml4t_data['total'] * 100:.1f}%"
    )

    print("\n" + "-" * 70)
    print(
        f"{'Total time':<30} {stats_yfinance['duration']:.1f}s{'':<15} {stats_ml4t_data['duration']:.1f}s"
    )
    print(
        f"{'Avg per ticker':<30} {stats_yfinance['avg_per_ticker']:.3f}s{'':<15} {stats_ml4t_data['avg_per_ticker']:.3f}s"
    )
    print(
        f"{'Throughput':<30} {3600 / stats_yfinance['avg_per_ticker']:.0f} tickers/hour{'':<5} {3600 / stats_ml4t_data['avg_per_ticker']:.0f} tickers/hour"
    )

    print("\n" + "-" * 70)
    print(
        f"{'Performance ratio':<30} {'1.0x (baseline)':<20} {stats_yfinance['avg_per_ticker'] / stats_ml4t_data['avg_per_ticker']:.1f}x slower"
    )

    print("\n" + "-" * 70)
    print("Failure breakdown:")
    print(
        f"  Empty/unavailable: {stats_yfinance['empty']:<15} {stats_ml4t_data['data_unavailable']}"
    )
    print(f"  Rate limited:      {'N/A (manual sleep)':<15} {stats_ml4t_data['rate_limited']}")
    print(
        f"  Network errors:    {stats_yfinance['errors']:<15} {stats_ml4t_data['network_errors']}"
    )

    # Calculate reliability benefit
    reliability_improvement = (
        (stats_ml4t_data["successful"] - stats_yfinance["successful"])
        / stats_yfinance["total"]
        * 100
    )
    performance_cost = (
        (stats_ml4t_data["duration"] - stats_yfinance["duration"])
        / stats_yfinance["duration"]
        * 100
    )

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if reliability_improvement > 2:
        print(f"✅ ML4T Data provides {reliability_improvement:.1f}% better success rate")
        print(f"   Worth the {performance_cost:.0f}% performance cost")
    elif reliability_improvement > 0:
        print(f"⚠️  ML4T Data provides {reliability_improvement:.1f}% better success rate")
        print(f"   Marginal benefit for {performance_cost:.0f}% performance cost")
    else:
        print(f"❌ ML4T Data provides NO reliability benefit ({reliability_improvement:.1f}%)")
        print(f"   Just {performance_cost:.0f}% slower for no gain")

    print("\n" + "=" * 80)


def main():
    """Run stress test comparing both approaches."""
    print("=" * 80)
    print("STRESS TEST: ML4T Data vs Direct yfinance")
    print("=" * 80)
    print("\nLoading 200 tickers from Quandl dataset...")

    tickers = load_quandl_tickers(limit=200)
    print(f"Loaded {len(tickers)} tickers")
    print(f"Sample: {', '.join(tickers[:10])}")

    # Test period: 1 year of data
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    print(f"\nTest period: {start_date} to {end_date}")
    print("\n" + "=" * 80)

    # Test 1: Direct yfinance (ml3t approach)
    print("\n[1/2] Testing DIRECT YFINANCE (100ms delay, manual error handling)")
    print("=" * 80)
    stats_yfinance = test_direct_yfinance(tickers, start_date, end_date)

    # Test 2: ML4T Data approach
    print("\n[2/2] Testing QDATA PROVIDER (0.5 req/sec, automatic rate limiting)")
    print("=" * 80)
    stats_ml4t_data = test_ml4t_data_provider(tickers, start_date, end_date)

    # Compare results
    print_comparison(stats_yfinance, stats_ml4t_data)

    # Save detailed results
    results_file = Path(__file__).parent / "stress_test_results.txt"
    with open(results_file, "w") as f:
        f.write("STRESS TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Tickers: {len(tickers)}\n")
        f.write(f"Period: {start_date} to {end_date}\n\n")

        f.write("Direct yfinance:\n")
        for k, v in stats_yfinance.items():
            if k != "failures":
                f.write(f"  {k}: {v}\n")

        f.write("\nML4T Data:\n")
        for k, v in stats_ml4t_data.items():
            if k != "failures":
                f.write(f"  {k}: {v}\n")

        f.write("\nFailed tickers (yfinance):\n")
        for ticker, reason in stats_yfinance["failures"][:20]:
            f.write(f"  {ticker}: {reason}\n")

        f.write("\nFailed tickers (ML4T Data):\n")
        for ticker, reason in stats_ml4t_data["failures"][:20]:
            f.write(f"  {ticker}: {reason}\n")

    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
