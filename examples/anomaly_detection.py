"""ML4T Data Anomaly Detection Example

This script demonstrates how to use ML4T Data's anomaly detection system
to identify data quality issues in market data.

Features Demonstrated:
    - Price staleness detection (gaps in data)
    - Return outlier detection (abnormal price movements)
    - Volume spike detection (unusual trading volume)
    - Batch anomaly analysis across multiple symbols
    - Report generation and filtering

Requirements:
    - ml4t-data installed: pip install ml4t-data
    - No API key needed (uses CoinGecko free tier)
    - Internet connection

Usage:
    python examples/anomaly_detection.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml4t.data.anomaly import (
    AnomalyManager,
    AnomalySeverity,
    PriceStalenessDetector,
    ReturnOutlierDetector,
    VolumeSpikeDetector,
)
from ml4t.data.providers import CoinGeckoProvider


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


def print_anomaly(anomaly) -> None:
    """Pretty-print a single anomaly."""
    # Emoji mapping for severity
    severity_emoji = {
        AnomalySeverity.INFO: "ℹ️",
        AnomalySeverity.WARNING: "⚠️",
        AnomalySeverity.ERROR: "❌",
        AnomalySeverity.CRITICAL: "🚨",
    }

    emoji = severity_emoji.get(anomaly.severity, "❓")
    print(f"{emoji} [{anomaly.severity.value.upper()}] {anomaly.type.value}")
    print(f"   Date: {anomaly.timestamp}")
    print(f"   Message: {anomaly.message}")
    if anomaly.details:
        print(f"   Details: {anomaly.details}")
    print()


def example_single_symbol():
    """Example 1: Detect anomalies in a single cryptocurrency."""
    print_section_header("Example 1: Single Symbol Anomaly Detection")

    # Fetch Bitcoin data (last 90 days)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"📈 Fetching Bitcoin data from {start_date} to {end_date}...")
    provider = CoinGeckoProvider()
    btc_data = provider.fetch_ohlcv("bitcoin", start_date, end_date)
    print(f"✅ Fetched {len(btc_data)} days of data")
    print()

    # Create anomaly manager with all detectors
    print("🔍 Configuring anomaly detectors...")
    manager = AnomalyManager()

    # Add detectors with custom thresholds
    manager.detectors.append(PriceStalenessDetector(max_gap_days=3))
    manager.detectors.append(ReturnOutlierDetector(threshold=5.0))
    manager.detectors.append(VolumeSpikeDetector(threshold=10.0))

    print("   ✓ Price Staleness Detector (max gap: 3 days)")
    print("   ✓ Return Outlier Detector (threshold: 5.0 sigma)")
    print("   ✓ Volume Spike Detector (threshold: 10.0x median)")
    print()

    # Run anomaly detection
    print("🔍 Analyzing data for anomalies...")
    report = manager.analyze(btc_data, symbol="bitcoin", asset_class="crypto")

    # Display results
    print_section_header("Detection Results")

    print(f"Symbol: {report.symbol}")
    print(
        f"Period: {report.start_date.strftime('%Y-%m-%d')} to {report.end_date.strftime('%Y-%m-%d')}"
    )
    print(f"Total Rows: {report.total_rows}")
    print(f"Detectors Used: {', '.join(report.detectors_used)}")
    print()

    if report.anomalies:
        print(f"⚠️  Found {len(report.anomalies)} anomalies:")
        print()

        # Show critical anomalies first
        critical = report.get_critical_anomalies()
        if critical:
            print("🚨 CRITICAL ANOMALIES:")
            print("-" * 80)
            for anomaly in critical:
                print_anomaly(anomaly)

        # Show all other anomalies
        other = [a for a in report.anomalies if a.severity != AnomalySeverity.CRITICAL]
        if other:
            print("OTHER ANOMALIES:")
            print("-" * 80)
            for anomaly in other[:5]:  # Limit to first 5
                print_anomaly(anomaly)

            if len(other) > 5:
                print(f"   ... and {len(other) - 5} more anomalies")
                print()

        # Show summary statistics
        print_section_header("Summary Statistics")
        stats = manager.get_statistics(report)
        print(f"Total Anomalies: {stats['total_anomalies']}")
        print(f"Detection Rate: {stats['detection_rate']:.2%}")
        print()
        print("By Severity:")
        for severity, count in stats["by_severity"].items():
            if count > 0:
                print(f"  {severity.upper()}: {count}")
        print()
        print("By Type:")
        for anom_type, count in stats["by_type"].items():
            print(f"  {anom_type}: {count}")
    else:
        print("✅ No anomalies detected - data quality looks good!")

    provider.close()


def example_batch_analysis():
    """Example 2: Batch anomaly detection across multiple symbols."""
    print_section_header("Example 2: Batch Anomaly Detection")

    # Define symbols to analyze
    symbols = ["bitcoin", "ethereum", "cardano"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    print(f"📊 Analyzing {len(symbols)} cryptocurrencies...")
    print(f"Period: {start_date} to {end_date}")
    print()

    # Fetch data for all symbols
    provider = CoinGeckoProvider()
    datasets = {}

    for symbol in symbols:
        print(f"  📈 Fetching {symbol}...")
        try:
            data = provider.fetch_ohlcv(symbol, start_date, end_date)
            datasets[symbol] = data
            print(f"     ✓ {len(data)} days")
        except Exception as e:
            print(f"     ✗ Error: {e}")

    print()

    # Create anomaly manager
    manager = AnomalyManager()
    manager.detectors.append(PriceStalenessDetector(max_gap_days=2))
    manager.detectors.append(ReturnOutlierDetector(threshold=4.0))
    manager.detectors.append(VolumeSpikeDetector(threshold=8.0))

    # Run batch analysis
    print("🔍 Running anomaly detection on all symbols...")
    reports = manager.analyze_batch(datasets, asset_classes=dict.fromkeys(symbols, "crypto"))

    # Display results
    print_section_header("Batch Analysis Results")

    for symbol, report in reports.items():
        anomaly_count = len(report.anomalies)
        critical_count = len(report.get_critical_anomalies())

        if anomaly_count > 0:
            emoji = "🚨" if critical_count > 0 else "⚠️"
            print(
                f"{emoji} {symbol.upper()}: {anomaly_count} anomalies ({critical_count} critical)"
            )
        else:
            print(f"✅ {symbol.upper()}: No anomalies detected")

    print()

    # Show which symbol has the most data quality issues
    worst_symbol = max(reports.items(), key=lambda x: len(x[1].anomalies))
    if len(worst_symbol[1].anomalies) > 0:
        print(
            f"⚠️  Most Issues: {worst_symbol[0].upper()} with {len(worst_symbol[1].anomalies)} anomalies"
        )
        print()
        print("Top 3 anomalies:")
        for anomaly in worst_symbol[1].anomalies[:3]:
            print_anomaly(anomaly)

    provider.close()


def example_custom_detector():
    """Example 3: Using custom detector configuration."""
    print_section_header("Example 3: Custom Detector Configuration")

    # Fetch data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print("📈 Fetching Ethereum data...")
    provider = CoinGeckoProvider()
    eth_data = provider.fetch_ohlcv("ethereum", start_date, end_date)
    print(f"✅ Fetched {len(eth_data)} days of data")
    print()

    # Create manager with very strict thresholds
    print("🔍 Configuring STRICT anomaly detection (more sensitive)...")
    manager = AnomalyManager()

    # Strict detectors - will find more anomalies
    manager.detectors.append(PriceStalenessDetector(max_gap_days=1))  # Flag any gaps
    manager.detectors.append(ReturnOutlierDetector(threshold=3.0))  # Lower threshold
    manager.detectors.append(VolumeSpikeDetector(threshold=5.0))  # Lower threshold

    print("   ✓ Max gap: 1 day (very strict)")
    print("   ✓ Return threshold: 3.0 sigma (more sensitive)")
    print("   ✓ Volume threshold: 5.0x median (more sensitive)")
    print()

    # Analyze
    report = manager.analyze(eth_data, symbol="ethereum", asset_class="crypto")

    print_section_header("Strict Detection Results")
    print(f"Found {len(report.anomalies)} anomalies with strict settings")

    if report.anomalies:
        # Filter to only show warnings and above
        filtered = manager.filter_by_severity(report, "warning")
        print(f"After filtering (warning+): {len(filtered.anomalies)} anomalies")
        print()

        for anomaly in filtered.anomalies[:5]:
            print_anomaly(anomaly)

        if len(filtered.anomalies) > 5:
            print(f"   ... and {len(filtered.anomalies) - 5} more")

    provider.close()


def example_save_report():
    """Example 4: Save anomaly reports to disk."""
    print_section_header("Example 4: Saving Anomaly Reports")

    # Fetch data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    provider = CoinGeckoProvider()
    btc_data = provider.fetch_ohlcv("bitcoin", start_date, end_date)

    # Run detection
    manager = AnomalyManager()
    manager.detectors.append(PriceStalenessDetector(max_gap_days=2))
    manager.detectors.append(ReturnOutlierDetector(threshold=4.0))
    manager.detectors.append(VolumeSpikeDetector(threshold=8.0))

    report = manager.analyze(btc_data, symbol="bitcoin", asset_class="crypto")

    # Save report
    output_dir = Path("./anomaly_reports")
    report_path = manager.save_report(report, output_dir)

    print(f"✅ Report saved to: {report_path}")
    print()
    print("Report contains:")
    print(f"  - Symbol: {report.symbol}")
    print(f"  - Period: {report.start_date} to {report.end_date}")
    print(f"  - Total anomalies: {len(report.anomalies)}")
    print(f"  - Detectors used: {', '.join(report.detectors_used)}")

    provider.close()


def main():
    """Run all anomaly detection examples."""
    print("=" * 80)
    print("  ML4T Data Anomaly Detection Examples")
    print("  Detect data quality issues in market data")
    print("=" * 80)

    try:
        # Run examples
        example_single_symbol()
        example_batch_analysis()
        example_custom_detector()
        example_save_report()

        # Summary
        print_section_header("What's Next?")
        print("You've learned how to:")
        print("  ✅ Detect anomalies in single symbols")
        print("  ✅ Run batch analysis across multiple symbols")
        print("  ✅ Configure custom detector thresholds")
        print("  ✅ Save reports to disk")
        print()
        print("Next steps:")
        print("  1. Integrate anomaly detection into your data pipeline")
        print("  2. Set up alerts for critical anomalies")
        print("  3. Create custom detectors for your specific needs")
        print("  4. Use reports to improve data quality over time")
        print()
        print("See: README.md → 'Advanced Features' section for more details")
        print()
        print("=" * 80)
        print("  Done! Happy trading! 🚀")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n\n❌ Error: {err}")
        print()
        print("Common issues:")
        print("  - Check internet connection")
        print("  - Verify ml4t-data is installed: pip install ml4t-data")
        print("  - Check CoinGecko API status: https://status.coingecko.com/")
        print()
        import traceback

        traceback.print_exc()
        sys.exit(1)
