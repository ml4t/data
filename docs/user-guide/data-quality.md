# Data Quality

ml4t-data provides two complementary quality systems: **OHLCV validation** for
structural correctness checks, and **anomaly detection** for statistical
pattern analysis. Both can be run from code or through the CLI.

## OHLCV Validation

The `OHLCVValidator` performs eight configurable checks on any DataFrame with
standard OHLCV columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`).

### Checks Performed

| Check | Severity | What it catches |
|-------|----------|-----------------|
| Null values | ERROR | Missing prices or volume in any OHLCV column |
| Price consistency | ERROR | `high < low`, `high < open`, `low > close`, etc. |
| Negative prices | CRITICAL | Any OHLC value below zero |
| Negative volume | ERROR | Volume below zero |
| Duplicate timestamps | ERROR | Multiple rows with the same timestamp |
| Chronological order | ERROR | Timestamps not sorted ascending |
| Price staleness | WARNING | Close price unchanged for N+ consecutive days |
| Extreme returns | WARNING | Single-period return exceeding threshold |

### Usage

```python
from ml4t.data.validation import OHLCVValidator

# Default settings
validator = OHLCVValidator()
result = validator.validate(df)

if not result.passed:
    for issue in result.issues:
        print(f"[{issue.severity}] {issue.check}: {issue.message}")

# Customize thresholds
validator = OHLCVValidator(
    max_return_threshold=0.3,    # flag returns > 30%
    staleness_threshold=10,      # flag 10+ unchanged days
    check_extreme_returns=False, # disable return checks entirely
)
```

Each `ValidationIssue` includes the check name, severity, affected row count,
and up to 10 sample row indices for debugging.

### Validation Rules and Presets

For different asset classes, use the built-in presets which adjust thresholds
to match expected behavior:

```python
from ml4t.data.validation.rules import ValidationRulePresets

# Crypto is more volatile - wider thresholds
crypto_rules = ValidationRulePresets.crypto_rules()
# max_return_threshold=0.5, staleness_threshold=3, no market hours check

# Forex is less volatile - tighter thresholds
forex_rules = ValidationRulePresets.forex_rules()
# max_return_threshold=0.1, staleness_threshold=10, 24/5 market

# Strict mode for production data
strict_rules = ValidationRulePresets.strict_rules()
# max_return_threshold=0.1, staleness_threshold=3, all checks enabled
```

Available presets: `equity_rules()`, `crypto_rules()`, `forex_rules()`,
`commodity_rules()`, `strict_rules()`, `relaxed_rules()`.

### Persistent Rule Sets

Save and load validation rules per symbol or pattern:

```python
from ml4t.data.validation.rules import ValidationRuleSet, ValidationRulePresets

ruleset = ValidationRuleSet(name="production")
ruleset.default_rule = ValidationRulePresets.equity_rules()
ruleset.add_rule("BTC*", ValidationRulePresets.crypto_rules())
ruleset.add_rule("EUR*", ValidationRulePresets.forex_rules())

# Save to YAML
ruleset.save(Path("validation_rules.yaml"))

# Load later
ruleset = ValidationRuleSet.load(Path("validation_rules.yaml"))
rules = ruleset.get_rules("BTCUSD")  # matches "BTC*" pattern
```

## Anomaly Detection

The anomaly detection system uses statistical methods to find data quality
issues that pass basic validation but indicate potential problems.

### Built-in Detectors

**ReturnOutlierDetector** -- flags unusually large price moves.

- Methods: MAD (default), z-score, IQR
- MAD is robust to fat tails common in financial data
- Severity scales with magnitude: INFO (3x) to CRITICAL (5x+)

**VolumeSpikeDetector** -- flags unusual volume relative to a rolling baseline.

- Uses rolling z-score over a configurable window (default: 20 bars)
- Filters out low-volume noise with a minimum volume threshold

**PriceStalenessDetector** -- flags periods where prices do not change.

- Can check close-only or all OHLC prices
- Severity scales with duration: INFO (5 days) to CRITICAL (20+ days)
- Groups consecutive unchanged periods to avoid duplicate alerts

### Running Anomaly Detection

```python
from ml4t.data.anomaly import AnomalyManager, AnomalyConfig

# Default configuration (all detectors enabled)
manager = AnomalyManager()
report = manager.analyze(df, symbol="AAPL")

print(f"Found {len(report.anomalies)} anomalies")
for anomaly in report.anomalies:
    print(anomaly)
    # [WARNING] AAPL @ 2024-03-15: Unusual return of -8.42% (MAD z-score: 3.21)

# Check for critical issues
if report.has_critical_issues():
    print("Critical data quality issues found!")

# Convert to DataFrame for analysis
anomaly_df = report.to_dataframe()
```

### Custom Configuration

```python
from ml4t.data.anomaly import AnomalyConfig
from ml4t.data.anomaly.config import (
    ReturnOutlierConfig,
    VolumeSpikeConfig,
    PriceStalenessConfig,
)

config = AnomalyConfig(
    return_outliers=ReturnOutlierConfig(
        method="mad",       # "mad", "zscore", or "iqr"
        threshold=4.0,      # stricter than default 3.0
        min_samples=50,
    ),
    volume_spikes=VolumeSpikeConfig(
        window=30,           # 30-day rolling baseline
        threshold=4.0,
        min_volume=1000,     # ignore low-volume days
    ),
    price_staleness=PriceStalenessConfig(
        max_unchanged_days=3,
        check_close_only=True,
    ),
)

manager = AnomalyManager(config=config)
```

### Asset-Class and Symbol Overrides

The config supports per-asset-class and per-symbol threshold overrides:

```python
config = AnomalyConfig(
    asset_overrides={
        "crypto": {
            "return_outliers": {"threshold": 5.0},  # crypto is volatile
            "price_staleness": {"max_unchanged_days": 2},
        }
    },
    symbol_overrides={
        "BTCUSD": {
            "return_outliers": {"threshold": 6.0},  # BTC even more volatile
        }
    },
)

manager = AnomalyManager(config=config)
report = manager.analyze(df, symbol="BTCUSD", asset_class="crypto")
```

### Batch Analysis and Reports

```python
# Analyze multiple symbols
datasets = {"AAPL": df_aapl, "MSFT": df_msft, "GOOGL": df_googl}
reports = manager.analyze_batch(datasets)

# Save reports to disk
for symbol, report in reports.items():
    if report.anomalies:
        manager.save_report(report, Path("./anomaly_reports"))

# Filter by severity
filtered = manager.filter_by_severity(report, min_severity="warning")

# Get statistics
stats = manager.get_statistics(report)
# {"total_anomalies": 12, "by_severity": {"warning": 8, "error": 3, ...}, ...}
```

### CLI Integration

Run validation and anomaly detection together from the command line:

```bash
# Basic validation
ml4t-data validate -s AAPL --storage-path ./data

# With anomaly detection
ml4t-data validate -s AAPL --anomalies --storage-path ./data

# Filter noise -- only show errors and critical
ml4t-data validate --all --anomalies --severity error --storage-path ./data

# Save anomaly reports for later review
ml4t-data validate --all --anomalies --save-report --storage-path ./data
```

The CLI returns exit code 1 if any issues are found, making it suitable for
CI pipelines and pre-processing checks.
