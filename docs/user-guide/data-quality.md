# Data Quality

ml4t-data provides two complementary quality systems: **OHLCV validation** for
structural correctness checks, and **anomaly detection** for statistical
pattern analysis. Both can be run from code or through the CLI.

## OHLCV Validation

The `OHLCVValidator` performs eight configurable checks and two structural checks on
any DataFrame with standard OHLCV columns (`timestamp`, `open`, `high`, `low`, `close`,
`volume`). Structural numeric-type and finite-value checks cannot be disabled.

### Checks Performed

| Check | Severity | What it catches |
|-------|----------|-----------------|
| Numeric columns | CRITICAL | Non-numeric OHLCV columns |
| Finite values | ERROR | NaN or positive or negative infinity |
| Null values | ERROR | Missing prices or volume in any OHLCV column |
| Price consistency | ERROR | `high < low`, `high < open`, `low > close`, etc. |
| Negative prices | ERROR or WARNING | Any OHLC value below zero, depending on policy |
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
# max_return_threshold=0.5, staleness_threshold=3

# Forex is less volatile - tighter thresholds
forex_rules = ValidationRulePresets.forex_rules()
# max_return_threshold=0.1, staleness_threshold=10

# Strict mode for production data
strict_rules = ValidationRulePresets.strict_rules()
# max_return_threshold=0.1, staleness_threshold=3, all checks enabled
```

Available presets: `equity_rules()`, `crypto_rules()`, `forex_rules()`,
`commodity_rules()`, `futures_rules()`, `strict_rules()`, `relaxed_rules()`.
Commodity and futures presets report negative prices as warnings. The
`negative_price_policy` setting accepts `"forbid"`, `"warn"`, or `"allow"`.
Presets adjust numerical thresholds, not calendar selection. Crypto and forex presets no longer
disable schedule checks implicitly; declare the market's actual schedule. Explicit per-check
opt-outs remain available.

### Validate Calendar Membership and Continuity

`CrossValidator`, `AssetValidator` and `CryptoSessionValidator` accept the canonical `calendar`
keyword and a shared `feed_spec` on construction and validation. A call's explicit calendar
overrides its constructor setting; the effective explicit setting overrides feed metadata, then
NYSE applies. Version-2 metadata can also explicitly disable calendars. Neither a descriptive
asset class nor an exchange name selects a schedule.

```python
from ml4t.data.providers import SyntheticProvider
from ml4t.data.validation.cross_validation import CrossValidator
from ml4t.specs import FeedSpec

with SyntheticProvider(calendar_mode="continuous") as provider:
    spot = provider.fetch_ohlcv("BTCUSD", "2025-07-12", "2025-07-12", "minute")

feed = FeedSpec(calendar="24/7", data_frequency="minute", timestamp_semantics="event_time")
report = CrossValidator(feed_spec=feed).validate(spot)
row_only = CrossValidator(calendar=None).validate(spot)
```

Calendar membership uses the same assignment as [native sessions](sessions.md), including DST,
holidays, early closes, overnight dates, breaks and registered custom schedules. Nanosecond instants
retain their precision in membership comparisons. Daily session
labels are not wall-clock observations; explicit event or close semantics override the daily-label
heuristic. Invalid calendars and incompatible temporal declarations fail clearly. Reported sample
indices refer to the original input positions. `calendar=None` disables temporal checks without
disabling price, volume, schema or reference-price checks.

`CrossValidator` checks membership, not missing-bar completeness. `CryptoSessionValidator` checks
continuity through the shared gap detector: explicit `"24/7"` still detects real outages, whereas
`None` performs no schedule-derived gap check. Intraday continuity requires the complete version-2
bar declaration described in [incremental updates](incremental-updates.md). An explicitly supplied
legacy `CryptoCalendar` object retains its elapsed continuous-gap and maintenance behavior; omission
no longer constructs that object automatically. `CrossValidator(is_crypto=True)` remains a legacy
24/7 hint only when neither an explicit calendar nor a feed calendar is declared. It cannot override
a canonical selection.

`DataManager.load()` and `import_data()` pass their effective declaration and acquisition frequency
to validation before storage. For spot data, declare `"24/7"`; use the selected exchange calendar
for futures. Explicit calendar disable does not bypass OHLCV validation.

`ValidationRuleConfig(calendar=...)` preserves omitted, named and explicitly disabled selections
through model copying, JSON and `ValidationRuleSet` YAML. New rule configurations use version 2;
version-1 payloads with a null calendar still inherit the default. Pass a restored configuration
as `CrossValidator(feed_spec=rules)` to apply its declaration, or override it with `calendar=...`.
Use ordinary `model_dump_json()` or `model_dump(mode="json")` without excluding explicit nulls.

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
