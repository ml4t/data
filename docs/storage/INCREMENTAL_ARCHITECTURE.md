# Incremental Update Architecture

This document describes ml4t-data's incremental update system for efficient data maintenance.

---

## Overview

Instead of re-downloading entire datasets, ml4t-data tracks what you have and fetches only what's missing.

**Benefits**:
- **10x faster updates**: Fetch 5 new rows instead of 5,000
- **API cost savings**: Fewer calls = lower costs for paid providers
- **Network efficiency**: Minimal bandwidth usage
- **Data integrity**: Gap detection catches missing data

---

## Update Strategies

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `INCREMENTAL` | Append from last timestamp | Daily updates |
| `APPEND_ONLY` | Never re-fetch, only append new | Immutable data |
| `FULL_REFRESH` | Re-download everything | Monthly rebuilds, data corrections |
| `BACKFILL` | Fill gaps in historical data | Data quality issues |

### INCREMENTAL (Default)

Most common strategy for daily updates:

1. Read existing data, find latest timestamp
2. Fetch data from `last_timestamp + 1 day` to `today`
3. Append to storage
4. Update metadata

```yaml
datasets:
  sp500:
    provider: yahoo
    strategy: incremental  # default
```

### APPEND_ONLY

For data that never changes (historical archives):

1. Read existing data, find latest timestamp
2. Fetch only data after latest timestamp
3. Never modify existing partitions

```yaml
datasets:
  wiki_prices:
    provider: wiki_prices
    strategy: append_only  # Historical data doesn't change
```

### FULL_REFRESH

For monthly rebuilds or data corrections:

1. Delete existing data
2. Fetch entire date range
3. Write fresh data

```yaml
# Use sparingly - expensive for API calls
datasets:
  equities:
    provider: yahoo
    strategy: full_refresh
```

### BACKFILL

For filling gaps in existing data:

1. Scan existing data for gaps
2. Fetch missing date ranges
3. Insert into appropriate partitions

```yaml
datasets:
  sp500:
    provider: yahoo
    strategy: backfill
```

---

## Metadata Tracking

ml4t-data maintains metadata for each dataset:

```
~/ml4t-data/
├── yahoo/
│   └── daily/
│       └── AAPL/
│           ├── data.parquet
│           └── .metadata.json  # Tracking info
```

### Metadata Structure

```json
{
  "symbol": "AAPL",
  "provider": "yahoo",
  "frequency": "daily",
  "first_timestamp": "2020-01-02T00:00:00",
  "last_timestamp": "2024-12-20T00:00:00",
  "row_count": 1250,
  "last_updated": "2024-12-21T18:00:00",
  "update_count": 45,
  "gaps": []
}
```

---

## Gap Detection

Gaps can occur due to:
- Network failures during download
- Provider outages
- Timezone mismatches
- Rate limiting

### Gap Detection Algorithm

```python
def detect_gaps(timestamps: list, frequency: str) -> list[tuple]:
    """
    Detect gaps in time series.

    Returns list of (gap_start, gap_end, missing_bars) tuples.
    """
    expected_gap = {
        "daily": timedelta(days=1),
        "1h": timedelta(hours=1),
        "1m": timedelta(minutes=1),
    }[frequency]

    max_gap = {
        "daily": timedelta(days=4),   # Weekends
        "1h": timedelta(hours=2),
        "1m": timedelta(minutes=5),
    }[frequency]

    gaps = []
    for i in range(1, len(timestamps)):
        actual_gap = timestamps[i] - timestamps[i-1]
        if actual_gap > max_gap:
            gaps.append((timestamps[i-1], timestamps[i], actual_gap))

    return gaps
```

### Automatic Gap Filling

When gaps are detected:

1. Log gap details for review
2. Optionally auto-fill with BACKFILL strategy
3. Mark filled gaps in metadata

---

## CLI Commands

```bash
# Incremental update (default)
ml4t-data update-all -c config.yaml

# Specific dataset
ml4t-data update-all -c config.yaml --dataset sp500

# Dry run (preview)
ml4t-data update-all -c config.yaml --dry-run

# Force full refresh
ml4t-data update-all -c config.yaml --strategy full_refresh

# Backfill gaps
ml4t-data update-all -c config.yaml --strategy backfill

# List stored data
ml4t-data list -c config.yaml
```

---

## Concurrent Updates

### File Locking

Prevent concurrent writes to same partition:

```python
from ml4t.data.utils.locking import FileLock

with FileLock(partition_path):
    # Safe to write
    storage.write(df, ...)
```

### Parallel Symbol Updates

Update multiple symbols concurrently:

```python
from concurrent.futures import ThreadPoolExecutor

def update_symbol(symbol):
    provider = YahooFinanceProvider()
    df = provider.fetch_ohlcv(symbol, start, end)
    storage.write(df, provider="yahoo", frequency="daily", symbol=symbol)
    provider.close()
    return symbol

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(update_symbol, symbols))
```

### Rate Limiting

Respect provider rate limits:

```yaml
datasets:
  sp500:
    provider: yahoo
    rate_limit: 2.0  # seconds between calls
    max_concurrent: 3  # parallel requests
```

---

## Update Scheduling

### Cron Examples

```bash
# Daily update after market close (6 PM ET)
0 18 * * 1-5 ml4t-data update-all -c config.yaml >> logs/update.log 2>&1

# Weekly full refresh (Sunday 2 AM)
0 2 * * 0 ml4t-data update-all -c config.yaml --strategy full_refresh

# Monthly backfill (1st of month, 3 AM)
0 3 1 * * ml4t-data update-all -c config.yaml --strategy backfill
```

### Recommended Schedule

| Frequency | Update Time | Strategy |
|-----------|-------------|----------|
| Daily | 6 PM ET (after market) | INCREMENTAL |
| Weekly | Sunday 2 AM | FULL_REFRESH |
| Monthly | 1st of month | BACKFILL |

---

## Logging and Monitoring

### Update Logs

```json
{
  "timestamp": "2024-12-21T18:05:00Z",
  "config": "ml4t-data.yaml",
  "duration_seconds": 45.2,
  "symbols_updated": 500,
  "rows_added": 2500,
  "gaps_detected": 3,
  "gaps_filled": 3,
  "errors": []
}
```

### Alerting

```python
def check_update_health(log_path: str) -> bool:
    """Check if update completed successfully."""
    log = json.load(open(log_path))

    if log["errors"]:
        send_alert(f"Update errors: {log['errors']}")
        return False

    if log["gaps_detected"] > 10:
        send_alert(f"Many gaps detected: {log['gaps_detected']}")

    return True
```

---

## Best Practices

### 1. Run Updates After Market Close

US markets close at 4 PM ET. Wait until 5-6 PM for data to settle:
- Delayed quotes finalize
- After-hours corrections applied
- Provider caches refreshed

### 2. Use Incremental for Daily, Full Refresh Periodically

```bash
# Daily (5 days/week)
0 18 * * 1-5 ml4t-data update-all -c config.yaml

# Weekly full refresh (catches any corrections)
0 2 * * 0 ml4t-data update-all -c config.yaml --strategy full_refresh
```

### 3. Monitor Gap Rates

If gap rate exceeds 1%, investigate:
- Provider issues
- Network problems
- Rate limiting

### 4. Keep Metadata Backed Up

Metadata enables incremental updates. Back up regularly:
```bash
tar -czf metadata_backup.tar.gz ~/ml4t-data/**/.metadata.json
```

---

## Troubleshooting

### "Symbol not found" Errors

- Check symbol format (AAPL vs AAPL.US)
- Verify symbol is active (not delisted)
- Check provider-specific requirements

### Large Gap Counts

- Provider outage during previous update
- Rate limiting caused timeouts
- Run BACKFILL strategy to fill

### Slow Updates

- Reduce `max_concurrent` if rate limited
- Use file-based symbols for 500+ symbols
- Consider switching to faster provider

---

## See Also

- [Partitioning Guide](PARTITIONING_GUIDE.md) - Storage structure
- [Workflow Notebooks](../../notebooks/workflows/02_incremental_updates.py) - Examples
- [Provider Audit](../providers/PROVIDER_AUDIT.md) - Rate limits by provider
