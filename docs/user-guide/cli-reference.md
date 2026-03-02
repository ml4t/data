# CLI Reference

The `ml4t-data` command-line tool provides unified access to data fetching, storage
management, validation, and system administration. All commands support `--verbose`
and `--quiet` flags for output control.

```bash
ml4t-data [--verbose | --quiet] <command> [options]
```

## Core Commands

### fetch

Fetch financial data from any configured provider.

```bash
# Single symbol
ml4t-data fetch -s AAPL --start 2024-01-01 --end 2024-12-31

# Multiple symbols
ml4t-data fetch -s BTC -s ETH --start 2024-01-01 --end 2024-06-30

# From a symbols file (one symbol per line)
ml4t-data fetch -f symbols.txt --start 2024-01-01 --end 2024-12-31

# Specific provider and frequency
ml4t-data fetch -s AAPL --provider yahoo --frequency daily \
    --start 2024-01-01 --end 2024-12-31

# Save output to file
ml4t-data fetch -s AAPL --start 2024-01-01 --end 2024-12-31 -o data.parquet

# Show progress bar for batch fetches
ml4t-data fetch -f sp500.txt --start 2024-01-01 --end 2024-12-31 --progress
```

| Option | Short | Description |
|--------|-------|-------------|
| `--symbol` | `-s` | Symbol(s) to fetch (repeatable) |
| `--symbols-file` | `-f` | File containing symbols, one per line |
| `--start` | | Start date (YYYY-MM-DD), required |
| `--end` | | End date (YYYY-MM-DD), required |
| `--frequency` | | `daily`, `hourly`, or `weekly` (default: `daily`) |
| `--provider` | `-p` | Specific provider to use |
| `--output` | `-o` | Output file path (.parquet or .csv) |
| `--config` | `-c` | JSON configuration file |
| `--progress` | | Show progress bar for batch fetches |

### update

Perform incremental data updates for symbols already in storage.

```bash
# Incremental update (default - only fetches new data)
ml4t-data update -s AAPL --storage-path ./data

# Full refresh
ml4t-data update -s AAPL --strategy full_refresh --storage-path ./data

# Backfill with specific date range
ml4t-data update -s AAPL --strategy backfill \
    --start 2020-01-01 --end 2020-12-31 --storage-path ./data
```

| Option | Description |
|--------|-------------|
| `--symbol`, `-s` | Symbol to update (required) |
| `--start` | Start date (defaults to 30 days ago) |
| `--end` | End date (defaults to today) |
| `--strategy` | `incremental`, `append_only`, `full_refresh`, or `backfill` |
| `--provider`, `-p` | Provider to use for fetching |
| `--storage-path` | Storage directory (default: `./data`) |

### validate

Validate data quality and integrity. Checks OHLC relationships, duplicates,
and optionally runs anomaly detection.

```bash
# Validate a single symbol
ml4t-data validate -s AAPL --storage-path ./data

# Validate all stored symbols
ml4t-data validate --all --storage-path ./data

# Include anomaly detection
ml4t-data validate -s AAPL --anomalies --storage-path ./data

# Filter by severity and save report
ml4t-data validate -s AAPL --anomalies --severity error --save-report
```

| Option | Description |
|--------|-------------|
| `--symbol`, `-s` | Symbol to validate |
| `--all` | Validate all symbols in storage |
| `--anomalies` | Run anomaly detection (return outliers, volume spikes, staleness) |
| `--save-report` | Save anomaly report to `./anomaly_reports/` |
| `--severity` | Minimum severity: `info`, `warning`, `error`, `critical` |
| `--storage-path` | Storage directory (default: `./data`) |

### status

Show system overview and health status of stored datasets.

```bash
ml4t-data status --storage-path ./data
ml4t-data status --detailed --storage-path ./data
```

The `--detailed` flag shows per-symbol panels with provider, frequency, row count,
date range, last update, and health status.

### export

Export stored data to CSV, JSON, or Parquet.

```bash
ml4t-data export -s AAPL -o aapl.csv --format csv
ml4t-data export -s AAPL -o aapl.json --format json
ml4t-data export -s AAPL -o aapl.parquet --format parquet
```

### list

List all stored datasets, organized by provider type.

```bash
ml4t-data list --config ml4t-data.yaml
ml4t-data list --storage-path ./data
```

Requires either `--config` (YAML file with `storage.path`) or `--storage-path`.

### info

Show detailed information about a specific stored symbol, including a data preview.

```bash
ml4t-data info -s AAPL --storage-path ./data
```

## Batch Operations

### update-all

Update all datasets defined in a YAML configuration file.

```bash
# Update everything
ml4t-data update-all -c ml4t-data.yaml

# Update a specific dataset group
ml4t-data update-all -c ml4t-data.yaml --dataset futures

# Preview what would be updated
ml4t-data update-all -c ml4t-data.yaml --dry-run
```

The configuration file defines datasets with inline symbols or file references:

```yaml
storage:
  path: ~/ml4t-data

datasets:
  equities:
    provider: yahoo
    symbols: [AAPL, MSFT, GOOGL]
  sp500:
    provider: yahoo
    symbols_file: sp500.txt
```

## Futures Commands

### download-futures

Download historical futures data from Databento (requires API key).

```bash
ml4t-data download-futures -c futures-config.yaml
ml4t-data download-futures -c futures-config.yaml --dry-run   # cost estimate only
ml4t-data download-futures -c futures-config.yaml -p ES -p NQ  # specific products
ml4t-data download-futures -c futures-config.yaml -j 4          # parallel downloads
```

### update-futures

Update existing futures data with the latest available bars.

### download-cot

Download CFTC Commitment of Traders data.

```bash
ml4t-data download-cot -p ES -p CL --start-year 2020
ml4t-data download-cot --list-products          # show available product codes
ml4t-data download-cot -c cot-config.yaml -o ~/ml4t-data/cot
```

## System Commands

| Command | Description |
|---------|-------------|
| `ml4t-data version` | Show version and Python information |
| `ml4t-data providers` | List available data providers with API key requirements |
| `ml4t-data config` | Show current configuration |
| `ml4t-data health` | Check health status of all datasets |
| `ml4t-data health --detailed --stale-days 7` | Show per-symbol staleness |
| `ml4t-data server --port 8000` | Start the REST API server (requires `ml4t-data[api]`) |
| `ml4t-data show-completion bash` | Print shell completion script |

## Shell Completion

Enable tab completion for your shell:

```bash
# Bash
eval "$(_ML4T_DATA_COMPLETE=bash_source ml4t-data)"

# Zsh
eval "$(_ML4T_DATA_COMPLETE=zsh_source ml4t-data)"

# Fish
eval (env _ML4T_DATA_COMPLETE=fish_source ml4t-data)
```
