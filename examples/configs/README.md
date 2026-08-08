# ML4T Data configuration examples

All files in this directory load through `ml4t.data.config.load_config()` and
`DataManager(config_path=...)`. Unknown fields are rejected. Paths are resolved relative to the
file that was passed to the loader.

Use `ml4t-starter.yaml` for a small Yahoo Finance dataset and `production.yaml` for a larger
production-oriented starting point. The other files demonstrate aliases, file includes, crypto,
FRED, and test configurations.

## Commands

```bash
ml4t-data update-all --config examples/configs/ml4t-starter.yaml --dry-run
ml4t-data update-all --config examples/configs/ml4t-starter.yaml
ml4t-data fetch --config examples/configs/basic.yaml --dataset tech_daily \
  --start 2024-01-01 --end 2024-01-31
```

## Canonical shape

Providers and datasets are lists of named objects:

```yaml
storage:
  strategy: hive
  base_path: ./market-data
  compression: zstd
  partition_granularity: month

providers:
  - name: yahoo
    type: yahoo
    rate_limit:
      requests_per_second: 2
      burst_size: 1

datasets:
  - name: daily_prices
    provider: yahoo
    symbols: [AAPL, MSFT]
    frequency: daily
    asset_class: equity
    lookback_days: 7
    initial_load_days: 3650
```

Use `symbols_file` for a file containing one symbol per line:

```yaml
datasets:
  - name: sp500
    provider: yahoo
    symbols_file: ../symbols/sp500.txt
```

The loader accepts the beta mapping form as an explicit compatibility migration, but saved and
maintained examples use the list form above.

## Credentials and provider settings

Reference credentials with `${VAR_NAME}`. An unresolved reference does not enable a provider and
is never sent as a literal credential. Configuration loading does not read `.env` files or mutate
the process environment.

Provider constructor settings belong under `extra`:

```yaml
providers:
  - name: oanda_practice
    type: oanda
    api_key: ${OANDA_API_KEY}
    extra:
      practice: true
```

Use a type shown by `ml4t-data providers`. The loader rejects credentials that the selected type
does not consume.

## Includes

```yaml
include:
  - modular/base.yaml
  - modular/providers.yaml
```

Included files are merged in order, then the main file is applied. Lists in a later file replace
earlier lists.
