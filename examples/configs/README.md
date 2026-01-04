# ML4T Data Configuration Examples

This directory contains example configuration files for the ml4t-data CLI. These examples demonstrate various use cases and configuration patterns.

## Configuration Files

### Basic Examples

- **`basic.yaml`** - Minimal configuration to get started with ml4t-data
  - Single provider (Yahoo Finance)
  - Simple daily stock data collection
  - Basic scheduling

### Advanced Examples

- **`advanced.yaml`** - Comprehensive configuration showing all features
  - Multiple providers with API keys
  - Environment variable interpolation
  - Complex workflows with hooks
  - Advanced scheduling options
  - Global validation and storage settings

- **`crypto_focus.yaml`** - Configuration optimized for cryptocurrency data
  - Multiple crypto exchanges (Binance, CryptoCompare)
  - High-frequency data collection (minute, hourly)
  - Crypto-specific validation rules
  - 24/7 scheduling

### Modular Configuration

The `modular/` directory demonstrates how to split configurations into reusable components:

- **`modular/base.yaml`** - Shared base configuration
- **`modular/providers.yaml`** - Reusable provider definitions
- **`modular/production.yaml`** - Production configuration using includes

### Environment-Specific

- **`development.yaml`** - Configuration for local development and testing
  - Mock data provider
  - Test datasets with fixed date ranges
  - Relaxed validation rules
  - Debug logging

## Usage

### Basic Usage

1. Copy an example configuration to your project root:
```bash
cp examples/configs/basic.yaml ml4t-data.yaml
```

2. Modify the configuration for your needs

3. Validate the configuration:
```bash
ml4t-data validate-config
```

4. Run a workflow:
```bash
ml4t-data run-workflow daily_update
```

### Using Environment Variables

Many configuration values can use environment variables with the `${VAR_NAME}` syntax:

```yaml
providers:
  - name: binance
    api_key: ${BINANCE_API_KEY}
    api_secret: ${BINANCE_API_SECRET}
```

You can also provide default values:
```yaml
base_dir: ${ML4T_DATA_DIR:./data}  # Uses ./data if ML4T_DATA_DIR is not set
```

### Loading Symbols from Files

Instead of listing symbols directly, you can load them from a file:

```yaml
datasets:
  - name: sp500_stocks
    symbols: "@symbols/sp500.txt"  # Loads symbols from file
```

### Using Modular Configurations

You can split your configuration into multiple files and use includes:

```yaml
include:
  - base.yaml
  - providers.yaml

# Override specific settings
log_level: WARNING
```

## Configuration Schema

### Top-Level Fields

- `version` - Configuration version (currently "1.0")
- `base_dir` - Base directory for data storage
- `log_level` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `providers` - List of data provider configurations
- `datasets` - List of dataset definitions
- `workflows` - List of workflow configurations
- `defaults` - Default settings for datasets
- `validation` - Global validation settings
- `storage` - Storage backend configuration
- `env` - Environment variable definitions

### Provider Configuration

```yaml
providers:
  - name: provider_name       # Unique name
    type: yahoo               # Provider type
    api_key: ${API_KEY}       # API key (if required)
    rate_limit: 1.0           # Requests per second
    timeout: 30               # Request timeout
    retry_count: 3            # Number of retries
    extra:                    # Provider-specific settings
      key: value
```

### Dataset Configuration

```yaml
datasets:
  - name: dataset_name        # Unique name
    symbols:                  # List of symbols or @file reference
      - AAPL
      - MSFT
    provider: provider_name   # Provider to use
    frequency: daily          # Data frequency
    asset_class: equity       # Asset class
    start_date: "2023-01-01"  # Optional start date
    end_date: "2023-12-31"    # Optional end date
    update_mode: incremental  # full or incremental
    validation:               # Dataset-specific validation
      strict: true
    storage:                  # Dataset-specific storage
      compression: zstd
```

### Workflow Configuration

```yaml
workflows:
  - name: workflow_name       # Unique name
    description: "..."        # Description
    datasets:                 # Datasets to process
      - dataset1
      - dataset2
    schedule:                 # Optional schedule
      type: daily             # Schedule type
      time: "09:30:00"        # Time (for daily/weekly)
      timezone: "UTC"         # Timezone
    enabled: true             # Enable/disable workflow
    on_error: stop            # Error handling: stop, continue, retry
    pre_hooks:                # Commands to run before
      - "command1"
    post_hooks:               # Commands to run after
      - "command2"
    notifications:            # Notification settings
      email:
        to: ["email@example.com"]
```

### Schedule Types

- `interval` - Run at fixed intervals (seconds)
- `daily` - Run daily at specified time
- `weekly` - Run weekly on specified day and time
- `cron` - Use cron expression
- `market_hours` - Run relative to market open/close

## Best Practices

1. **Use Environment Variables for Secrets**
   - Never commit API keys or secrets to version control
   - Use `${ENV_VAR}` syntax for sensitive values

2. **Start Simple**
   - Begin with a basic configuration
   - Add complexity as needed

3. **Validate Before Production**
   - Always run `ml4t-data validate-config` before deploying
   - Test workflows with `--dry-run` flag

4. **Use Modular Configurations for Large Setups**
   - Split configurations into logical components
   - Use includes to share common settings

5. **Set Appropriate Rate Limits**
   - Respect provider API limits
   - Use lower rates during development

6. **Configure Proper Error Handling**
   - Use `on_error: stop` for critical workflows
   - Use `on_error: continue` for best-effort updates

7. **Enable Validation**
   - Use appropriate validation rules for your asset class
   - Save validation reports for audit trails

## Troubleshooting

### Configuration Not Found
- ml4t-data looks for configuration files in these locations:
  - `./ml4t-data.yaml`, `./ml4t-data.yml`
  - `./.ml4t-data.yaml`, `./.ml4t-data.yml`
  - `./config/ml4t-data.yaml`
  - `~/.config/ml4t-data/config.yaml`

### Environment Variables Not Resolved
- Ensure variables are exported: `export BINANCE_API_KEY=your_key`
- Check for typos in variable names
- Use defaults: `${VAR_NAME:default_value}`

### Validation Errors
- Run `ml4t-data validate-config --show-warnings` for detailed output
- Check that all referenced providers and datasets exist
- Verify date formats are YYYY-MM-DD

### Workflow Not Running
- Check that workflow is enabled: `enabled: true`
- Verify schedule configuration is valid
- Check pre-hooks are not failing
- Review logs: `tail -f logs/ml4t-data.log`

## Support

For more information, see the main ml4t-data documentation or file an issue on GitHub.
