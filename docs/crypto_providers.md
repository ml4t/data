# Cryptocurrency Data Providers

ML4T Data supports multiple cryptocurrency data providers for fetching historical and real-time market data.

## Available Providers

### CryptoCompare
- **Type**: Free tier available (100,000 calls/month)
- **Markets**: Spot prices from multiple exchanges
- **Data**: OHLCV data with aggregated pricing
- **Frequencies**: minute, hourly, daily
- **API Key**: Optional (higher rate limits with key)

### Binance
- **Type**: Free (no API key required for public data)
- **Markets**: Spot and Futures
- **Data**: OHLCV data directly from Binance
- **Frequencies**: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
- **Rate Limits**: 1200 weight per minute

## Usage Examples

### Loading Bitcoin Daily Data

```bash
# Using CryptoCompare
ml4t-data load --provider cryptocompare --symbol BTC --start 2024-01-01 --end 2024-01-31 --frequency daily --asset-class crypto

# Using Binance Spot
ml4t-data load --provider binance --symbol BTC --start 2024-01-01 --end 2024-01-31 --frequency daily --asset-class crypto

# Using Binance Futures
ml4t-data load --provider binance_futures --symbol BTC --start 2024-01-01 --end 2024-01-31 --frequency daily --asset-class crypto
```

### Loading Minute-Level Data

```bash
# Fetch minute data for Ethereum
ml4t-data load --provider binance --symbol ETH --start 2024-01-01 --end 2024-01-01 --frequency minute --asset-class crypto
```

### Incremental Updates

```bash
# Update existing crypto data
ml4t-data update --symbol BTC --frequency daily --asset-class crypto --lookback-days 7
```

## Symbol Formats

The providers automatically normalize various symbol formats:

- `BTC` → `BTCUSDT` (Binance) or `BTC/USD` (CryptoCompare)
- `BTC-USD` → normalized appropriately
- `BTC/USD` → normalized appropriately
- `BTCUSDT` → used directly for Binance

## Python API

```python
from ml4t.data.providers.cryptocompare import CryptoCompareProvider
from ml4t.data.providers.binance import BinanceProvider
from ml4t.data.pipeline import Pipeline
from ml4t.data.storage.filesystem import FileSystemBackend
from ml4t.data.core.config import Config

# Initialize provider
provider = BinanceProvider(market="spot")

# Initialize storage and config
storage = FileSystemBackend(data_root="./data")
config = Config()

# Create pipeline
pipeline = Pipeline(provider=provider, storage=storage, config=config)

# Load data
key = pipeline.run_load(
    symbol="BTC",
    start="2024-01-01",
    end="2024-01-31",
    frequency="daily",
    asset_class="crypto"
)

# Read the data
data_obj = storage.read(key)
df = data_obj.data  # Polars DataFrame
```

## 24/7 Market Considerations

Cryptocurrency markets operate 24/7, unlike traditional equity markets. The ML4T Data system handles this by:

1. **No Weekend Gaps**: Gap detection doesn't flag weekends as missing data for crypto
2. **UTC Timezone**: All crypto data uses UTC timestamps for consistency
3. **Extended Hours**: For intraday data, the end date is extended to 23:59:59 to capture the full day

## Rate Limiting

### CryptoCompare
- Free tier: 100,000 calls/month
- Automatic rate limiting with delays between requests
- API key can be provided for higher limits

### Binance
- Public endpoints: 1200 weight per minute
- Automatic retry on rate limit (429) errors
- Built-in delays between requests (0.05s)

## Data Storage

Crypto data is stored in the same format as other asset classes:

```
data/
├── crypto/
│   ├── daily/
│   │   ├── BTC.parquet
│   │   ├── ETH.parquet
│   │   └── ...
│   ├── minute/
│   │   ├── BTC.parquet
│   │   └── ...
│   └── hourly/
│       └── ...
```

## Metadata

Each dataset includes metadata about:
- Provider used (e.g., "binance_spot", "cryptocompare")
- Symbol normalization
- Date range
- Update history
- Data quality metrics

## Error Handling

The providers handle common errors:
- Invalid symbols return empty DataFrames
- Rate limit errors trigger automatic retries with backoff
- Network errors are retried with exponential backoff
- API errors are logged with clear messages

## Best Practices

1. **Use appropriate frequencies**: Minute data generates large files; use daily/hourly when possible
2. **Respect rate limits**: Don't make parallel requests to the same provider
3. **Store API keys securely**: Use environment variables or secure configuration
4. **Monitor data quality**: Check for gaps and stale data regularly
5. **Use incremental updates**: After initial load, use `ml4t-data update` for efficiency

## Future Enhancements

- [ ] WebSocket support for real-time data
- [ ] Additional exchanges (Coinbase, Kraken, etc.)
- [ ] Order book data
- [ ] Tick-level data support
- [ ] Cross-exchange arbitrage metrics
