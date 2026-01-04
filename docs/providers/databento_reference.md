# Databento Provider Reference

## Official Documentation
**Main Documentation**: https://databento.com/docs
**Historical API Reference**: https://databento.com/docs/api-reference-historical/client?historical=python&live=python&reference=python
**Datasets & Venues**: https://databento.com/docs/venues-and-datasets?historical=python&live=python&reference=python
**Schemas & Data Formats**: https://databento.com/docs/schemas-and-data-formats?historical=python&live=python&reference=python

## Key Concepts

### Datasets
Databento organizes data by dataset, which typically corresponds to an exchange or data source:
- **XNAS.ITCH**: NASDAQ ITCH feed (equities)
- **GLBX.MDP3**: CME Globex MDP 3.0 (futures)
- **OPRA.PILLAR**: Options Price Reporting Authority
- **DBEQ.BASIC**: Databento Equities Basic

### Schemas
Data schemas define the structure and granularity of data:
- **ohlcv-1m/1h/1d**: OHLC bars with volume at different frequencies
- **trades**: Individual trade records
- **tbbo**: Top of book bid/offer (quotes)
- **mbo**: Market by order (full order book)
- **mbp-1/10**: Market by price (aggregated order book levels)

### Symbol Types (stype_in)
- **raw_symbol**: Exchange-native symbols (e.g., "ESH4" for March 2024 E-mini)
- **parent**: Root symbol (e.g., "ES" for all E-mini S&P contracts)
- **continuous**: Continuous contracts (e.g., "ES.v.0" for front month)

### Continuous Futures Notation
Databento uses `.v.N` notation for continuous contracts:
- `ES.v.0`: Front month E-mini S&P 500
- `CL.v.1`: Second month Crude Oil
- `NQ.v.0`: Front month E-mini NASDAQ

### Session Dates for Non-Calendar Trading
Some exchanges have trading sessions that don't align with calendar dates:
- **CME Futures**: Trading day starts at 5:00 PM CT (previous calendar day)
- **Other Futures Markets**: May have different session start times
- **24/7 Markets (Crypto)**: Typically use calendar dates

The provider supports configurable session date adjustment:
```python
# For CME futures with session starting at 5pm CT (10pm UTC summer time)
provider = DatabentaProvider(
    dataset="GLBX.MDP3",
    adjust_session_dates=True,
    session_start_hour_utc=22
)

# For equities or crypto (calendar dates)
provider = DatabentaProvider(
    dataset="XNAS.ITCH",
    adjust_session_dates=False  # Default
)
```

## Implementation Notes

### Rate Limits
- Historical API: 100 requests/second (very generous)
- Real-time API: Varies by subscription
- Metadata API: 10 requests/second

### Data Formats
- **DBN**: Databento's native binary format (most efficient)
- **CSV**: Text format (larger, slower)
- **JSON**: For small datasets only
- **Parquet**: Available through client conversion

### Cost Considerations
- Billed per symbol-day of data
- Different schemas have different costs
- MBO/full book data is more expensive than OHLCV
- Check pricing at https://databento.com/pricing

## Usage Examples

### Basic OHLCV Fetch
```python
# For futures with session date adjustment
provider = DatabentaProvider(
    api_key="YOUR_KEY",
    dataset="GLBX.MDP3",
    default_schema="ohlcv-1m",
    adjust_session_dates=True,  # Enable for CME futures
    session_start_hour_utc=22   # 5pm CT = 10pm UTC (summer)
)

# For equities (no session adjustment needed)
provider = DatabentaProvider(
    api_key="YOUR_KEY",
    dataset="XNAS.ITCH",
    default_schema="ohlcv-1m"
)

# Fetch data
df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31", "minute")
```

### Multiple Schemas
```python
# Fetch both trades and quotes
schemas = ["trades", "tbbo"]
data = provider.fetch_multiple_schemas("AAPL", "2024-01-01", "2024-01-01", schemas)
trades_df = data["trades"]
quotes_df = data["tbbo"]
```

### Continuous Futures
```python
# Fetch front month crude oil continuous contract
df = provider.fetch_continuous_futures(
    root_symbol="CL",
    start="2024-01-01",
    end="2024-01-31",
    frequency="daily",
    version=0  # Front month
)
```

## Error Handling

Common errors and solutions:

1. **Authentication Error**: Check API key is valid and has permissions
2. **Dataset Not Found**: Verify dataset name and subscription
3. **Symbol Not Found**: Check symbol format and stype_in parameter
4. **Rate Limit**: Implement exponential backoff (handled by base provider)
5. **Invalid Date Range**: Ensure data exists for requested period

## Testing

When testing the Databento provider:
1. Use mock responses for unit tests (avoid API costs)
2. Create integration tests with `DATABENTO_API_KEY` env var
3. Test continuous contract handling
4. Test session date adjustment configuration (when enabled)
5. Test multiple schema fetching
6. Verify behavior with different datasets (equities, futures, options)

## Migration from ML3T Pattern

The ML4T Data Databento provider improves on the ML3T pattern by:
1. Using Polars throughout (no pandas conversion)
2. Supporting multiple schemas in one call
3. Configurable session date handling (not hardcoded for CME)
4. Built-in continuous contract support
5. Comprehensive error handling with circuit breaker
6. Support for all Databento datasets (not just futures)

## References

- [API Client Documentation](https://databento.com/docs/api-reference-historical/client)
- [Symbology Guide](https://databento.com/docs/symbology)
- [Data Quality Information](https://databento.com/docs/data-quality)
- [Changelog & Updates](https://databento.com/docs/changelog)
