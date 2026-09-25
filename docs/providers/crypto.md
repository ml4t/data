# Cryptocurrency Data Sources

Cryptocurrency data is fragmented across venues, instrument types, and chains. Preserve exchange,
pair, quote currency, contract type, and symbol history. Aggregated prices can conceal venue
closures, wash trading, and differences between spot, futures, perpetuals, and options.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the vendor contract and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [Binance Spot API](https://github.com/binance/binance-spot-api-docs) | Binance spot pairs, trades, order books, and bars | REST and WebSocket APIs | Public market endpoints; credentials for private or higher-limit use | Venue availability and product access vary by jurisdiction | `BinanceProvider` |
| [Binance Public Data](https://data.binance.vision/) | Bulk spot and futures market files | Public archive download | Free | Files are venue-specific and symbol coverage changes over time | `BinancePublicProvider` |
| [OKX Market Data](https://www.okx.com/docs-v5/en/#rest-api-market-data) | Spot and derivatives, including perpetual funding and premium data | REST and WebSocket APIs | Public market endpoints | Instrument and jurisdiction availability change; retain instrument metadata | `OKXProvider` |
| [CoinGecko API](https://docs.coingecko.com/) | Aggregated asset, exchange, market, and on-chain DEX data | REST API | Demo and paid API plans | Aggregated asset identity and market prices are not a substitute for venue-level execution data | `CoinGeckoProvider` |
| [CryptoCompare API](https://developers.cryptocompare.com/documentation) | Aggregated and exchange-specific crypto market data | REST and streaming APIs | API key for supported usage | Release qualification depends on live credential validation; aggregation methodology matters | `CryptoCompareProvider` |
| [Kaiko Market Data](https://www.kaiko.com/products/market-data) | Centralized and decentralized venues, spot and derivatives, trades and order books | API, streaming, and cloud delivery | Institutional license | Venue and instrument history depend on the contracted product | No |
| [Tardis.dev historical data](https://docs.tardis.dev/historical-data-details/overview) | Raw and normalized messages for centralized crypto exchanges, including closed venues | API and downloadable files | Commercial plan; limited samples | Reconstruction requires exchange-specific message semantics and snapshot handling | No |
| [CoinAPI Market Data](https://www.coinapi.io/products/market-data-api) | Multi-exchange spot and derivatives market data | REST, WebSocket, FIX, and files | API key; plan-dependent | Normalized symbols and aggregate feeds can hide exchange-specific contract details | No |

On-chain metrics, developer activity, and social signals belong in the
[alternative-data reference](alternative_data.md). This page covers tradable market data.

## Related References

- [Alternative data sources](alternative_data.md)
- [Futures data sources](futures.md)
- [Options data sources](options.md)
- [Market data selection](market_data.md)
