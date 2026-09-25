# Foreign Exchange Data Sources

Spot foreign exchange trades over the counter, so there is no consolidated tape. A dealer quote,
an interdealer venue price, and an official reference rate answer different research questions.
Reported volume is source-specific.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the vendor contract and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [OANDA v20](https://developer.oanda.com/rest-live-v20/introduction/) | OANDA account prices and candles for supported instruments | REST and streaming APIs | OANDA account and token | One broker's executable or indicative prices are not the whole OTC market | `OandaProvider` |
| [Twelve Data](https://twelvedata.com/docs) | Currency-pair time series and streaming data | REST and WebSocket APIs | API key; plan-dependent | Provider aggregation and delay differ by pair and account | `TwelveDataProvider` |
| [FXMacroData](https://fxmacrodata.com/) | Public USD rates and plan-dependent FX macro context | REST API | Public endpoints or API key | Macro and reference series are not executable dealer quotes | `FXMacroDataProvider` |
| [Massive Forex](https://massive.com/docs/rest/forex/overview) | Currency aggregates, quotes, and reference data | REST, WebSocket, and flat files | API key; plan-dependent | Confirm contributor, timestamp, and quote-side semantics | `MassiveProvider` |
| [Dukascopy historical data](https://www.dukascopy.com/swiss/english/marketwatch/historical/) | FX and CFD history from the Dukascopy trading environment | Download interface | Free | Single broker and venue context; CFD instruments are not spot transactions | No |
| [TrueFX historical downloads](https://www.truefx.com/truefx-historical-downloads/) | Top-of-book quotes for major and cross currency pairs | Registered file download | Free registration | Contributor set and retention differ from an institutional consolidated product | No |
| [ECB Data Portal exchange rates](https://data.ecb.europa.eu/key-figures/ecb-interest-rates-and-exchange-rates/exchange-rates) | Official euro reference rates and related statistical series | [SDMX REST API](https://data.ecb.europa.eu/help/getting-data-web-services-sdmx-0) and downloads | Free | Daily reference rates are not intraday or executable market prices | No |
| [LSEG Tick History](https://www.lseg.com/en/data-analytics/market-data/data-feeds/tick-history) | Contributed and venue FX quotes within a cross-asset archive | API and managed file delivery | Institutional license | Contributor and venue coverage depend on the licensed package | No |

## Related References

- [Futures data sources](futures.md)
- [Market data selection](market_data.md)
