# Options Data Sources

Options research needs dated contract terms, underlying corporate actions, and a clear distinction
between observed quotes and vendor-computed analytics. Implied volatility and Greeks from two
vendors can differ even when they begin with the same market data.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the vendor contract and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [Databento OPRA](https://databento.com/datasets/OPRA.PILLAR) | US equity and index option trades and quotes from April 2013 | API and batch download | Usage-based account | Raw OPRA data does not include a vendor volatility surface or survivorship-cleaned underlying panel | `DataBentoProvider` |
| [Massive Options](https://massive.com/options) | US option chains, trades, quotes, aggregates, and plan-dependent analytics | REST, WebSocket, and flat files | API key; plan-dependent | Confirm whether Greeks and implied volatility are observed snapshots or recomputed fields | `MassiveProvider` |
| [OptionMetrics IvyDB US](https://optionmetrics.com/united-states/) | US equity and index options from 1996 | Institutional files, commonly through WRDS | Academic or institutional license | Daily standardized analytics are vendor estimates; intraday data is a separate product | No |
| [ORATS Data API](https://orats.com/data-api) | US equity, ETF, and index options; end-of-day history from 2007 and intraday snapshots | REST API and bulk files | Commercial account | Smoothed volatility and Greeks depend on ORATS models and cleaning rules | No |
| [Cboe DataShop](https://datashop.cboe.com/) | Historical option trades, quotes, summaries, and analytics across Cboe products and OPRA | Download and cloud delivery | Product-specific purchase | Coverage and analytics differ across DataShop products; select the exact schema before comparing results | No |
| [AlgoSeek US Options](https://www.algoseek.com/products.html#us_options) | OPRA-derived trades, quotes, bars, and analytics | Cloud delivery and bulk files | Commercial license; samples available | Quote cleaning and calculated analytics depend on the selected product | No |
| [CME DataMine options](https://www.cmegroup.com/datamine.html) | Options on CME Group futures across product families | API, exchange downloads, and cloud delivery | Paid by dataset | Futures-option symbology and exercise terms require the matching contract reference data | No |

`DataBentoProvider` exposes option-chain and option-quote helpers. `MassiveProvider` exposes
multi-asset market data, including options. Neither mapping implies support for every analytics
field sold by the vendor.

## Related References

- [Futures data sources](futures.md)
- [Equity data sources](equities.md)
- [Market data selection](market_data.md)
