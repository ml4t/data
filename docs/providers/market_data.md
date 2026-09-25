# Market Data Sources

Use the asset-class references below to compare data products that `ml4t-data` wraps with
external sources that have a reproducible API, bulk-download route, or institutional feed.
Provider support means that the named public class is included in the package. It does not imply
that every product or field sold by the source is implemented.

!!! note "Verified September 2026"
    Coverage, history, access, and licensing change. The linked official product pages support the
    entries as of September 2026. Confirm current terms before building a durable dataset.

## Asset Classes

| Reference | Main selection questions |
|-----------|--------------------------|
| [Equities](equities.md) | Survivorship, corporate actions, consolidated versus venue data, and timestamp semantics |
| [ETFs](etfs.md) | Delisted funds, holdings and classifications, fees, and adjusted prices |
| [Futures](futures.md) | Individual contracts, roll rules, continuous-series construction, and exchange licenses |
| [Options](options.md) | Contract coverage, quote history, implied-volatility methodology, and corporate actions |
| [Foreign exchange](fx.md) | Dealer or venue provenance, reference rates versus executable quotes, and volume interpretation |
| [Cryptocurrency](crypto.md) | Venue survivorship, instrument type, market fragmentation, and exchange reliability |

## Shared Due Diligence

- A historical universe should include securities, contracts, pairs, and venues that later
  disappeared. Current-symbol lists create survivorship bias.
- Preserve raw prices and corporate-action or roll inputs where possible. Adjusted histories and
  continuous futures are constructed series that can change when recomputed.
- Daily bars, minute bars, trades and quotes, and full order-book messages are different products.
  Confirm session definitions, timestamp origin, and venue coverage.
- Exchange and vendor licenses can restrict redistribution, display, production use, or derived
  data. A research entitlement does not establish production rights.

## Name Changes and Retired Products

| Previous name or product | Current status |
|--------------------------|----------------|
| Thomson Reuters / Refinitiv Tick History | [LSEG Tick History](https://www.lseg.com/en/data-analytics/market-data/data-feeds/tick-history) |
| Polygon.io | [Massive](https://massive.com/), renamed in October 2025; the library class is `MassiveProvider` |
| Quandl WIKI Prices | Frozen since April 2018; available through `WikiPricesProvider` as a static research dataset |
| IEX Cloud | Retired August 31, 2024; excluded from the source references |

## See Also

- [Provider comparison](index.md)
- [Fixed-income data sources](fixed_income.md)
- [Macroeconomic data sources](macro.md)
- [Fundamental data sources](fundamentals.md)
- [Alternative data sources](alternative_data.md)
