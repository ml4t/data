# Equity Data Sources

This reference compares reproducible equity price and microstructure products. For cross-sectional
research, prefer sources that retain delisted securities and expose corporate-action inputs. For
intraday research, distinguish consolidated US feeds from a single exchange or venue.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the vendor contract and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [Yahoo Finance](https://finance.yahoo.com/) | Global listed equities; vendor-defined history | Web-backed API through the provider | No credential | Not a survivorship-free research panel; adjustment and retention rules are vendor controlled | `YahooFinanceProvider` |
| [Alpaca Market Data](https://docs.alpaca.markets/docs/about-market-data-api) | US equities and ETFs from 2016 | REST and WebSocket APIs | Account; feed entitlement depends on plan | The basic feed is IEX rather than the consolidated SIP; delisted coverage is not documented | `AlpacaDataProvider` |
| [Tiingo End-of-Day](https://www.tiingo.com/documentation/end-of-day) | US equities, funds, and ETFs with raw and adjusted prices | REST API | API key; plan-dependent limits | Not documented as a survivorship-free universe | `TiingoProvider` |
| [EODHD](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes/) | Global end-of-day and intraday securities data | REST API | API key; coverage depends on plan | Exchange, adjustment, and delisted coverage vary by market and subscription | `EODHDProvider` |
| [Finnhub](https://finnhub.io/docs/api) | US quotes plus plan-dependent global and historical data | REST and WebSocket APIs | API key; endpoint entitlement depends on plan | Free access does not establish a complete historical universe | `FinnhubProvider` |
| [Twelve Data](https://twelvedata.com/docs) | Multi-exchange equities and other assets | REST and WebSocket APIs | API key; plan and exchange entitlements apply | Coverage and delays differ across exchanges | `TwelveDataProvider` |
| [Massive Stocks](https://massive.com/stocks) | US stock trades, quotes, aggregates, and reference data | REST, WebSocket, and flat files | API key; history and feed depend on plan | Confirm SIP versus exchange coverage and corporate-action treatment | `MassiveProvider` |
| [CRSP US Stock Databases](https://www.crsp.org/research/crsp-us-stock-databases/) | Active and inactive US securities, monthly from 1925 and daily from 1962 | Institutional files, commonly through WRDS | Academic or institutional license | Identifier and delisting-return conventions require deliberate handling | No |
| [Norgate Data](https://norgatedata.com/data-content-tables.php) | US, Australian, and Canadian stocks with plan-dependent delisted history | Local database and client integrations | Paid subscription | Daily data only; survivorship and index-membership features depend on tier | No |
| [Sharadar Equity Prices](https://data.nasdaq.com/databases/SEP) | Active and delisted US equities from 1998 | API and bulk delivery | Paid subscription | Daily bars; adjustment fields must be selected consistently | No |
| [NYSE TAQ](https://www.nyse.com/market-data/historical/daily-taq) | Trades and quotes reported by US exchanges | Daily institutional files | Exchange data license | Raw records need symbol-history, correction, and session processing | No |
| [AlgoSeek US Equities](https://www.algoseek.com/products.html#us_equity) | US trades, quotes, bars, reference data, and historical constituents | Cloud delivery and bulk files | Commercial license; samples available | Product schemas and SIP-derived fields differ; confirm the exact package | No |
| [Databento Nasdaq TotalView-ITCH](https://databento.com/datasets/XNAS.ITCH) | Nasdaq order-level data from 2018 | API and batch download | Usage-based account | Raw exchange messages are not adjusted and do not represent the full US consolidated market | `DataBentoProvider` |
| [Nasdaq TotalView-ITCH samples](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/) | Selected full trading days for Nasdaq-listed securities | Public binary files | Free samples | Samples are sparse and unsuitable for a broad historical panel | `ITCHSampleProvider` |
| [LSEG Tick History](https://www.lseg.com/en/data-analytics/market-data/data-feeds/tick-history) | Cross-asset trades, quotes, and depth from global venues | API and managed file delivery | Institutional license | Venue, contributor, and field coverage depend on the licensed package | No |

`DataBentoProvider` and `MassiveProvider` are multi-asset classes. The mappings above describe only
their equity capabilities. See the [provider comparison](index.md) for package-level credentials
and interfaces.

## Related References

- [ETF data sources](etfs.md)
- [Fundamental data sources](fundamentals.md)
- [Market data selection](market_data.md)
