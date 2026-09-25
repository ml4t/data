# ETF Data Sources

ETF research often needs more than traded prices. Fund identity, liquidation history, holdings,
classifications, fees, distributions, and net asset value can each come from a different product.
Do not treat an active-fund screener as a historical universe.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the vendor contract and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [Yahoo Finance](https://finance.yahoo.com/markets/etfs/) | Listed ETF prices and current descriptive data | Web-backed API through the provider | No credential | Current listings and vendor-adjusted prices do not form a survivorship-free fund panel | `YahooFinanceProvider` |
| [Alpaca Market Data](https://docs.alpaca.markets/docs/about-market-data-api) | US-listed ETF trades, quotes, and bars from 2016 | REST and WebSocket APIs | Account; feed entitlement depends on plan | Provides market data, not a historical holdings or classification database | `AlpacaDataProvider` |
| [Tiingo End-of-Day](https://www.tiingo.com/documentation/end-of-day) | US equity, mutual-fund, and ETF prices | REST API | API key; plan-dependent limits | Not documented as a complete dead-fund universe | `TiingoProvider` |
| [Tiingo Fund and ETF Fees](https://www.tiingo.com/documentation/mutual-fund-and-etf-fees) | Current and historical fee records for funds and ETFs | Enterprise data delivery | Institutional agreement | Fee history is a separate product from prices and holdings | `TiingoProvider` for prices only |
| [EODHD](https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds) | ETF prices plus plan-dependent holdings and fund fields | REST API | API key; plan-dependent | Holdings dates, classifications, and constituent weights require point-in-time checks | `EODHDProvider` |
| [Massive Stocks](https://massive.com/stocks) | US-listed ETF trades, quotes, aggregates, and reference data | REST, WebSocket, and flat files | API key; plan-dependent | ETF-specific holdings and net asset value are not the same as exchange market data | `MassiveProvider` |
| [Sharadar Fund Prices](https://sharadar.com/docs/funds) | Active and delisted US-listed funds, ETFs, closed-end funds, ETNs, and ETDs from December 1997 | API and bulk delivery | Paid subscription | Daily price product; verify whether a separate dataset is needed for holdings or classifications | No |
| [Norgate Data](https://norgatedata.com/data-package-faq.php) | US ETF and ETN history from 1993, with plan-dependent delisted coverage | Local database and client integrations | Paid subscription | Daily frequency; constituent and classification features depend on package | No |
| [CRSP Survivor-Bias-Free US Mutual Fund Database](https://www.crsp.org/research/) | Active and inactive US open-end mutual funds with fund characteristics and returns | Institutional files, commonly through WRDS | Academic or institutional license | It is a mutual-fund research database, not a complete ETF holdings product | No |

The same ticker can represent different share classes or change its fund objective. Preserve stable
identifiers and effective dates for classifications, fees, and holdings rather than joining only on
the current ticker.

## Related References

- [Equity data sources](equities.md)
- [Fixed-income data sources](fixed_income.md)
- [Market data selection](market_data.md)
