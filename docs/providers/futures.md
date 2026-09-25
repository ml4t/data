# Futures Data Sources

Futures history is stored by expiring contract. A continuous series is a research construction,
so record the contract-selection rule, roll trigger, and price adjustment along with the output.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the vendor contract and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [Databento CME Globex MDP 3.0](https://databento.com/datasets/GLBX.MDP3) | CME Group futures and options from June 2010; order-level depth from March 2017 | API and batch download | Usage-based account | Continuous symbols embed a roll rule; retain individual contracts for reproducibility | `DataBentoProvider` |
| [Massive Futures](https://massive.com/futures) | Futures aggregates, trades, quotes, and reference data | REST, WebSocket, and flat files | API key; plan-dependent | Product history and exchange entitlements depend on the account | `MassiveProvider` |
| [CME DataMine](https://www.cmegroup.com/datamine.html) | Historical CME, CBOT, NYMEX, and COMEX futures and options; history varies by dataset | API, exchange downloads, and cloud delivery | Paid by dataset; academic program available | Data is contract-specific; continuous series and rolls are user-defined | No |
| [ICE Futures market data](https://www.ice.com/market-data) | ICE futures markets across energy, agriculture, financials, and other products | Feeds, APIs, and historical files | Exchange or vendor license | Coverage, history, and redistribution rights are product-specific | No |
| [Norgate Futures](https://norgatedata.com/futurespackage.php) | Global futures and cash commodities | Local database and client integrations | Paid subscription | Daily data; back-adjustment and roll settings materially affect returns | No |
| [FirstRate Data Futures](https://firstratedata.com/it/futures) | Individual and continuous histories for actively traded futures | Downloadable files | One-time or subscription purchase | Vendor-built continuous series may not match a strategy's intended roll rule | No |
| [CFTC Commitments of Traders](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm) | Weekly aggregate positions for reportable US futures and options markets | Public bulk files and reports | Free | Position reports are delayed aggregates, not price or order-book data | No |

Exchange codes, contract multipliers, tick values, trading hours, and symbology change over time.
Keep dated reference data with prices and open interest.

## Related References

- [Options data sources](options.md)
- [Market data selection](market_data.md)
