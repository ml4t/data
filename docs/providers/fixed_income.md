# Fixed-Income Data Sources

Fixed-income research combines security reference data, evaluated or traded prices, yields,
cash flows, ratings, and transactions. A yield curve or evaluated price is a model output, while a
TRACE record is an executed trade. Keep those sources distinct.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the vendor contract and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [FRED and ALFRED](https://fred.stlouisfed.org/docs/api/fred/) | Treasury yields, credit spreads, policy rates, indexes, and other fixed-income series from contributing agencies | REST API | Free API key | Many series are aggregates or model outputs; use ALFRED vintages when release-time correctness matters | `FREDProvider` |
| [US Treasury Fiscal Data](https://fiscaldata.treasury.gov/api-documentation/) | Marketable debt, interest expense, auction, ownership, and related fiscal datasets | REST API and downloadable files | Free | Fiscal series and security records are not secondary-market quotes | No |
| [New York Fed Markets Data](https://www.newyorkfed.org/markets/reference-rates) | SOFR, EFFR, OBFR, repo rates, operations, and related money-market data | Markets Data APIs and downloads | Free | Reference rates are published aggregates with documented revision rules, not executable quotes | No |
| [FINRA Fixed Income and TRACE](https://www.finra.org/finra-data/fixed-income) | Corporate, agency, securitized, and 144A security and trade data | Website, daily files, feeds, and enhanced historical products | Public summaries or licensed subscription | TRACE disseminates executed trades, not quotes; caps, delays, and corrections affect interpretation | No |
| [CRSP US Treasury Database](https://www.crsp.org/research/) | Bills, notes, bonds, and fixed-term indexes for academic research | Institutional files | Academic or institutional license | CRSP conventions and month-end index calculations differ from live execution data | No |
| [ICE Fixed Income Data](https://www.ice.com/fixed-income-data-services) | Evaluated pricing, reference data, indexes, analytics, and market data across global debt markets | Feeds, APIs, and managed files | Institutional license | Evaluated prices are estimates and methodology, contributor coverage, and redistribution rights are product-specific | No |

For exchange-traded interest-rate and bond futures, use the [futures data reference](futures.md).
For policy rates and economic releases, use the [macroeconomic data reference](macro.md).

## Selection Notes

- Use security-level reference data with dated identifiers, coupons, call schedules, and cash
  flows. CUSIPs can change and a current description is not a historical master.
- Avoid treating a stale last trade as a current market price. Corporate bonds can trade
  infrequently, and evaluated prices add a pricing model.
- Preserve publication and revision timestamps for curves and macro-derived spreads.
