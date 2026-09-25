# Fundamental Data Sources

Fundamental research needs the value that was public at the decision date. Companies amend
filings, vendors standardize line items, and current databases often overwrite earlier values.
Keep the filing or release timestamp, accession or version, period end, currency, and issuer
identifier with every observation.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the vendor contract and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | US registrant submissions and XBRL facts; structured statement coverage phases in from 2009 to 2011 | REST APIs and nightly bulk ZIP files | Free; identifying `User-Agent` and fair-access limits apply | `companyfacts` retains filing metadata, but the cross-sectional `frames` endpoint returns the last-filed value for a period | No |
| [SEC Financial Statement Data Sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets) | Quarterly as-filed statement facts from 2009 | Bulk ZIP files | Free | Raw XBRL tags and duplicate facts require issuer, accession, unit, period, and form-level normalization | No |
| [Yahoo Finance](https://finance.yahoo.com/) | Current statements, ratios, and analyst fields for listed securities | Web-backed API through the provider | No credential | Latest values lack reliable first-publication and revision history | `YahooFinanceProvider` |
| [EODHD Fundamentals](https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds) | Global company statements, estimates, and related fields with market-dependent history | REST API | API key; paid fundamentals entitlement | Filing dates are present, but the public documentation does not establish complete restatement versioning | `EODHDProvider` |
| [Finnhub reported financials](https://finnhub.io/docs/api/financials-reported) | Global standardized data and SEC as-reported statements | REST API | API key; endpoint entitlement depends on plan | The as-reported endpoint carries filing dates; standardized statements do not provide the same point-in-time contract | `FinnhubProvider` |
| [Massive stock financials](https://massive.com/docs/rest/stocks/fundamentals/income-statements) | US company statements and ratios from SEC-derived data | REST API | API key; plan-dependent | A filing date can identify the latest filing that contains a period rather than its first publication | `MassiveProvider` for company metrics only |
| [Sharadar Core US Fundamentals](https://sharadar.com/docs/fundamentals) | Active and delisted US companies from 1998 with as-reported and restated dimensions | API and bulk delivery | Paid subscription | Select AR dimensions for as-reported research and preserve `datekey`; MR dimensions are restated | No |
| [SimFin fundamentals](https://www.simfin.com/en/fundamental-data-download/) | Standardized company statements with plan-dependent US and international history | API and bulk files | Free samples and paid tiers | Standard files can contain latest restatements; as-reported availability depends on product | No |
| [Financial Modeling Prep statements](https://site.financialmodelingprep.com/developer/docs/stable/income-statement) | Global standardized and as-reported statements with plan-dependent history | REST API and bulk products | API key; plan-dependent | Filing and acceptance dates do not by themselves establish that every superseded version is retained | No |
| [Intrinio US Fundamentals](https://intrinio.com/products/us-fundamentals) | US SEC filers with reported and standardized statements | REST API and bulk products | Commercial license | Standardization and restatement fields are product-specific; verify the licensed endpoint | No |
| [Compustat Financials](https://www.marketplace.spglobal.com/en/datasets/compustat-financials-(8)) | North American and global standardized financials | Xpressfeed, cloud, and institutional files; academic access through WRDS | Institutional license | Standard annual and quarterly files are restated; point-in-time products are separate | No |
| [FactSet Fundamentals Point-in-Time](https://www.factset.com/marketplace/catalog/product/factset-fundamentals-point-in-time) | Global normalized company fundamentals with historical versions | APIs, feeds, and managed files | Institutional license | Point-in-time coverage begins later than some standard-history products and depends on the subscription | No |
| [LSEG Company Fundamentals](https://www.lseg.com/en/data-catalogue/company-data) | Global company statements, estimates, and related reference data | APIs, feeds, cloud, and institutional files | Institutional license | As-reported, standardized, estimates, and point-in-time histories are separate products | No |

## Package Contract

Yahoo, EODHD, Finnhub, and Massive expose fundamental or company-metric methods. Statement frames
use a long schema with `period_end` and an optional `filed_at`. An empty `filed_at` means the frame
does not establish when the value became public and should not be used as point-in-time evidence.

`MassiveProvider.fetch_financials()` still targets a retired vendor endpoint. The supported mapping
above is limited to `fetch_company_metrics()` until that runtime contract changes.

## Selection Notes

- Use EDGAR or an explicitly point-in-time product when filing chronology is central.
- Include delisted issuers and a dated mapping between issuer identifiers and traded securities.
- Treat analyst estimates separately from reported facts and preserve every estimate vintage.

## Related References

- [Equity data sources](equities.md)
- [Research factor data sources](factors.md)
- [Provider comparison](index.md)
