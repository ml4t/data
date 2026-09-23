# Fundamental Data Sources

This page compares sources of equity fundamentals: financial statements, as-reported versus
restated values, and analyst estimates. It covers the free SEC routes, retail APIs, and the
institutional and academic databases, and says which ones `ml4t-data` can fetch.

!!! note "Verified 2026-09"
    Vendor coverage, access terms, endpoint names and ownership change often. Every entry below
    was checked against the vendor's own documentation in September 2026. Confirm the current terms
    before you build a dataset on any of them.

---

## Choosing a Source

### Point-in-time first

A backtest must only see the value that was public at the decision date. Companies restate
earlier periods, and vendors often overwrite the original figure with the restated one. A
dataset that stores only the latest value leaks future information into every historical
signal built from it, and the leak is invisible in the data itself.

A source is point-in-time when each value carries the date it became public and superseded
versions are kept. The tables below use three grades:

| Grade | Meaning |
|-------|---------|
| **Yes** | Each value carries its filing or release date, and earlier versions of a restated period remain available. The field that makes it so is named. |
| **Partial** | A date is present but has a caveat, for example the date of the latest filing that included the period rather than the first, or point-in-time only in a paid dataset. |
| **No** | Only the current (possibly restated) value, with no date of public availability. |

Two further points apply even with a point-in-time source:

- A filing date is not a trading date. Filings arrive during or after the session; use the
  acceptance timestamp where available and apply the value from the next session.
- The SEC XBRL Frames API returns one value per company and period, the last one filed. It
  looks convenient for cross-sections and is not point-in-time.

### Survivorship

The universe must include companies that were later delisted, acquired or went bankrupt. A
source that covers only today's listed companies biases every cross-sectional result toward
survivors. Sharadar, Zacks, Compustat and CRSP state delisted coverage explicitly; for the others,
check before relying on it.

### Identifier mapping

Tickers change and are reused. Fundamentals are keyed by the issuer (SEC CIK, Compustat GVKEY,
vendor IDs); prices are usually keyed by the security (ticker, CRSP PERMNO, FIGI). Joining the two
needs a dated mapping, and building one is usually more work than fetching either dataset.

### In the book

- Chapter 2, "A due diligence framework for data sourcing" (section 2.3): point-in-time
  correctness, survivorship bias, identifier integrity, vendor tiers, and a vendor due diligence
  checklist.
- Chapter 4, "The point-in-time pipeline" (section 4.1): building a bitemporal fundamentals
  dataset from SEC EDGAR, and why the Frames API is not safe for backtests. "Entity resolution and
  mapping" (section 4.2) covers the identifier problem.

---

## Free SEC Routes

The SEC publishes every XBRL financial statement that US registrants file. Detailed XBRL tagging
phased in from 2009 (large filers first) to 2011. Access needs no key; requests must send a
`User-Agent` header that identifies the caller, and the SEC's fair-access policy limits request
rates.

| Source | What it returns | Point-in-time | Estimates | ml4t-data |
|--------|-----------------|---------------|-----------|-----------|
| [EDGAR `companyfacts` / `companyconcept` APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Every XBRL fact for one company (all concepts, or one concept), from every filing that reported it | **Yes**: each fact carries `filed`, `accn` (accession number) and `form` | No | No |
| [EDGAR `submissions` API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Filing history per company: forms, filing and acceptance dates, former names and tickers | Filing metadata, used to date the facts above | No | No |
| [EDGAR `frames` API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | One value per company for a concept and calendar period | **No**: the last-filed value for each period | No | No |
| [Financial Statement Data Sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets) and [Financial Statement and Notes Data Sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets) | Quarterly bulk files from 2009: submissions (`sub`), numeric values (`num`), tags, and for the Notes sets the footnote detail | **Yes**: values are as filed; `sub` holds `filed` and `accepted` per submission | No | No |

`companyfacts.zip` and `submissions.zip` provide the same content as the APIs in one nightly bulk
download. Python wrappers such as `edgartools` parse filings and statements on top of these
endpoints.

---

## Retail and Developer APIs

| Source | Coverage | Point-in-time | Estimates | Access | ml4t-data |
|--------|----------|---------------|-----------|--------|-----------|
| [Sharadar Core US Fundamentals (SF1)](https://sharadar.com/docs/fundamentals) | About 18,000 active and delisted US companies, from 1998 | **Yes**: as-reported dimensions (`ARQ`, `ARY`, `ART`) are indexed by `datekey`, the SEC filing date. The most-recent dimensions (`MRQ`, `MRY`, `MRT`) are restated. | No | Paid. Sold direct at sharadar.com since July 2026; institutional access remains through Nasdaq Data Link. | No |
| [SimFin](https://www.simfin.com/en/fundamental-data-download/) | About 5,000 US stocks, from 2003 | **Partial**: rows carry `Publish Date` and `Restated Date`, but the standard datasets hold the latest restated value. As-reported bulk datasets are on paid tiers. | No | Free tier (5 years of fundamentals, delayed bulk download); paid tiers extend history. | No |
| [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/stable/income-statement) | Global, 70,000+ securities; up to 30+ years for large caps | **Partial**: statements carry `filingDate` and `acceptedDate`; separate as-reported endpoints return the filed values. Confirm how a restatement is versioned. | Yes | Free tier (US only); paid tiers for global coverage and full history. | No |
| [EODHD](https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds) | 70+ exchanges; major US companies from 1985, non-US from 2000 | **Partial**: each yearly and quarterly entry has `filing_date`; the documentation does not say whether values are restated. | Yes: current consensus and revision trend | Paid; fundamentals are not in the free plan. | Yes: `EODHDProvider.fetch_financials()`, `fetch_company_metrics()` |
| [Intrinio](https://intrinio.com/products/us-fundamentals) | US SEC filers; annual from 2007, quarterly from 2009 | **Yes**: each fundamental has `filing_date`, and `is_latest` marks whether a later filing superseded it. Both reported and standardized feeds. | Yes: Zacks estimates, sold separately | Paid | No |
| [Zacks](https://zacksdata.com/datasets/fundamental-data/) | US and Canada; 19,500+ companies including 9,000+ delisted on Nasdaq Data Link (ZFA); Zacks Data history from 1979 | **Yes** in Zacks Data's point-in-time history; check the specific Nasdaq Data Link table before assuming it | Yes: Zacks consensus is the core product | Paid: Nasdaq Data Link, or Zacks Data for institutions | No |
| [Finnhub](https://finnhub.io/docs/api/financials-reported) | Global standardized statements; as-reported statements from SEC filings | **Partial**: `/stock/financials-reported` returns each filing with `filedDate` and `acceptedDate`. The standardized `/stock/financials` endpoint has no filing date. | Yes (premium) | Free tier includes as-reported statements; standardized statements and estimates are premium. | Yes: `FinnhubProvider.fetch_financials()` (standardized endpoint), `fetch_company_metrics()` |
| [Massive](https://massive.com/docs/rest/stocks/fundamentals/income-statements) (formerly Polygon.io) | About 6,700 US companies, from 2009 | **Partial**: `filing_date` is the date of the most recent SEC filing that included the period, not the original filing. Use the EDGAR filings index for the first filing date. | No | Paid: Stocks Advanced plan or the Financials & Ratios expansion. | Partial: `MassiveProvider.fetch_company_metrics()` (ratios) |
| [Tiingo](https://www.tiingo.com/documentation/fundamentals) | 5,500+ US equities and ADRs, 20+ years | **Partial**: `asReported=true` returns as-reported instead of restated values; `date` is the release date | No | Paid add-on; a free evaluation covers the Dow 30 for 3 years. | No (the Tiingo provider covers prices) |
| Yahoo Finance, via `yfinance` (unofficial) | Most listed tickers; about five annual and five quarterly periods | **No**: latest values only, no filing date | Yes: current consensus only | Free, unofficial; Yahoo's terms restrict use. | Yes: `YahooFinanceProvider.fetch_financials()`, `fetch_company_metrics()` |

---

## Institutional and Academic Databases

These are licensed by firms and universities. Students usually reach them through
[WRDS](https://wrds-www.wharton.upenn.edu/) (Wharton Research Data Services) when their
institution subscribes; which databases are available depends on the institution's licenses.

| Source | Coverage | Point-in-time | Estimates | Access | ml4t-data |
|--------|----------|---------------|-----------|--------|-----------|
| [Compustat](https://www.marketplace.spglobal.com/en/datasets/compustat-financials-(8)) (S&P Global Market Intelligence) | North America and Global | **No** in the standard annual and quarterly files, which hold restated values | No (see Capital IQ) | Institutional; academic via WRDS | No |
| Compustat Point-in-Time and Compustat Snapshot | North America, snapshots from 1987 | **Yes**: every value as it was known at each snapshot date, including preliminary figures | No | Institutional; Snapshot via WRDS where licensed | No |
| CRSP/Compustat Merged (CCM) | US; links CRSP securities (PERMNO) to Compustat issuers (GVKEY) | Inherits the Compustat file it is joined to; the link table itself is dated | No | Academic via WRDS. CRSP was acquired by Morningstar in February 2026. | No |
| [S&P Capital IQ Financials](https://www.marketplace.spglobal.com/en/datasets/s-p-capital-iq-financials-(10)) | Global | **Yes**: Premium Financials carries filing dates for the full history and product delivery dates from 2004 | Yes: Capital IQ Estimates | Institutional (Capital IQ Pro, Xpressfeed, Snowflake); some via WRDS | No |
| [FactSet Fundamentals](https://www.factset.com/marketplace/catalog/product/factset-fundamentals-point-in-time) | Global | **Yes**: FactSet Fundamentals Point-in-Time, from February 1999 | Yes: FactSet Estimates, with point-in-time consensus | Institutional | No |
| [LSEG Company Fundamentals](https://www.lseg.com/en/data-catalogue/company-data) (includes Worldscope) and [I/B/E/S](https://www.lseg.com/en/data-catalogue/company-data/ibes-estimates/broker-estimates) (formerly Refinitiv, Thomson Reuters) | Fundamentals: 120,000+ companies on 150+ exchanges, US from the early 1980s, other markets from the 1990s. I/B/E/S: North America from 1976, other markets from 1987 | **Yes**: point-in-time versions of fundamentals and estimates are offered | Yes: I/B/E/S | Institutional; academic via WRDS where licensed | No |
| [Bloomberg](https://professional.bloomberg.com/products/data/enterprise-catalog/investment-research-data/) | Global | **Yes**: Company Financials, Estimates and Pricing Point-in-Time through Data License | Yes: Bloomberg Estimates (BEst) | Institutional (Terminal, Data License) | No |

---

## Fundamentals in ml4t-data

The Yahoo, EODHD, Finnhub and Massive providers expose `fetch_financials()` and
`fetch_company_metrics()`. Statements come back in one long-format schema:

```python
from ml4t.data.providers import EODHDProvider

provider = EODHDProvider()  # reads EODHD_API_KEY
income = provider.fetch_financials("AAPL", statement="income", period="quarterly")
# columns: symbol, provider, statement_type, period_type, period_end, line_item, value,
#          currency, fiscal_year, fiscal_period, filed_at, source
```

`filed_at` is filled when the vendor response carries a filing date (`filing_date`, `filingDate`,
`filedDate` or `acceptedDate`) and is empty otherwise. Treat `period_end` as the end of the
reporting period, never as the date the value was known. When `filed_at` is empty, the frame is
not point-in-time; use it for exploration, not for backtests.

`MassiveProvider.fetch_financials()` still targets the retired Financials endpoint; Massive now
serves statements from `/stocks/financials/v1/income-statements`, `balance-sheets` and
`cash-flow-statements`.

---

## Name Changes and Closures

| Then | Now |
|------|-----|
| Quandl | Nasdaq Data Link (September 2021); legacy `quandl` names persist in URLs and code |
| Polygon.io | Massive (October 30, 2025); `api.polygon.io` still answers alongside `api.massive.com` |
| Refinitiv (Thomson Reuters Financial & Risk) | LSEG Data & Analytics |
| CRSP (University of Chicago) | Owned by Morningstar since February 2, 2026; CRSP indexes are being renamed Morningstar indexes |
| Sharadar on Nasdaq Data Link only | Also sold direct at sharadar.com since July 2026 |
| IEX Cloud | Retired August 31, 2024; its endpoints no longer answer |

## See Also

- [Provider reference](index.md)
- [EODHD](eodhd.md), [Finnhub](finnhub.md), [Massive](massive.md), [Yahoo Finance](yahoo.md)
