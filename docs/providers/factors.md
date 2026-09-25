# Research Factor Data Sources

Factor libraries publish returns from research portfolios, not returns from investable products.
Choose a source by universe, formation method, weighting, frequency, and revision policy. Do not
combine similarly named factors without reconciling their construction.

!!! note "Verified September 2026"
    This category has enough distinct, maintained sources for a standalone reference. Entries were
    checked against the linked official or author-maintained data page.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | US and international factors, sorted portfolios, industries, and breakpoints; core US monthly series begin in 1926 | Downloadable ZIP, CSV, and text files | Free | Histories can be reconstructed after upstream CRSP revisions; archive vintages when reproducibility matters | `FamaFrenchProvider` |
| [AQR Data Library](https://www.aqr.com/Insights/Datasets) | Equity and cross-asset value, momentum, quality, low-beta, trend, and long-history research portfolios | Downloadable spreadsheets | Free under dataset terms | Series are hypothetical research portfolios, not AQR product returns; methodology differs by paper | `AQRFactorProvider` |
| [Global Factor Data](https://jkpfactors.com/data) | 153 characteristics across 93 countries and four regions, with portfolio sorts and reference files | Configurable file downloads; stock-level data through WRDS | Factor returns free under CC BY-NC 4.0; stock-level access requires WRDS | Noncommercial data license and global data-screening choices constrain reuse and comparison | No |

`FamaFrenchProvider.fetch()` selects supported factor, portfolio, industry, international, or
breakpoint datasets. `AQRFactorProvider.download()` acquires the spreadsheets and `fetch()` reads a
supported local dataset. Neither class computes live portfolio returns.

## Selection Notes

- Use Fama-French for canonical academic benchmarks and portfolio sorts.
- Use AQR for paper-specific equity and cross-asset premia such as QMJ, BAB, VME, and TSMOM.
- Use Global Factor Data for broad characteristic definitions and consistent country coverage.
- Store the downloaded file and retrieval date because published research histories may be
  revised.
