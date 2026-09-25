# Macroeconomic Data Sources

Macroeconomic observations are released, revised, benchmarked, and sometimes replaced. A valid
historical model uses the vintage available at the decision date, not the latest value assigned to
an old observation period.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Coverage and access
    depend on the source and can change.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [FRED and ALFRED](https://fred.stlouisfed.org/docs/api/fred/) | US and international economic and financial series from many contributing agencies; ALFRED stores vintages | REST API | Free API key | FRED's latest value can include revisions; use real-time periods or ALFRED for point-in-time research | `FREDProvider` |
| [FXMacroData](https://fxmacrodata.com/) | FX-oriented macroeconomic and reference-rate series | REST API | Public USD endpoints or API key | Coverage and release metadata vary by endpoint; reference series are not executable FX quotes | `FXMacroDataProvider` |
| [US Bureau of Labor Statistics](https://www.bls.gov/developers/) | Employment, prices, productivity, compensation, and related US labor statistics | Public Data API and files | Free; registration key raises limits | Seasonal adjustment and annual benchmark revisions can alter history | No |
| [US Bureau of Economic Analysis](https://apps.bea.gov/api/) | National, industry, international, and regional economic accounts | REST API | Free API key | Vintage availability differs by dataset; later benchmark revisions can be large | No |
| [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392) | Nearly 16,000 indicators across more than 45 databases, with many series extending over 50 years | REST API and bulk downloads | Free, no key | Country definitions, source agencies, observation status, and revisions differ by indicator | No |
| [Eurostat APIs](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/) | European economic, demographic, trade, and social statistics | Statistics and SDMX REST APIs plus bulk files | Free | The public database exposes the latest dataset and does not preserve prior versions | No |
| [ECB Data Portal](https://data.ecb.europa.eu/help/getting-data-web-services-sdmx-0) | Euro-area monetary, financial, market, banking, and economic statistics | SDMX REST API and downloads | Free | Frequency, seasonal adjustment, and revision policy are series-specific | No |

## Release-Time Checklist

- Store observation period, publication timestamp, revision or vintage timestamp, units,
  seasonal-adjustment status, and source agency.
- Join low-frequency releases to the first trading session after publication. Do not forward-fill
  from the observation period before the release occurred.
- Treat forecasts, flash estimates, and final releases as separate vintages.

## Related References

- [Foreign exchange data sources](fx.md)
- [Fixed-income data sources](fixed_income.md)
- [Prediction-market data sources](prediction_markets.md)
