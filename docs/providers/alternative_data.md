# Alternative Data Sources

Alternative data covers observations outside standard prices, statements, and macro series. The
main selection questions are when a record first became observable, whether history was backfilled
or revised, how representative the panel is, and whether the proposed use is permitted.

!!! note "Verified September 2026"
    Entries were checked against the linked official product documentation. Vendor ownership,
    access, and methodology change frequently.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [GDELT](https://www.gdeltproject.org/data.html) | Global news events from 1979 and full-text-derived themes and tone from 2013 | Public files, feeds, and BigQuery | Free | Older history was assembled retrospectively; event and batch timestamps have different meanings | No |
| [SEC EDGAR full-text search](https://www.sec.gov/edgar/search/efts-faq.html) | Searchable US filing text from 2001 | Public search API and filing archives | Free; fair-access policy applies | Use SEC acceptance timestamps, not reporting periods or local download times | No |
| [RavenPack News Analytics](https://www.ravenpack.com/products/news-analytics) | Entity-tagged financial news events, relevance, novelty, and sentiment | API, feeds, and managed files | Institutional; some academic access through WRDS | Vendor taxonomies and model versions can change; confirm whether historical scores are restated | No |
| [LSEG MarketPsych Analytics](https://www.lseg.com/en/data-analytics/market-data/quantitative-economic-data-solutions/marketpsych-analytics-and-models) | News and social-media sentiment across assets and macro topics | Feeds and institutional delivery | Institutional license | Contributor coverage, language models, and revision policy are product-specific | No |
| [Reddit Data API](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki) | Posts and comments subject to approved use | OAuth API and licensed products | Approval and commercial terms depend on use | Deleted content, community selection, bots, and policy changes create unstable historical samples | No |
| [X API](https://docs.x.com/x-api/getting-started/pricing) | Posts, users, and engagement subject to product access | REST and streaming APIs | Pay-per-use and enterprise access | Sampling, deletions, bot activity, and changing access tiers affect longitudinal research | No |
| [Similarweb Data and APIs](https://www.similarweb.com/corp/ourdata/) | Modeled website and app traffic and engagement | API and managed exports | Commercial license | Panel composition and estimation models are not direct observations of all traffic | No |
| [Sensor Tower App Performance](https://sensortower.com/product/mobile-app/app-performance-insights) | Estimated app downloads, revenue, rankings, and usage | API and exports | Commercial license | Modeled estimates and store coverage require product-specific validation | No |
| [Revelio Labs workforce data](https://www.reveliolabs.com/data) | Standardized job postings, employee profiles, and workforce measures | API, cloud, and managed files | Commercial license | Source-platform coverage, deduplication, and profile updates can revise history | No |
| [Consumer Edge transaction data](https://consumer-edge.com/data/) | Aggregated consumer transaction panels and company metrics | Feeds and managed files | Institutional license | Panel representativeness, merchant mapping, and privacy controls are central to validity | No |
| [Foursquare Movement](https://foursquare.com/products/movement/) | Aggregated foot-traffic and visitation measures | API and managed delivery | Commercial license | Device-panel composition, venue mapping, and privacy thresholds can change coverage | No |
| [Planet Monitoring](https://www.planet.com/products/monitoring/) | Repeated satellite imagery over land areas | API and cloud delivery | Commercial license; research programs may apply | Clouds, revisit gaps, sensor changes, and labeling choices can dominate a derived signal | No |
| [MarineTraffic API](https://www.marinetraffic.com/en/ais-api-services) | AIS vessel positions, port calls, and voyage information | REST and streaming APIs | Commercial license | Terrestrial and satellite AIS have different coverage; vessel identifiers and reported fields can be missing or false | No |
| [OpenWeather API](https://openweathermap.org/api) | Current, forecast, and plan-dependent historical weather | REST API and bulk products | API key; plan-dependent | Station and model changes, spatial interpolation, and forecast vintage determine valid use | No |
| [USGS EarthExplorer](https://earthexplorer.usgs.gov/) | Public remote-sensing and geospatial archives, including Landsat products | Search API, bulk download, and cloud mirrors | Free account for some downloads | Sensor generations, processing levels, and cloud masks must be made consistent | No |

Prediction markets are documented separately because contract rules, order books, and settlement
make them a market-data category rather than a general alternative-data feed. See
[Prediction-Market Data Sources](prediction_markets.md).

## Selection Notes

- Record first-availability time separately from event time and observation period.
- Ask whether backfills, panel changes, or model updates rewrite history.
- Validate coverage by company, geography, sector, and time before testing a signal.
- Resolve provenance, privacy, material non-public information, and redistribution rights before
  acquisition.

## Related References

- [Fundamental data sources](fundamentals.md)
- [Prediction-market data sources](prediction_markets.md)
- [Provider comparison](index.md)
