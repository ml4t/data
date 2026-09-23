# Alternative Data Sources

This page maps sources of alternative data: news and text, social and web attention, app and
web usage, job postings, card transactions, foot traffic, satellite and shipping data, ESG
ratings, supply-chain relationships, and prediction markets, plus the marketplaces where such
datasets are found. For each source it lists what the data measures, how far back it goes, what
is known about its point-in-time behavior, the access tier, and whether `ml4t-data` wraps it.

!!! note "Verified September 2026"
    Alternative data vendors are acquired, merged and repackaged often. Every entry below was
    checked in September 2026 against the vendor's own site, a press release, or published reporting. Few
    vendors publish their timestamp and revision policy; where it is not public, the table says
    "not documented" and the question belongs in your due diligence. Confirm the current terms
    before you build on any of them.

---

## Choosing a Source

### When was each record first observable?

A backtest is only valid if every record is used after the time it could first have been seen.
Ask the vendor for the timestamp that marks first availability, not the event date, and whether
history was backfilled when the dataset launched or when coverage expanded. A backfilled history
looks continuous but was assembled later, often by a method that did not exist at the time.

### Revisions and methodology changes

Panels change composition (card and device panels gain and lose users), models are re-estimated,
and rating methodologies are revised. ESG ratings are the clearest case: providers disagree with
each other and revise their own methods over time. Ask whether old values are restated when the
method changes and whether earlier vintages are kept.

### Coverage and representativeness

Check which names, sectors and regions a dataset actually covers, and whether coverage is stable
across the sample. A signal that exists only for large US consumer companies is a narrower
signal than its marketing suggests.

### Legal and privacy

Provenance, permitted use and redistribution rights are hard constraints. Narrow panels can carry
material non-public information, and granular location or device data carries privacy risk even
without names. These questions are answered before any backtest, not after.

### In the book

- Chapter 4, "Understanding alternative data" (section 4.4): "The alternative data landscape",
  "The evaluation framework", "Build versus buy for alternative data", and "Data sources and
  implementation".
- Chapter 4, "Using text data for NLP features" (section 4.5): extracting and storing text from
  filings point-in-time. Chapter 10 covers turning text into features.
- Chapter 2, "A modern taxonomy of financial data" (section 2.1), subsection "Alternative data -
  High variety, high validation burden".

---

## News, Text and Sentiment

| Source | What it measures | History | Point-in-time and revisions | Access | ml4t-data |
|--------|------------------|---------|-----------------------------|--------|-----------|
| [GDELT Project](https://gdeltproject.org/data.html) | Global news events and tone (Event database); themes, entities and sentiment from full text (Global Knowledge Graph) | Events from 1979; Knowledge Graph from April 2013 | Published in 15-minute batches; the batch time is a usable first-availability time. Older history was built retrospectively. | Free and open | No |
| [RavenPack](https://www.ravenpack.com/) | Entity-level news analytics: relevance, sentiment and event classification | From 2000 (Dow Jones edition), 2007 (web edition) | Records carry the news timestamp; revision policy not documented publicly | Institutional; academic via WRDS | No |
| [LSEG MarketPsych Analytics](https://www.lseg.com/en/data-analytics/market-data/quantitative-economic-data-solutions/marketpsych-analytics-and-models) | Sentiment and emotion scores from news and social media across equities, macro, FX, crypto and commodities | From 1998 | Minute-level to daily series; revision policy not documented publicly | Institutional | No |
| [Dow Jones Factiva DNA](https://www.dowjones.com/business-intelligence/factiva/) | Licensed premium news archive with metadata, via API, snapshots and streams | Archive depth varies by publication | Publication timestamps per article | Institutional | No |
| [Benzinga Newsfeed API](https://docs.benzinga.com/api-reference/news-api/overview) | Financial news headlines and articles, Wilshire 5000 and TSX coverage | Not documented | `created` and `updated` timestamps per article; `updatedSince` returns changes | Paid | No |
| [SEC EDGAR full-text search](https://www.sec.gov/edgar/search/efts-faq.html) | Full text of SEC filings | From 2001 | Every filing carries its acceptance timestamp | Free | No |

---

## Social Media and Web Attention

Access to social-media data has narrowed since 2023, so check the current terms before planning
a project around any of these.

| Source | What it measures | Status | Access | ml4t-data |
|--------|------------------|--------|--------|-----------|
| [Stocktwits API](https://api.stocktwits.com/developers) | Investor message stream with cashtags and self-reported sentiment | Closed to new developer registrations | Not open to new developers | No |
| [Reddit Data API](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki) | Posts and comments, including investing subreddits | Paid for commercial use since July 2023 | Commercial license required | No |
| [X API](https://docs.x.com/x-api/getting-started/pricing) | Posts and engagement | Pay-per-use credits since February 2026, replacing subscription tiers | Paid | No |

---

## Web, App and Workforce Data

| Source | What it measures | History | Point-in-time and revisions | Access | ml4t-data |
|--------|------------------|---------|-----------------------------|--------|-----------|
| [Similarweb](https://www.similarweb.com/corp/ourdata/) | Website and app traffic and engagement estimates | About 10 years | Modeled estimates; revision policy not documented publicly | Free tools, paid, enterprise | No |
| [Sensor Tower](https://sensortower.com/product/mobile-app/app-performance-insights) | App downloads, revenue, rankings and usage estimates; includes the former data.ai | Not documented | Modeled estimates; revision policy not documented publicly | Enterprise; free top charts | No |
| [Thinknum](https://www.thinknum.com/) | Web-collected company data: job listings, headcount, product prices, store locations | Not documented | Not documented | Institutional | No |
| [YipitData](https://www.yipitdata.com/) | Company KPI and revenue estimates from web, app and transaction data | Not documented | Not documented | Institutional | No |
| [Revelio Labs](https://www.reveliolabs.com/data) | Workforce composition and flows, job postings, employee sentiment, layoff notices | Workforce composition from 2007, transitions from 2008, postings from 2021 | Not documented | Paid; academic via WRDS | No |
| [LinkUp](https://www.linkup.com/data) | Job postings collected directly from employer websites | From 2007 | New, updated and removed listings are captured nightly | Institutional | No |

---

## Card Transactions

Card panels measure the spending of a sample of consumers. Panel composition changes over time,
so ask how the vendor weights the panel and whether history is reweighted when it changes.

| Source | What it measures | History | Access | ml4t-data |
|--------|------------------|---------|--------|-----------|
| [Bloomberg Second Measure](https://secondmeasure.com/) | US consumer card spending by merchant and company | 8+ years; 2 to 7 day reporting lag | Bloomberg Terminal and enterprise | No |
| [Consumer Edge Transact](https://www.consumeredge.com/products/transact/) | Card transactions by merchant, company and demographic; now includes Earnest Analytics | About 9 years | Institutional | No |
| [Facteus](https://facteus.com/) | Debit and credit card transactions, synthesized for privacy | Not documented; about 1-day lag | Institutional feeds (API, AWS, Snowflake) | No |

---

## Foot Traffic, Satellite and Shipping

| Source | What it measures | History | Access | ml4t-data |
|--------|------------------|---------|--------|-----------|
| [Placer.ai](https://www.placer.ai/products/api) | Visits to stores and venues from a mobile-device panel | Not documented | Free tools, paid platform, enterprise API | No |
| [Advan Research Patterns+](https://advanresearch.com/products/patternsplus) | Foot traffic to points of interest, US and Canada; took over SafeGraph's Patterns product | Weekly updates; history from 2019 | Institutional | No |
| [Planet Labs](https://www.planet.com/industries/education-and-research/) | Satellite imagery: near-daily medium resolution and tasked high resolution | Archive from 2009 (RapidEye, 2009 to 2020) | Commercial; education and research program for university users | No |
| [RS Metrics](https://rsmetrics.com/) | Asset-level signals from satellite imagery: metals production at smelters, retail parking lots | Not documented | Institutional; data for research | No |
| [Kpler](https://www.kpler.com/product/commodities) | Commodity flows from ship tracking (AIS), satellite, customs and port data; now owns MarineTraffic and Spire Maritime | Not documented | Institutional | No |

---

## ESG

ESG ratings disagree across providers and change when methodologies change. Treat a rating
history as a sequence of vintages, and check which methodology produced each value.

| Source | What it measures | History and methodology | Access | ml4t-data |
|--------|------------------|-------------------------|--------|-----------|
| [MSCI ESG Ratings](https://www.msci.com/data-and-analytics/sustainability-solutions/esg-ratings) | Industry-relative ESG ratings (AAA to CCC), about 17,000 issuers | Time series since 2007 | Institutional; free public lookup for a subset | No |
| [Morningstar Sustainalytics ESG Risk Ratings](https://www.sustainalytics.com/esg-data) | Unmanaged ESG risk score, 16,000+ companies | A score change log exists; vintage policy not documented publicly | Institutional | No |
| [LSEG ESG Scores](https://www.lseg.com/en/media-centre/press-releases/2026/lseg-launches-new-suite-esg-scores-sustainability-analytics) | ESG scores from 220+ indicators, plus controversies | New suite launched March 2026 on 220 standardised indicators; how it relates to earlier LSEG scores is not documented publicly | Institutional | No |
| [S&P Global ESG Scores](https://www.marketplace.spglobal.com/en/datasets/s-p-global-esg-scores-(171)) | Scores from the annual Corporate Sustainability Assessment | Annual assessments; history not documented publicly | Institutional | No |
| [RepRisk](https://www.reprisk.com/insights/resources/methodology) | Reputational and conduct risk from news and stakeholder sources | Daily series with a consistent methodology since January 2007 | Institutional | No |

---

## Supply Chain and Trade

| Source | What it measures | History | Access | ml4t-data |
|--------|------------------|---------|--------|-----------|
| [FactSet Supply Chain Relationships](https://www.factset.com/marketplace/catalog/product/factset-supply-chain-relationships) (formerly Revere) | Customer, supplier, partner and competitor links between companies | North America from 2003; other regions from 2011 to 2016 | Institutional; academic via WRDS | No |
| [Bloomberg Supply Chain (SPLC)](https://professional.bloomberg.com/institutions/corporations/supply-chain/) | Supplier and customer relationships with revenue and cost exposure, 100,000+ companies | From 2006 | Bloomberg Terminal and enterprise feed | No |
| [S&P Global Panjiva](https://www.marketplace.spglobal.com/en/datasets/panjiva-supply-chain-intelligence-(22)) | Shipment-level customs records, 2 billion+ records | Not documented | Institutional | No |
| [ImportGenius](https://www.importgenius.com/pricing) | US and international bill-of-lading records | US imports from 2006, exports from January 2014 on the Pro and Enterprise tiers | Self-serve subscriptions; enterprise plans | No |

---

## Prediction Markets

Prediction-market prices are market data: each trade and quote has an exchange timestamp and is
never revised. The caveats are thin liquidity outside headline contracts and markets that are
listed and resolved over time, so a set of markets picked today is a survivor sample.

| Source | What it measures | History | Access | ml4t-data |
|--------|------------------|---------|--------|-----------|
| [Kalshi](https://docs.kalshi.com/welcome) | CFTC-regulated event contracts: economics, politics, weather, sports | From the July 2021 launch | Public market-data API; trading needs a verified account | Yes: [`KalshiProvider`](kalshi.md) |
| [Polymarket](https://docs.polymarket.com/) | Event contracts on a crypto platform; open to US users again since December 2025 through QCEX, a CFTC-licensed exchange acquired in 2025 | From 2020 | Public market data | Yes: [`PolymarketProvider`](polymarket.md) |

---

## Marketplaces and Discovery

Marketplaces simplify procurement; they do not validate point-in-time behavior or methodology.

| Source | What it offers | Access |
|--------|----------------|--------|
| [Nasdaq Data Link](https://data.nasdaq.com/) (formerly Quandl) | Catalog of financial, economic and alternative datasets, some free | Free account for free datasets; paid for premium |
| [Snowflake Marketplace](https://docs.snowflake.com/en/collaboration/collaboration-marketplace-about) | Datasets shared into a Snowflake account and queried in place | Free, trial and paid listings |
| [AWS Data Exchange](https://aws.amazon.com/data-exchange/) | Datasets delivered into AWS, billed through AWS | Per-product subscriptions |
| [Eagle Alpha](https://www.eaglealpha.com/) | Alternative data discovery and advisory, 2,500+ dataset profiles | Institutional |
| [Neudata](https://www.neudata.co/) | Alternative data discovery and research, 7,000+ datasets in its catalog | Institutional |
| [Datarade](https://datarade.ai/platforms) | Directory of data providers and data marketplaces | Free to browse; each provider sets its own terms |

Of the free datasets that made Quandl popular, the WIKI prices table is frozen at April 2018
(`ml4t-data` wraps it as [Wiki Prices](wiki_prices.md)), and the Zillow real-estate database
no longer has a catalog page and was last refreshed in July 2025.

---

## Alternative Data in ml4t-data

`ml4t-data` wraps two prediction-market sources, [Kalshi](kalshi.md) and
[Polymarket](polymarket.md). The FXMacroData provider also exposes news, risk-sentiment and CFTC
positioning endpoints alongside its release-timestamped macro data. Everything else on this page
needs its own client.

---

## Name Changes and Closures

| Then | Now |
|------|-----|
| data.ai (formerly App Annie) | Part of Sensor Tower since 2024 |
| Earnest Analytics | Part of Consumer Edge |
| SafeGraph Patterns (foot traffic) | Sold to Advan Research; SafeGraph now sells places, geometry and spend data |
| Orbital Insight | Acquired by Privateer, May 2024 |
| MarineTraffic | Acquired by Kpler, March 2023 |
| Spire Maritime | Acquired by Kpler, April 2025 |
| Refinitiv ESG scores | LSEG ESG data; a new LSEG ESG Scores suite launched March 2026 |
| Reddit and X APIs | Paid access (Reddit since July 2023; X pay-per-use since February 2026) |
| Stocktwits API | Closed to new developers |
| Polymarket (closed to US users 2022 to 2025) | US access since December 2025 through QCEX, a CFTC-licensed exchange acquired in 2025 |
| PredictIt | CFTC approval to operate as a regulated exchange, September 2025 |
| Quandl | Nasdaq Data Link (September 2021) |

## See Also

- [Provider comparison](index.md)
- [Market data sources](market_data.md)
- [Fundamental data sources](fundamentals.md)
