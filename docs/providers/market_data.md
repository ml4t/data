# Market Data Sources

This page maps the market data vendor landscape: US equities and ETFs (daily and intraday),
futures, options, FX and crypto, including the institutional and academic sources that
`ml4t-data` does not wrap. For the providers the library does wrap (free tiers, async support,
credentials), the [provider comparison](index.md) is the reference; this page links to it rather
than repeating it.

!!! note "Verified September 2026"
    Vendor coverage, history, access terms and ownership change often. Every entry below was
    checked against the vendor's own documentation or a published announcement in September 2026.
    Where a vendor does not document a property, the table says "not documented" instead of
    guessing. Confirm the current terms before you build a dataset on any of them.

---

## Choosing a Source

### Survivorship

A research universe must contain the securities that later delisted, merged or went bankrupt.
Free price sources usually carry only what trades today, so any cross-sectional backtest built
on them overstates returns. The "Delisted" columns below say whether a vendor keeps dead
securities.

### Adjustments and corporate actions

Splits, dividends and spin-offs break raw price series. Vendors differ in what they deliver:
unadjusted prices, adjusted prices, or both with the adjustment factors. Keep the unadjusted
series and the factors if you can: adjusted history is rewritten every time a new corporate
action occurs, so an adjusted series downloaded today differs from one downloaded last year.
Futures have the equivalent problem at each roll; a continuous contract is a construction choice,
not a market price.

### Point-in-time universe membership

An index strategy must trade the constituents as of each date, not today's members. Only a few
vendors sell historical index membership; without it, "S&P 500 stocks" silently means today's
survivors.

### Granularity and timestamps

Daily bars, minute bars, trades and quotes, and full order-book messages are different products.
For US equities, a consolidated (SIP) feed and a single-venue feed (for example IEX only) report
different volumes and prices. Check which timestamp a vendor records (exchange, SIP or receipt)
and how it defines the trading session.

### Licensing

Exchange data carries its own license terms. Redistribution, display and derived-data rights
vary by vendor and tier, and a dataset licensed for research may not be licensed for production.

### In the book

- Chapter 2, "The asset-class market data landscape" (section 2.2): what each asset class lets
  you observe, its failure modes, and the engineering decisions it forces.
- Chapter 2, "A due diligence framework for data sourcing" (section 2.3): survivorship,
  corporate actions, identifier integrity, and a vendor checklist.
- Chapter 3, "The anatomy of modern market data feeds" (section 3.2) and "Microstructure data
  quality and sessionization" (section 3.6): feeds, timestamps and session handling for intraday
  data.

---

## Providers ml4t-data Wraps

Yahoo Finance, Alpaca, EODHD, Tiingo, Twelve Data, Massive (formerly Polygon.io), Finnhub,
Databento, Oanda, Binance, Binance Public, OKX, CoinGecko, CryptoCompare, Wiki Prices and NASDAQ
ITCH samples all have `ml4t-data` providers. Their asset classes, free tiers and credentials are
in the [provider comparison](index.md) and on each provider's page. Three of them appear in the
tables below as well, because they are also the reference source for a data class: Databento
(order-book and CME data), Alpaca (US equities with a stated history start) and the NASDAQ ITCH
samples.

---

## US Equities and ETFs: Daily Research Panels

| Source | Coverage and history | Delisted | Adjustments | Finest granularity | Point-in-time index membership | Access | ml4t-data |
|--------|----------------------|----------|-------------|--------------------|--------------------------------|--------|-----------|
| [CRSP US Stock Databases (on WRDS)](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/center-for-research-in-security-prices-crsp/) | NYSE from December 1925, NASDAQ from December 1972 | Yes, keyed by PERMNO and PERMCO | Returns with and without dividends; full corporate-action history | Daily and monthly | Yes, index constituent files on WRDS | Academic and institutional, mainly via WRDS | No |
| [Norgate Data](https://norgatedata.com/data-content-tables.php) | US, Australian and Canadian stocks and ETFs; US stocks essentially complete from 1992 | Yes, on the higher tiers | Four modes: unadjusted, capital-reconstruction adjusted, plus special distributions, total return | Daily | Yes, on the higher tiers | Retail paid subscription | No |
| [Sharadar Equity Prices (SEP)](https://data.nasdaq.com/databases/SEP) | About 21,000 US equities and funds, from 1998 | Yes | Unadjusted, split-adjusted, and split, dividend and spin-off adjusted | Daily | Separate S&P 500 constituents table | Paid; direct at sharadar.com since July 2026, or Nasdaq Data Link | No |
| [FirstRate Data](https://firstratedata.com/about/FAQ) | 16,000+ US tickers from 2000, including 7,000+ delisted | Partly: included where the vendor could source them | Unadjusted, split-adjusted, and split and dividend adjusted | 1-minute bars | No | Retail paid, one-off purchase | No |
| [Kibot](https://www.kibot.com/faq.html) | US stocks and ETFs; minute bars from 1998 | Partly: active and delisted lists, stated as incomplete | Split and dividend adjusted series plus an adjustments API | Tick with bid and ask | No | Free guest access (daily only), paid tiers | No |
| [Alpaca](https://docs.alpaca.markets/us/docs/about-market-data-api) | US equities and ETFs, from 2016 | Not documented | `adjustment` parameter: raw, split, dividend or all | Trades and quotes | No | Free (IEX feed, delayed SIP); paid for real-time SIP | Yes: `AlpacaProvider` |

The free wrapped sources (Yahoo Finance, Tiingo, EODHD free tier) carry currently listed
securities and vendor-adjusted prices; treat them as prototyping data, not as a survivorship-free
panel. Wiki Prices stopped updating in April 2018.

---

## US Equities: Trades, Quotes and Order Books

| Source | Coverage and history | Delisted | Adjustments | Finest granularity | Point-in-time index membership | Access | ml4t-data |
|--------|----------------------|----------|-------------|--------------------|--------------------------------|--------|-----------|
| [NYSE TAQ](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/) (Daily TAQ) | All US exchanges from 1993; millisecond timestamps from 2003 | Not applicable: daily files of every security traded that day | Raw prices; adjust with CRSP factors | Trades and NBBO quotes | No | Academic and institutional via WRDS | No |
| [AlgoSeek](https://algoseek.com/us-equities-package/) | About 27,500 US securities from 1998 (SIP); adjustment factors from 2007 | Yes | Raw and adjusted prices plus adjustment-factor files | Trades and quotes with nanosecond timestamps; bars from 1 second | Yes, Index Components dataset | Institutional; free sandbox with up to one year of data | No |
| [Databento](https://databento.com/datasets/XNAS.ITCH) | US equities across exchanges and ATSs; Nasdaq TotalView-ITCH from 2018 | Not applicable: raw exchange feeds | Unadjusted; corporate actions sold as separate reference data | Full order book (market by order) | No | Usage-based; see the [Databento page](databento.md) | Yes: `DataBentoProvider` |
| [LSEG Tick History](https://www.lseg.com/en/data-analytics/market-data/data-feeds/tick-history) (formerly Refinitiv Tick History) | Cross-asset, 500+ venues, from 1996 | Not documented | Not documented | Trades, quotes and order-book depth | No | Institutional; some universities license it | No |
| [Nasdaq TotalView-ITCH samples](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/) | Selected full-day files for Nasdaq-listed stocks | Not applicable | Raw messages | Every order add, cancel and execution | No | Free samples; the full archive is a paid Nasdaq product | Yes: [NASDAQ ITCH provider](nasdaq_itch.md) |
| [DTN IQFeed](https://www.iqfeed.net/dev/) | Equities, futures, options, FX; tick history rolling 180 days, minute bars from 2005 to 2007 depending on asset | Not documented | Not documented | Trades and quotes | No | Retail paid | No |
| [Kinetick](https://ninjatrader-live.ninjatrader.com/support/helpguides/nt8/data_by_provider.htm) | Market data service for NinjaTrader users; equities, futures, FX | Not documented | Not documented | Tick | No | Retail; free end-of-day tier | No |

---

## Futures

| Source | Coverage and history | Expired contracts | Continuous series | Finest granularity | Access | ml4t-data |
|--------|----------------------|-------------------|-------------------|--------------------|--------|-----------|
| [CME DataMine](https://www.cmegroup.com/datamine.html) | CME, CBOT, NYMEX and COMEX futures and options; history varies by dataset | Not documented | No, individual contracts | Market depth and time and sales | Paid per dataset; academic discount | No |
| [Databento CME Globex (GLBX.MDP3)](https://databento.com/datasets/GLBX.MDP3) | CME Group futures and options from June 2010 | Yes, per contract | Continuous symbology by roll rule | Full order book | Usage-based | Yes: `DataBentoProvider.fetch_continuous_futures()` |
| [Norgate Data](https://norgatedata.com/data-content-tables.php) | Global futures and cash commodities | Yes | Back-adjusted continuous contracts | Daily | Retail paid | No |
| [FirstRate Data](https://firstratedata.com/about/FAQ) | Futures, about 15 years intraday | Not documented | Unadjusted, absolute-adjusted and ratio-adjusted continuous series | 1-minute bars | Retail paid | No |

---

## Options

| Source | Coverage and history | Delisted underlyings | Analytics | Finest granularity | Access | ml4t-data |
|--------|----------------------|----------------------|-----------|--------------------|--------|-----------|
| [OptionMetrics IvyDB](https://optionmetrics.com/united-states/) | US equity and index options from 1996; Europe from 2002 | Yes | Implied volatility, Greeks, volatility surfaces; adjusts for dividends, splits and spin-offs | Daily; a separate intraday product exists | Academic via WRDS, institutional | No |
| [ORATS](https://orats.com/data-api) | US equity, ETF and index options (no futures options); end of day from 2007, 1-minute from August 2020 | Yes | Smoothed implied volatility and Greeks | 1-minute snapshots | Retail paid, enterprise | No |
| [Cboe DataShop](https://datashop.cboe.com/documentation) (LiveVol) | Options quotes, trades and end of day from 2010; VIX futures from 2004 | Not documented | Calculated implied volatility and Greeks in some datasets | Trades and quotes | Paid, with trial | No |
| [Databento OPRA](https://databento.com/datasets/OPRA.PILLAR) | All US equity options from March 2023 | Not applicable | None: raw feed | Trades and quotes | Usage-based | Yes: `DataBentoProvider.fetch_option_chain()`, `fetch_option_quotes()` |
| [AlgoSeek](https://algoseek.com/us-equities-package/) | US options as an add-on to equities | Not documented | Vendor-computed analytics on some products | Trades and quotes | Institutional | No |

---

## Foreign Exchange

Spot FX trades over the counter, so there is no consolidated tape: every source reports one
dealer's or one venue's prices, and volumes are not comparable across sources.

| Source | Coverage and history | Finest granularity | Access | ml4t-data |
|--------|----------------------|--------------------|--------|-----------|
| [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) | FX, CFDs, commodities, indices from one Swiss bank's platform | Tick | Free download | No |
| [TrueFX](https://www.truefx.com/truefx-historical-downloads-2/) | Major and cross pairs, top of book | Tick, millisecond timestamps | Free historical downloads with registration; real-time is institutional | No |
| [LSEG Tick History](https://www.lseg.com/en/data-analytics/market-data/data-feeds/tick-history) | Contributed and venue FX quotes from 1996 | Tick | Institutional | No |
| [Oanda](oanda.md) | One broker's quotes | Seconds-level candles | Account required | Yes: `OandaProvider` |

---

## Crypto

Crypto trades on many venues around the clock, and reported volume on some venues is unreliable.
A source that keeps delisted pairs and closed exchanges (FTX, for example) matters for the same
survivorship reason as delisted stocks.

| Source | Coverage and history | Delisted pairs and venues | Finest granularity | Access | ml4t-data |
|--------|----------------------|---------------------------|--------------------|--------|-----------|
| [Kaiko](https://www.kaiko.com/products/l1-l2-data) | 100+ centralized exchanges plus DeFi; trades from 2010, order-book snapshots from 2015 | Not documented | Trades, best bid and offer, full order book | Institutional | No |
| [Tardis.dev](https://docs.tardis.dev/historical-data-details/overview) | 50+ exchanges, spot, derivatives and options; most from March 2019 | Yes: closed venues such as FTX and BitMEX history retained | Raw exchange messages, full order-book depth | Paid, including an academic tier | No |
| [CoinAPI](https://www.coinapi.io/products/market-data-api) | 400+ exchanges; spot, futures, perpetuals, options | Not documented | Trades, quotes, order books | Free credits, paid, enterprise | No |
| Binance, Binance Public, OKX, CoinGecko, CryptoCompare | Venue or aggregator data; see the [provider comparison](index.md) | Venue-specific | Bars; funding rates and premium index for perpetuals | Free or free tier | Yes |

---

## Name Changes and Closures

| Then | Now |
|------|-----|
| CRSP (University of Chicago) | Owned by Morningstar since February 2, 2026; CRSP indexes renamed Morningstar indexes |
| Thomson Reuters / Refinitiv Tick History | LSEG Tick History |
| Polygon.io | Massive (October 30, 2025); `api.polygon.io` still answers alongside `api.massive.com` |
| Sharadar on Nasdaq Data Link only | Also sold direct at sharadar.com since July 2026 |
| Quandl WIKI Prices | Frozen since April 2018; still available as a static dataset |
| IEX Cloud | Retired August 31, 2024 |

## See Also

- [Provider comparison](index.md)
- [Fundamental data sources](fundamentals.md)
- [Alternative data sources](alternative_data.md)
