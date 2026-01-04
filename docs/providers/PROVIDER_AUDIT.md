# Provider Audit Report

*Last Updated: 2025-12-27*

This document provides a comprehensive audit of all 21 data providers in ml4t-data, including capabilities, pricing, terms, and gap analysis.

---

## Provider Summary Matrix

| Provider | Category | API Key | Free Tier | Pricing URL |
|----------|----------|---------|-----------|-------------|
| YahooFinance | Equity | No | Unlimited* | [Terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) |
| Databento | Multi-Asset | Yes | $125 credit | [Pricing](https://databento.com/pricing) |
| Polygon/Massive | Multi-Asset | Yes | 5 calls/min | [Pricing](https://massive.com/pricing) |
| EODHD | Equity | Yes | 20 calls/day | [Pricing](https://eodhd.com/pricing) |
| Tiingo | Equity | Yes | 1,000 req/day | [Pricing](https://tiingo.com/about/pricing) |
| Finnhub | Multi-Asset | Yes | 60 req/min | [Pricing](https://finnhub.io/pricing) |
| TwelveData | Multi-Asset | Yes | 800 calls/day | [Pricing](https://twelvedata.com/pricing) |
| CoinGecko | Crypto | No | 50 calls/min | [Pricing](https://www.coingecko.com/en/api/pricing) |
| Binance | Crypto | No | Generous | [API](https://www.binance.com/en/binance-api) |
| BinancePublic | Crypto | No | Unlimited | [Data](https://data.binance.vision) |
| CryptoCompare | Crypto | Yes | 250k calls/mo | [Pricing](https://min-api.cryptocompare.com/pricing) |
| Oanda | Forex | Yes | Trial | [Pricing](https://www.oanda.com/fx-for-business/fx-data-services) |
| FRED | Economic | Yes | 120 req/min | [API](https://fred.stlouisfed.org/docs/api/fred/) |
| AQR | Factor | No | Free | [Datasets](https://www.aqr.com/Insights/Datasets) |
| Fama-French | Factor | No | Free | [Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) |
| Kalshi | Prediction | No | Free | [Developer](https://kalshi.com/developer) |
| Polymarket | Prediction | No | Free | [Docs](https://docs.polymarket.com) |
| WikiPrices | Historical | No | Free | Local file |
| NASDAQ ITCH | Tick | No | Free | [Sample](https://www.nasdaqtrader.com/trader.aspx?id=HistoricalData) |
| Synthetic | Testing | No | N/A | N/A |
| Mock | Testing | No | N/A | N/A |

*Yahoo Finance: Personal use only per Terms of Service

---

## Detailed Provider Audits

### 1. Yahoo Finance (YahooFinanceProvider)

**Website**: https://finance.yahoo.com
**API Wrapper**: [yfinance](https://pypi.org/project/yfinance/) (Apache License)

#### Terms & Licensing
- **Terms**: Personal use only
- **Important**: "Yahoo! finance API is intended for personal use only"
- **Disclaimer**: yfinance is NOT affiliated with Yahoo, Inc.
- **References**:
  - [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
  - [API Terms](https://policies.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.htm)

#### What We Implement
| Feature | Implemented | Notes |
|---------|-------------|-------|
| OHLCV (daily) | Yes | Primary use case |
| OHLCV (intraday) | Yes | 1m, 5m, 15m, 30m, 1h |
| Adjusted prices | Yes | Split & dividend adjusted |
| Batch downloads | Yes | fetch_batch_ohlcv() |

#### What Provider Offers (NOT Implemented)
| Feature | Available | Priority |
|---------|-----------|----------|
| Options chains | Yes | HIGH |
| Financial statements | Yes | HIGH |
| Earnings data | Yes | MEDIUM |
| Analyst recommendations | Yes | LOW |
| Holders data | Yes | LOW |
| ESG scores | Yes | LOW |
| News headlines | Yes | LOW |

#### Rate Limits
- ~2,000 requests/hour (informal, not documented)
- No API key required
- IP-based throttling

---

### 2. Databento (DataBentoProvider)

**Website**: https://databento.com
**Pricing Page**: https://databento.com/pricing

#### Pricing Tiers
| Tier | Price | Features |
|------|-------|----------|
| Free Trial | $125 credit | Historical data only |
| Usage-based | Pay as you go | Historical data, $/GB |
| Standard | $179/mo | Live data, 15+ years core, 1yr L1 |
| Plus | $1,500/mo + fees | External distribution, 15yr L1 |
| Unlimited | $4,000/mo + fees | 15+ years all schemas |

#### What We Implement
| Feature | Implemented | Notes |
|---------|-------------|-------|
| OHLCV schemas | Yes | ohlcv-1m, ohlcv-1h, ohlcv-1d |
| Continuous futures | Yes | fetch_continuous_futures() |
| Multiple schemas | Yes | fetch_multiple_schemas() |
| Batch API | Yes | Historical downloads |

#### What Provider Offers (NOT Implemented)
| Feature | Available | Priority |
|---------|-----------|----------|
| MBO (Market by Order) | Yes | LOW (out of scope) |
| MBP-10 (Market depth) | Yes | LOW |
| TBBO (Top of book) | Yes | MEDIUM |
| Trades (tick) | Yes | MEDIUM |
| WebSocket streaming | Yes | NOT SUPPORTING |
| OPRA options | Yes | HIGH |
| Symbology API | Yes | LOW |

#### Coverage
- 45+ exchanges
- 650,000+ symbols
- 15+ years history
- CME, CBOT, NYMEX, COMEX, Eurex, ICE, EEX

---

### 3. Polygon / Massive (PolygonProvider)

**Website**: https://massive.com (formerly polygon.io)
**Pricing Page**: https://massive.com/pricing

**Note**: Polygon.io rebranded to Massive.com in 2025.

#### Pricing Tiers (Stocks)
| Tier | Price | Features |
|------|-------|----------|
| Basic (Free) | $0/mo | 5 calls/min, 2yr history, EOD |
| Starter | $29/mo | Unlimited calls, 5yr history, 15min delayed |
| Developer | $79/mo | 10yr history, trades data |
| Advanced | $199/mo | 20yr+ history, real-time, quotes, financials |

#### What We Implement
| Feature | Implemented | Notes |
|---------|-------------|-------|
| OHLCV (daily) | Yes | Aggregates endpoint |
| OHLCV (minute) | Yes | Requires paid tier |

#### What Provider Offers (NOT Implemented)
| Feature | Available | Priority |
|---------|-----------|----------|
| Options chains | Yes | HIGH |
| Options Greeks | Yes | HIGH |
| Financials & ratios | Yes | HIGH |
| Trades (tick) | Yes | MEDIUM |
| Quotes (NBBO) | Yes | MEDIUM |
| WebSockets | Yes | NOT SUPPORTING |
| Crypto | Yes | LOW (use Binance) |
| Forex | Yes | LOW (use Oanda) |
| Indices | Yes | MEDIUM |
| Futures | Yes | LOW (use Databento) |

---

### 4. EODHD (EODHDProvider)

**Website**: https://eodhd.com
**Pricing Page**: https://eodhd.com/pricing

#### Pricing Tiers
| Tier | Price | Features |
|------|-------|----------|
| Free | $0/mo | 20 calls/day, 1yr history |
| EOD All World | $19.99/mo | 100k calls/day, 30yr+ history |
| EOD+Intraday | $29.99/mo | + Intraday data |
| Fundamentals | $59.99/mo | + Stock fundamentals |
| ALL-IN-ONE | $99.99/mo | Everything included |

**Note**: 50% student discount available.

#### What We Implement
| Feature | Implemented | Notes |
|---------|-------------|-------|
| OHLCV (daily) | Yes | 60+ exchanges |
| Adjusted prices | Yes | Split & dividend adjusted |
| Delisted data | Yes | Survivorship-bias free |

#### What Provider Offers (NOT Implemented)
| Feature | Available | Priority |
|---------|-----------|----------|
| Intraday data | Yes (paid) | MEDIUM |
| Fundamentals | Yes (paid) | HIGH |
| Earnings data | Yes | MEDIUM |
| Insider transactions | Yes | LOW |
| ETF fundamentals | Yes | MEDIUM |
| Options (marketplace) | Yes ($29.99) | HIGH |
| Tick data (marketplace) | Yes ($9.99) | LOW |
| Real-time WebSocket | Yes | NOT SUPPORTING |

#### Coverage
- 60+ exchanges worldwide
- 150,000+ tickers
- 30+ years history
- Stocks, ETFs, Mutual Funds, Forex, Crypto

---

### 5. Tiingo (TiingoProvider)

**Website**: https://tiingo.com
**Pricing Page**: https://tiingo.com/about/pricing

#### Pricing
- Free tier: 1,000 requests/day, 500 symbols/month
- Paid tiers available for higher limits

#### What We Implement
| Feature | Implemented | Notes |
|---------|-------------|-------|
| OHLCV (daily) | Yes | US equities |
| Adjusted prices | Yes | |

---

### 6. Finnhub (FinnhubProvider)

**Website**: https://finnhub.io
**Pricing Page**: https://finnhub.io/pricing

#### Pricing
- Free tier: 60 requests/min
- Paid tiers: $49+/mo for premium features

#### What We Implement
| Feature | Implemented | Notes |
|---------|-------------|-------|
| OHLCV | Yes | Multi-asset |
| Real-time quotes | Yes | Free tier |

#### What Provider Offers (NOT Implemented)
| Feature | Available | Priority |
|---------|-----------|----------|
| Company metrics | Yes | HIGH |
| Analyst estimates | Yes | MEDIUM |
| Earnings calendar | Yes | MEDIUM |
| ESG scores | Yes | LOW |
| News sentiment | Yes | LOW |

---

### 7. FRED (FREDProvider)

**Website**: https://fred.stlouisfed.org
**API Docs**: https://fred.stlouisfed.org/docs/api/fred/

#### Pricing
- Free with API key
- 120 requests/minute

#### What We Implement
| Feature | Implemented | Notes |
|---------|-------------|-------|
| Economic series | Yes | 800,000+ series |
| OHLCV mapping | Yes | For compatible series |

---

### 8. Academic Factor Providers

#### AQR (AQRFactorProvider)
**Website**: https://www.aqr.com/Insights/Datasets
**Pricing**: Free (academic use)

| Dataset | Implemented |
|---------|-------------|
| Quality Minus Junk (QMJ) | Yes |
| Betting Against Beta (BAB) | Yes |
| Value Momentum Everywhere (VME) | Yes |
| Time Series Momentum (TSMOM) | Yes |
| Century of Factor Premia | Yes |
| 16 total datasets | Yes |

#### Fama-French (FamaFrenchProvider)
**Website**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
**Pricing**: Free (academic use)

| Dataset | Implemented |
|---------|-------------|
| 3-Factor Model | Yes |
| 5-Factor Model | Yes |
| Momentum Factor | Yes |
| Industry Portfolios | Yes |
| 50+ total datasets | Yes |

---

### 9. Crypto Providers

#### CoinGecko (CoinGeckoProvider)
**Website**: https://www.coingecko.com
**Pricing**: https://www.coingecko.com/en/api/pricing

- Free: 50 calls/min
- Demo API key available
- Pro plans: $129+/mo

#### Binance (BinanceProvider / BinancePublicProvider)
**Website**: https://www.binance.com
**Data Portal**: https://data.binance.vision

- **BinanceProvider**: Live API (may have geo-restrictions in US)
- **BinancePublicProvider**: Bulk historical downloads (no restrictions)

#### CryptoCompare (CryptoCompareProvider)
**Website**: https://min-api.cryptocompare.com
**Pricing**: https://min-api.cryptocompare.com/pricing

- Free: 250,000 calls/month
- Paid: $19.99+/mo

---

### 10. Prediction Markets

#### Kalshi (KalshiProvider)
**Website**: https://kalshi.com
**Developer**: https://kalshi.com/developer

- US-regulated prediction market
- Free API access
- Event contracts only

#### Polymarket (PolymarketProvider)
**Website**: https://polymarket.com
**Docs**: https://docs.polymarket.com

- Crypto-based prediction market
- Free API access
- CLOB (Central Limit Order Book)

---

### 11. Special Purpose Providers

#### WikiPrices (WikiPricesProvider)
- Historical US equities (1962-2018)
- 3,199 stocks, 15.4M rows
- Survivorship-bias free
- Local Parquet file required

#### NASDAQ ITCH (ITCHSampleProvider)
- NASDAQ TotalView-ITCH sample data
- Tick-level order book
- Free sample files from NASDAQ

#### Synthetic (SyntheticProvider)
- Generates synthetic OHLCV data
- For testing and demos
- No network required

---

## Gap Analysis Summary

### HIGH Priority Gaps (Should Implement)

| Gap | Providers | Effort |
|-----|-----------|--------|
| **Options chains** | Polygon, Yahoo, Databento, EODHD | High |
| **Financial statements** | Yahoo, Finnhub, Polygon, EODHD | Medium |
| **Company metrics** | Finnhub, Polygon | Medium |

### MEDIUM Priority Gaps

| Gap | Providers | Effort |
|-----|-----------|--------|
| Earnings data | Yahoo, EODHD, Finnhub | Medium |
| Analyst estimates | Finnhub | Low |
| ETF fundamentals | EODHD | Low |
| Trades (tick) | Databento, Polygon | High |

### Explicitly NOT Supporting

| Feature | Reason |
|---------|--------|
| WebSocket streaming | Out of scope for batch data library |
| Real-time order book | Use native provider APIs directly |
| News/sentiment | Focus on price data |

---

## Provider Selection Guide

### By Use Case

| Use Case | Recommended Provider |
|----------|---------------------|
| Quick start (free) | YahooFinance |
| US equities (production) | EODHD or Polygon/Massive |
| Global equities | EODHD |
| Futures/options | Databento |
| Crypto | BinancePublic |
| Forex | Oanda |
| Academic factors | Fama-French + AQR |
| Economic indicators | FRED |
| Historical backtesting | WikiPrices |
| Prediction markets | Kalshi (US) or Polymarket |

### By Budget

| Budget | Providers |
|--------|-----------|
| $0 | Yahoo, CoinGecko, BinancePublic, WikiPrices, FRED, AQR, Fama-French |
| <$30/mo | EODHD ($19.99), Polygon Starter ($29) |
| <$100/mo | EODHD All-in-One ($99.99), Polygon Developer ($79) |
| Professional | Databento ($179+), Polygon Advanced ($199) |

---

## Update Log

- 2025-12-27: Initial audit with Chrome DevTools research
- Polygon.io rebranded to Massive.com
- Verified all pricing pages and terms
