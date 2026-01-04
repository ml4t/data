# Provider Selection Guide

Choose the right data provider for your needs with this decision flowchart.

## Quick Decision Flowchart

```mermaid
flowchart TD
    Start[What data do you need?] --> AssetType{Asset Type?}

    AssetType -->|Crypto| CryptoChoice{API key OK?}
    AssetType -->|US Stocks| USChoice{Budget?}
    AssetType -->|Global Stocks| GlobalChoice{Budget?}
    AssetType -->|Forex| ForexChoice{Professional?}
    AssetType -->|Futures/Options| FuturesChoice
    AssetType -->|Multiple Assets| MultiChoice{Which combination?}

    %% Crypto Path
    CryptoChoice -->|No API key| CoinGecko[CoinGecko<br/>FREE unlimited]
    CryptoChoice -->|Want real-time| CryptoCompare[CryptoCompare<br/>FREE/Paid]

    %% US Stocks Path
    USChoice -->|FREE only| USFree{Call volume?}
    USChoice -->|Can pay| USPaid{Budget?}

    USFree -->|Low 25/day| AlphaVantage[Alpha Vantage<br/>FREE 25/day]
    USFree -->|Medium 1000/day| Tiingo[Tiingo<br/>FREE 1000/day]
    USFree -->|High 50K/mo| IEXCloud[IEX Cloud<br/>FREE 50K msg/mo]

    USPaid -->|Budget €20/mo| EODHD[EODHD<br/>€19.99/mo]
    USPaid -->|Professional $60+| Finnhub[Finnhub<br/>$59.99+/mo]

    %% Global Stocks Path
    GlobalChoice -->|FREE testing| EODHDFree[EODHD<br/>FREE 500/day]
    GlobalChoice -->|Budget €20/mo| EODHDPaid[EODHD<br/>€19.99/mo<br/>60+ exchanges]
    GlobalChoice -->|Professional| FinnhubGlobal[Finnhub<br/>$59.99+/mo<br/>70+ exchanges]

    %% Forex Path
    ForexChoice -->|Research only| AlphaVantageForex[Alpha Vantage<br/>FREE 25/day]
    ForexChoice -->|Trading/Production| OANDA[OANDA<br/>Professional]
    ForexChoice -->|Multi-asset| TwelveDataForex[Twelve Data<br/>800/day]

    %% Futures Path
    FuturesChoice -->|Institutional| Databento[Databento<br/>Paid only]

    %% Multi-Asset Path
    MultiChoice -->|Stocks+Forex+Crypto| TwelveData[Twelve Data<br/>FREE 800/day]
    MultiChoice -->|US Stocks+Global| EODHDMulti[EODHD<br/>€19.99/mo]
    MultiChoice -->|Professional all| PolygonMulti[Polygon<br/>Paid]

    %% Style definitions
    classDef freeProvider fill:#90EE90,stroke:#228B22,stroke-width:2px
    classDef paidProvider fill:#FFB6C1,stroke:#DC143C,stroke-width:2px
    classDef decision fill:#87CEEB,stroke:#4682B4,stroke-width:2px

    class CoinGecko,AlphaVantage,Tiingo,IEXCloud,EODHDFree,AlphaVantageForex,TwelveData,CryptoCompare freeProvider
    class EODHD,Finnhub,EODHDPaid,FinnhubGlobal,OANDA,Databento,TwelveDataForex,EODHDMulti,PolygonMulti paidProvider
    class AssetType,CryptoChoice,USChoice,GlobalChoice,ForexChoice,USFree,USPaid,GlobalChoice,ForexChoice,MultiChoice decision
```

## Provider Comparison Matrix

### By Asset Class

| Provider | Crypto | US Stocks | Global Stocks | Forex | Futures | API Key | Free Tier | Best For |
|----------|--------|-----------|---------------|-------|---------|---------|-----------|----------|
| **CoinGecko** | ✅ | ❌ | ❌ | ❌ | ❌ | No | Unlimited | Crypto historical |
| **CryptoCompare** | ✅ | ❌ | ❌ | ❌ | ❌ | Optional | Good | Crypto real-time |
| **Tiingo** | ✅ | ✅ | ❌ | ❌ | ❌ | Yes | 1000/day | High-quality stocks |
| **IEX Cloud** | ❌ | ✅ | ❌ | ❌ | ❌ | Yes | 50K/mo | US equities + fundamentals |
| **Alpha Vantage** | ✅ | ✅ | ✅ | ✅ | ❌ | Yes | 25/day | Multi-asset research |
| **EODHD** | ❌ | ✅ | ✅ | ❌ | ❌ | Yes | 500/day | Global stocks, best value |
| **Finnhub** | ✅ | ✅ | ✅ | ✅ | ❌ | Yes | Paid only | Professional grade |
| **Twelve Data** | ✅ | ✅ | ❌ | ✅ | ❌ | Yes | 800/day | Multi-asset + indicators |
| **OANDA** | ❌ | ❌ | ❌ | ✅ | ❌ | Yes | No | Professional forex |
| **Databento** | ❌ | ✅ | ❌ | ❌ | ✅ | Yes | Paid only | Institutional derivatives |
| **Polygon** | ✅ | ✅ | ❌ | ✅ | ❌ | Yes | Paid only | Professional multi-asset |

### By Pricing

#### Free Tier (No Credit Card Required)

| Provider | Daily Limit | Monthly Limit | Best Use Case |
|----------|-------------|---------------|---------------|
| **CoinGecko** | 10-50 calls/min | Unlimited | Crypto research |
| **IEX Cloud** | N/A | 50K messages | US stock research |
| **Tiingo** | 1000 calls | 500 symbols | Daily stock updates |
| **Alpha Vantage** | 25 calls | 750 calls | Conservative research |
| **EODHD** | 500 calls | ~15K calls | Global stock testing |
| **Twelve Data** | 800 calls | ~24K calls | Multi-asset research |
| **CryptoCompare** | Varies | Varies | Crypto real-time |

#### Paid Tiers (Affordable)

| Provider | Price | What You Get |
|----------|-------|--------------|
| **EODHD** | €19.99/mo | 150K+ global tickers, unlimited calls |
| **Twelve Data** | $9.99/mo | 800 calls/min, multi-asset |
| **Tiingo** | $30/mo | 20K calls/hour, news, fundamentals |
| **Alpha Vantage** | $49.99/mo | 75 calls/min, all features |

#### Professional Tiers

| Provider | Starting Price | What You Get |
|----------|----------------|--------------|
| **Finnhub** | $59.99/mo | Historical OHLCV, 70+ exchanges |
| **Polygon** | $99/mo | Multi-asset, professional-grade |
| **Databento** | Custom | Institutional derivatives data |

## Decision Guidelines

### For Beginners
1. **Start with CoinGecko** (crypto) - No API key, unlimited free tier
2. **Try Tiingo** (stocks) - Generous free tier (1000/day), great quality
3. **Experiment freely** - All providers have free tiers or trials

### For Researchers
1. **Tiingo** - High-quality stock data with generous limits
2. **Alpha Vantage** - Multi-asset but limited (25/day)
3. **EODHD** - Global coverage, excellent value (€19.99/mo)

### For Traders
1. **EODHD** - Best value for global stocks (€19.99/mo)
2. **OANDA** - Professional forex data
3. **Finnhub** - Real-time + historical, professional grade

### For Institutions
1. **Databento** - Tick-level derivatives data
2. **Polygon** - Multi-asset professional data
3. **Finnhub** - Global exchange coverage

## Quick Recommendations

### "I want crypto data"
→ **CoinGecko** (no API key, unlimited)

### "I want US stock data for free"
→ **Tiingo** (1000/day) or **IEX Cloud** (50K msg/month)

### "I want global stock data"
→ **EODHD** (€19.99/mo for 60+ exchanges)

### "I want stocks + forex + crypto"
→ **Twelve Data** (800/day free) or **Alpha Vantage** (25/day free)

### "I'm building a trading system"
→ **Finnhub** ($59.99+/mo) or **EODHD** (€19.99/mo)

### "I need professional derivatives data"
→ **Databento** (paid only)

## Rate Limit Considerations

### Conservative (Good for testing)
- **Alpha Vantage**: 25/day
- **EODHD Free**: 500/day
- **Twelve Data Free**: 800/day

### Moderate (Good for research)
- **Tiingo**: 1000/day
- **IEX Cloud**: 50K messages/month
- **CoinGecko**: 10-50/min

### High Volume (Good for production)
- **EODHD Paid**: Unlimited
- **Finnhub**: 60/min (free), higher (paid)
- **Tiingo Paid**: 20K/hour

## Next Steps

1. **Read the full provider docs** in the [main README](../README.md)
2. **Check provider status** in your target asset class
3. **Get API keys** from provider websites
4. **Test with free tiers** before committing to paid plans
5. **Use incremental updates** to minimize API calls

## Getting Help

- **Documentation**: See [docs/user-guide/](user-guide/)
- **Examples**: Check [examples/](../examples/) directory
- **Issues**: Report problems on GitHub
- **Community**: Join discussions for provider recommendations
