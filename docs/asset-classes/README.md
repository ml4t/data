# Asset Class Guides

**Complete guides** for fetching market data across crypto, equities, forex, and futures markets using ML4T Data.

---

## Overview

These guides help you choose the right data provider and implement best practices for each asset class. Each guide includes:

- ✅ **Provider recommendations** by use case
- ✅ **Quick start examples** with working code
- ✅ **Feature comparisons** (free tier, rate limits, coverage)
- ✅ **Complete API references** for each provider
- ✅ **Best practices** specific to the asset class
- ✅ **Common gotchas** and how to avoid them

---

## Available Guides

### [📈 Cryptocurrency Data Guide](./crypto.md)

**Asset Class**: Bitcoin, Ethereum, Altcoins, Stablecoins
**Providers**: CoinGecko, CryptoCompare
**Difficulty**: 🟢 Beginner-friendly
**Free Tier**: ✅ Generous (50 calls/min to 100K/month)

**Best for**:
- Crypto trading strategies
- Portfolio analysis
- Academic research on digital assets
- 24/7 market analysis

**Quick start**:
```python
from ml4t.data.providers import CoinGeckoProvider

# No API key needed!
provider = CoinGeckoProvider()
df = provider.fetch_ohlcv("BTC", start="2024-01-01", end="2024-12-31")
```

**[Read the full guide →](./crypto.md)**

---

### [📊 Equities Data Guide](./equities.md)

**Asset Classes**: US Stocks, Global Stocks, ETFs, Indices
**Providers**: Tiingo, IEX Cloud, Alpha Vantage, EODHD, Finnhub, Twelve Data, Polygon
**Difficulty**: 🟡 Moderate
**Free Tier**: ⚠️ Varies (25/day to 1000/day depending on provider)

**Best for**:
- US stock backtesting
- Global equity strategies
- Portfolio optimization
- Factor research

**Quick start** (US stocks):
```python
from ml4t.data.providers import TiingoProvider

provider = TiingoProvider(api_key="your_key")
df = provider.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-12-31")
```

**Quick start** (Global stocks):
```python
from ml4t.data.providers import EODHDProvider

provider = EODHDProvider(api_key="your_key", exchange="US")
df_us = provider.fetch_ohlcv("AAPL.US", start="2024-01-01", end="2024-12-31")
df_uk = provider.fetch_ohlcv("VOD.LSE", start="2024-01-01", end="2024-12-31")
```

**[Read the full guide →](./equities.md)**

---

### [💱 Forex Data Guide](./forex.md)

**Asset Class**: Currency Pairs (FX)
**Providers**: OANDA (primary), Twelve Data, Alpha Vantage, Polygon
**Difficulty**: 🟡 Moderate
**Free Tier**: ⚠️ OANDA requires demo account (free), others have limits

**Best for**:
- FX trading strategies
- Currency risk analysis
- Global macro research
- High-frequency forex strategies (OANDA offers 5-second to monthly bars)

**Quick start**:
```python
from ml4t.data.providers import OandaProvider

provider = OandaProvider(api_key="your_key", account_type="practice")
df = provider.fetch_ohlcv("EUR_USD", start="2024-01-01", end="2024-12-31", frequency="H1")
```

**[Read the full guide →](./forex.md)**

---

### [Futures Data Guide](./futures.md)

**Asset Classes**: Futures Contracts
**Providers**: Databento (primary), Massive for listed options
**Difficulty**: 🔴 Advanced
**Free Tier**: ❌ None (Databento requires paid subscription ~$30-50/month)

**Best for**:
- Professional futures trading
- Derivatives research
- High-frequency strategies
- Institutional backtesting

**Quick start**:
```python
from ml4t.data.providers import DataBentoProvider

provider = DataBentoProvider(api_key="your_key", dataset="GLBX.MDP3")
df = provider.fetch_ohlcv("ES.v.0", start="2024-01-01", end="2024-12-31")  # S&P 500 futures
```

**[Read the full guide →](./futures.md)**

---

## Choosing the Right Asset Class

### By Use Case

| Use Case | Asset Class | Recommended Provider(s) |
|----------|-------------|-------------------------|
| **Crypto trading** | Cryptocurrency | CoinGecko (free), CryptoCompare |
| **US stock backtesting** | Equities | Tiingo (1000/day free) |
| **Global equities** | Equities | EODHD (€20/month best value) |
| **Forex strategies** | Forex | OANDA (professional-grade) |
| **Futures trading** | Futures | Databento (institutional quality) |
| **Multi-asset portfolio** | Mixed | Twelve Data (stocks+forex+crypto) |
| **Research (low volume)** | Any | Alpha Vantage (25/day free) |

### By Budget

**$0/month (Free Tier)**:
- ✅ **Crypto**: CoinGecko (50 calls/min, no API key)
- ✅ **US Stocks**: Tiingo (1000/day), IEX Cloud (50K messages/month)
- ✅ **Forex**: OANDA demo account
- ❌ **Futures**: Not available free

**< $50/month**:
- ✅ **Global Stocks**: EODHD (€19.99/month unlimited)
- ✅ **Futures**: Databento ($30-50/month starter)
- ✅ **Multi-asset**: Twelve Data ($49/month)

**$100+/month (Professional)**:
- ✅ **Global Stocks**: Finnhub ($59.99/month)
- ✅ **Multi-asset**: Polygon ($199-399/month)
- ✅ **Futures**: Databento ($50-100/month professional)

### By Market Hours

**24/7 Trading**:
- Cryptocurrency → [Crypto Guide](./crypto.md)

**24/5 Trading** (weekdays only):
- Forex → [Forex Guide](./forex.md)
- Futures → [Futures Guide](./futures.md)

**Market Hours Only** (9:30 AM - 4:00 PM ET typically):
- Equities → [Equities Guide](./equities.md)

---

## Quick Provider Comparison

### All Providers at a Glance

| Provider | Crypto | Stocks | Forex | Futures | Free Tier | API Key | Best For |
|----------|--------|--------|-------|---------|-----------|---------|----------|
| **CoinGecko** | ✅ | ❌ | ❌ | ❌ | 50/min | Optional | Crypto (beginners) |
| **CryptoCompare** | ✅ | ❌ | ❌ | ❌ | 100K/month | Optional | Crypto (advanced) |
| **Tiingo** | ❌ | ✅ US | ❌ | ❌ | 1000/day | Required | US stocks |
| **IEX Cloud** | ❌ | ✅ US | ❌ | ❌ | 50K msgs/mo | Required | US stocks + fundamentals |
| **Alpha Vantage** | ⚠️ | ✅ | ⚠️ | ❌ | 25/day | Required | Low-volume research |
| **EODHD** | ❌ | ✅ Global | ❌ | ❌ | 500/day | Required | Global stocks (best value) |
| **Finnhub** | ⚠️ | ✅ Global | ⚠️ | ❌ | Real-time only | Required | Professional stocks |
| **OANDA** | ❌ | ❌ | ✅ | ❌ | Demo account | Required | Professional forex |
| **Twelve Data** | ✅ | ✅ | ✅ | ❌ | 800/day | Required | Multi-asset |
| **Polygon** | ✅ | ✅ | ✅ | ✅ | 5/min | Required | Multi-asset (paid) |
| **Databento** | ❌ | ⚠️ | ❌ | ⚠️ native SDK | None | Required | Institutional futures |

**Legend**:
- ✅ Full support
- ⚠️ Limited support
- ❌ Not supported

---

## Common Patterns Across Asset Classes

### 1. Incremental Updates

**All asset classes** benefit from incremental updates to reduce API calls:

```python
from ml4t.data.provider_updater import ProviderUpdater

# Works with any provider
updater = ProviderUpdater(provider="tiingo")  # or coingecko, oanda, etc.

# First run: Fetch all history
df = updater.update("AAPL", frequency="daily", lookback_days=365)

# Subsequent runs: Only fetch new data
df = updater.update("AAPL", frequency="daily", lookback_days=7)
# 100-1000x fewer API calls!
```

**See**: [Tutorial 03: Incremental Updates](../tutorials/03_incremental_updates.md)

### 2. Data Quality Validation

**Always validate** OHLCV data after fetching:

```python
from ml4t.data.validation import validate_ohlcv

df = provider.fetch_ohlcv(symbol="BTC", ...)

# Validate invariants
validate_ohlcv(df)  # Raises exception if data is corrupt

# Checks:
# - high >= low ✓
# - high >= open, close ✓
# - low <= open, close ✓
# - volume >= 0 ✓
```

**See**: [Tutorial 04: Data Quality](../tutorials/04_data_quality.md)

### 3. Multi-Provider Fallback

**Never depend on a single provider**:

```python
def fetch_data_robust(symbol: str, asset_class: str):
    """Fetch data with fallback providers."""

    if asset_class == "crypto":
        providers = [CoinGeckoProvider, CryptoCompareProvider]
    elif asset_class == "stocks":
        providers = [TiingoProvider, IEXCloudProvider, AlphaVantageProvider]
    elif asset_class == "forex":
        providers = [OandaProvider, TwelveDataProvider]

    for ProviderClass in providers:
        try:
            provider = ProviderClass()
            return provider.fetch_ohlcv(symbol, ...)
        except Exception as e:
            print(f"{ProviderClass.__name__} failed: {e}")

    raise RuntimeError("All providers failed")
```

**See**: [Tutorial 05: Multi-Provider Strategies](../tutorials/05_multi_provider.md)

### 4. Rate Limiting

**Respect provider rate limits** to avoid bans:

```python
# ML4T Data automatically rate-limits, but be aware of limits

# CoinGecko: 50 calls/min
# Tiingo: 1000 calls/day
# Alpha Vantage: 5 calls/min, 25 calls/day
# OANDA: 120 calls/second (very generous)

# Use incremental updates to minimize calls
# Batch requests when possible
# Implement exponential backoff on failures
```

**See**: [Tutorial 02: Rate Limiting](../tutorials/02_rate_limiting.md)

---

## Asset-Specific Considerations

### Cryptocurrency

**Key differences**:
- ✅ 24/7 markets (no weekends/holidays)
- ✅ No adjusted close needed (no corporate actions)
- ⚠️ High volatility (5-10%+ daily moves common)
- ⚠️ Liquidity varies by exchange

**[Full details in Crypto Guide →](./crypto.md)**

### Equities

**Key differences**:
- ⚠️ Market hours only (9:30 AM - 4:00 PM ET typically)
- ⚠️ Weekends and holidays (no data)
- ✅ Need adjusted close (splits, dividends)
- ⚠️ Symbol format varies by provider (AAPL vs AAPL.US)

**[Full details in Equities Guide →](./equities.md)**

### Forex

**Key differences**:
- ✅ 24/5 markets (Sunday 5 PM - Friday 5 PM ET)
- ⚠️ Session-based volatility (London/NY overlap highest)
- ⚠️ Spreads matter (0.1-2 pips transaction cost)
- ⚠️ Rollover interest (overnight positions)

**[Full details in Forex Guide →](./forex.md)**

### Futures

**Key differences**:
- ⚠️ Contract expiration (must roll monthly/quarterly)
- ⚠️ Leverage (20:1 to 100:1 typical)
- ⚠️ Margin calls (marked-to-market daily)
- ⚠️ Tick sizes (minimum price increments)
- ❌ Most expensive data ($30-50+/month minimum)

**[Full details in Futures Guide →](./futures.md)**

---

## Next Steps

1. **Read the guide** for your target asset class
2. **Choose a provider** based on budget and requirements
3. **Get API keys** (where required)
4. **Try the Quick Start** examples
5. **Implement best practices** from the tutorials
6. **Build multi-provider fallback** for production systems

---

## Related Documentation

**Core Tutorials**:
- [Tutorial 01: Understanding OHLCV Data](../tutorials/01_understanding_ohlcv.md)
- [Tutorial 02: Rate Limiting Best Practices](../tutorials/02_rate_limiting.md)
- [Tutorial 03: Incremental Updates](../tutorials/03_incremental_updates.md)
- [Tutorial 04: Data Quality Validation](../tutorials/04_data_quality.md)
- [Tutorial 05: Multi-Provider Strategies](../tutorials/05_multi_provider.md)

**Provider Setup**:
- [Provider Selection Guide](../provider-selection-guide.md)
- [Creating a Custom Provider](../creating_a_provider.md)
- [Extending ML4T Data](../extending_ml4t-data.md)

**Getting Started**:
- [Main README](../../README.md)
- [Quick Start Guide](../../README.md#quick-start)
- [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

**Questions or issues?** Open a GitHub issue or see [CONTRIBUTING.md](../../CONTRIBUTING.md) for help.
