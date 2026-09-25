# Prediction-Market Data Sources

Prediction-market prices are contract-specific. Preserve the complete question, resolution
source, close and settlement rules, outcome token, market status, and fee model. A quoted price is
an implied market probability only after accounting for spread, liquidity, and contract terms.

!!! note "Verified September 2026"
    This category has enough distinct sources and selection criteria for a standalone reference.
    Entries were checked against current official API documentation.

| Source and product | Coverage or history | Acquisition | Access | Material research caveat | ml4t-data provider |
|--------------------|---------------------|-------------|--------|--------------------------|--------------------|
| [Kalshi API](https://docs.kalshi.com/getting_started/quick_start_market_data) | Regulated event contracts across economics, politics, climate, technology, and other categories | REST and WebSocket APIs | Public market data; credentials for authenticated endpoints | Market tickers encode series and contract terms; use settled outcomes and rule changes, not titles alone | `KalshiProvider` |
| [Polymarket APIs](https://docs.polymarket.com/) | Binary and multi-outcome markets represented by outcome tokens on Polygon | Gamma and CLOB REST APIs plus WebSocket market data | Public market data; trading has additional requirements | Slugs, condition IDs, and outcome token IDs are different identifiers; YES and NO prices include spread | `PolymarketProvider` |
| [Manifold API and dumps](https://docs.manifold.markets/) | User-created social prediction markets across many question types | REST API and downloadable data dumps | Public data under documented licensing | Play-money markets, user-defined resolution, and automated-market-maker mechanics are not directly comparable with regulated cash markets | No |

`KalshiProvider` lists series and markets and converts native candlesticks to the package OHLCV
schema. `PolymarketProvider` resolves slugs or condition IDs to outcome tokens and aggregates price
history. Public data access does not imply that trading is available in every jurisdiction.

## Selection Notes

- Use Kalshi when regulated contract definitions and exchange settlement are required.
- Use Polymarket for crypto-settled CLOB markets and on-chain outcome tokens.
- Use Manifold for social forecasting research where play-money incentives are acceptable.
