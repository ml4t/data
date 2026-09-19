# Choose a provider

Choose a provider from the data type, authentication, licensing, and update frequency required by
the workflow. Provider pricing, quotas, and available history change independently of this library;
confirm them with the provider before scheduling recurring work.

## Start without an external service

Use `SyntheticProvider` for installation checks, tests, and examples that must run offline:

```python
from ml4t.data.providers import SyntheticProvider

data = SyntheticProvider(seed=42).fetch_ohlcv(
    "SYNTH", "2024-01-01", "2024-01-10", "daily"
)
```

## Select by responsibility

| Requirement | Providers to evaluate | External boundary |
| --- | --- | --- |
| Public cryptocurrency market data | CoinGecko, Binance, Binance Public, OKX | Network access and provider usage terms |
| Equity bars | Yahoo Finance, Alpaca, EODHD, Tiingo, Twelve Data, Massive, Finnhub | Some adapters require credentials, accounts, or paid history |
| Foreign exchange | OANDA, Twelve Data, FXMacroData | OANDA and Twelve Data require credentials |
| Futures and options | Databento, Binance, OKX | Historical or licensed data may be metered |
| Macroeconomic series | FRED, FXMacroData | FRED requires an API key for normal use |
| Research factors | Fama-French, AQR | Network access and source-specific terms |
| Prediction markets | Kalshi, Polymarket | Network access and changing public endpoints |
| Frozen equity history | Wiki Prices | Local historical dataset ending in 2018 |
| CFTC positioning | COT | Install the `cot` extra |

CryptoCompare remains available for evaluation but is not release-qualified until its live provider
contract runs successfully on a release commit.

## Check dependencies and credentials

Provider extras are explicit:

```bash
pip install "ml4t-data[yahoo]"
pip install "ml4t-data[databento]"
pip install "ml4t-data[oanda]"
pip install "ml4t-data[cot]"
```

Credentialed adapters read the environment variables named in their provider guides. Do not place
credentials in source code, notebooks, or committed configuration files.

## Confirm capabilities in code

The registry records asset classes, authentication, and supported fetch operations:

```python
from ml4t.data.providers import advertised_provider_specs

for spec in advertised_provider_specs():
    print(spec.name, sorted(spec.capabilities), spec.access_label, spec.extra)
```

A registered provider may support OHLCV, economic series, factors, or another capability-specific
method. Do not assume that every adapter implements every operation.

## Before scheduling updates

1. Read the provider-specific guide in the [provider reference](../providers/index.md).
2. Confirm current account quotas, history, licensing, and redistribution terms with the provider.
3. Run a small request for the exact market, symbol format, frequency, and date range.
4. Set `ML4T_DATA_PATH` to an explicit storage location.
5. Use incremental updates so requests cover only missing ranges.
6. Handle authentication, rate-limit, network, and unavailable-data errors separately.

For a first successful workflow, continue with the [quickstart](quickstart.md).
