# FXMacroData

`FXMacroDataProvider` retrieves FX-oriented macroeconomic and reference-rate data. It is a
specialized provider rather than an OHLCV adapter, so use its macro methods directly instead of
passing it to `DataManager.fetch()`.

## Access

Public USD endpoints work without a credential. Set `FXMACRODATA_API_KEY` or `FXMD_API_KEY` when
the selected endpoint or account requires a key.

```python
from ml4t.data.providers import FXMacroDataProvider

provider = FXMacroDataProvider()
```

The provider does not turn macro or reference series into executable currency quotes. For spot
and intraday FX products, compare the [foreign exchange data sources](fx.md). For economic
release and revision requirements, see [macroeconomic data sources](macro.md).
