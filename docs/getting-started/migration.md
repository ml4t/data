# Migrate compatibility names

## Names preserved through the 0.x series

The package preserves two former QLDM names during the 0.x release series. Their use emits
`DeprecationWarning`. They will not be removed before version 1.0.

## Data root

Replace:

```bash
export QLDM_DATA_ROOT=/srv/market-data
```

with:

```bash
export ML4T_DATA_PATH=/srv/market-data
```

## Base exception

Replace:

```python
from ml4t.data.core.exceptions import QldmError
```

with:

```python
from ml4t.data.core.exceptions import ML4TDataError
```

The former exception name resolves to the same class during the migration period. Code that treats
deprecation warnings as errors should migrate before upgrading.

## Changes for version 0.2

Version 0.2 removes three compatibility interfaces that emitted `DeprecationWarning` in the 0.1
series. Update callers and saved configuration before upgrading.

### Massive provider name

Replace `PolygonProvider` imports and the `polygon` registry name with `MassiveProvider` and
`massive`:

```python
from ml4t.data.providers import MassiveProvider

provider = MassiveProvider()
```

The service account can still use `POLYGON_API_KEY`; new environments should set
`MASSIVE_API_KEY`.

### Negative-price validation

Replace `check_negative_prices=True` with `negative_price_policy="forbid"`, and replace
`check_negative_prices=False` with `negative_price_policy="allow"`. Apply the same substitution in
saved `ValidationRuleConfig` YAML.

Use `negative_price_policy="warn"` when negative prices should produce validation warnings without
failing validation.

### Synthetic GARCH configuration

Remove the `garch_omega` argument from `SyntheticProvider`. The provider derives the per-period
constant from `annual_volatility`, the requested frequency, `garch_alpha`, and `garch_beta` so the
configured unconditional variance is preserved.
