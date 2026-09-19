# Migrate former QLDM names

The package preserves two former public names during the 0.x release series. Their use emits
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
