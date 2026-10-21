# Fama-French Provider

**Provider**: `FamaFrenchProvider`
**Website**: [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
**API Key**: Not required
**Free Tier**: Free (academic use)

---

## Overview

The Ken French Data Library provides 50+ factor datasets including the famous Fama-French 3-Factor and 5-Factor models.

**Best For**: Academic factors, portfolio research, factor exposure analysis

---

## Quick Start

```python
from ml4t.data.providers import FamaFrenchProvider

provider = FamaFrenchProvider()

# Core factors
ff3 = provider.fetch("ff3", frequency="monthly")   # Mkt-RF, SMB, HML, RF
ff5 = provider.fetch("ff5", frequency="daily")     # +RMW, CMA, RF
mom = provider.fetch("mom", frequency="monthly")   # Momentum

# Combined (Carhart 4-Factor)
carhart = provider.fetch_combined(["ff3", "mom"])

provider.close()
```

---

## Available Datasets

### Core Factors

| Dataset | Factors | Frequency |
|---------|---------|-----------|
| `ff3` | Mkt-RF, SMB, HML, RF | Daily, Monthly |
| `ff5` | Mkt-RF, SMB, HML, RMW, CMA, RF | Daily, Monthly |
| `mom` | MOM (Momentum) | Daily, Monthly |
| `st_rev` | Short-Term Reversal | Monthly |
| `lt_rev` | Long-Term Reversal | Monthly |

### Industry Portfolios

| Dataset | Description |
|---------|-------------|
| `ind_5` | 5 Industries |
| `ind_10` | 10 Industries |
| `ind_12` | 12 Industries |
| `ind_17` | 17 Industries |
| `ind_30` | 30 Industries |
| `ind_48` | 48 Industries (most common) |
| `ind_49` | 49 Industries |

### Size/Value Portfolios

| Dataset | Description |
|---------|-------------|
| `port_size_bm_6` | 6 Size/BM (2x3) |
| `port_size_bm_25` | 25 Size/BM (5x5) |
| `port_size_bm_100` | 100 Size/BM (10x10) |

### International

| Dataset | Region |
|---------|--------|
| `ff3_developed` | Developed Markets |
| `ff3_europe` | Europe |
| `ff3_japan` | Japan |
| `ff3_asia_pacific` | Asia Pacific ex Japan |

---

## Data Format

- Returns in decimal format (0.01 = 1%)
- Daily and monthly frequencies
- History from 1926 (US) or 1990 (international)

---

## Combined Factors

Create multi-factor models:

```python
# Carhart 4-Factor (FF3 + Momentum)
ff3_mom = provider.fetch_combined(["ff3", "mom"])

# 6-Factor (FF5 + Momentum)
ff5_mom = provider.fetch_combined(["ff5", "mom"])
```

---

## Cache policy and source missingness

The existing default remains an unbounded cache read, not a guarantee of source freshness.
Choose a maximum age when freshness matters:

```python
from datetime import timedelta

with FamaFrenchProvider(cache_path="./factor-cache") as provider:
    fresh = provider.fetch("ff3_mom", frequency="daily", refresh=True)
    cached = provider.fetch(
        "ff3_mom", frequency="daily", offline=True, max_age=timedelta(days=1)
    )
```

Offline calls never request the source. Under the declared age policy, stale or absent caches fail;
`allow_stale=True` explicitly accepts stale data and labels it in `last_fetch_metadata`.
`refresh=True` cannot be combined with `offline=True`. Downloads and cache contents are validated
before atomic single-file replacement. A failed download or interrupted write preserves the last
valid cache. Combined datasets retain separate capture metadata, not a fabricated common vintage.

Source sentinels `-99.99` and `-999` mean missing. Other extreme returns remain observed values;
a valid `-95%` is not a missing cell. Native tables retain nulls. The optional
[Empyrical acquisition adapter](../getting-started/migration.md#empyrical-acquisition-utilities)
uses the legacy complete-case sample, `Mom` spelling and pandas presentation without changing
native `MOM` columns or date filters.

## Academic Citations

- **FF3**: Fama and French (1993)
- **FF5**: Fama and French (2015)
- **Momentum**: Jegadeesh and Titman (1993), Carhart (1997)

---

## See Also

- [Research factor data sources](factors.md)
- [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- [AQR Provider](aqr.md)
- [Provider reference](index.md)
