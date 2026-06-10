# Polygon Provider

`PolygonProvider` remains available as a deprecated compatibility alias for
existing Polygon.io integrations. Polygon.io rebranded to Massive.com, and
existing Polygon API keys continue to work.

New code should use:

```python
from ml4t.data.providers import MassiveProvider
```

Set `MASSIVE_API_KEY` for new environments. Existing `POLYGON_API_KEY` values are
still supported.

See the canonical [Massive provider](massive.md) documentation for current
coverage, examples, and limitations.
