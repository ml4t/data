# Contributing

Contributions should preserve the public provider, storage, and validation contracts while keeping
the default test suite deterministic and offline.

## Set up the repository

Install `uv`, clone the canonical repository, and create the complete locked environment:

```bash
git clone https://github.com/ml4t/data.git
cd data
uv sync --locked --all-extras --all-groups
uv run pre-commit install
```

Run the offline test lane once to confirm the checkout:

```bash
uv run pytest tests -q -ra
```

Provider tests that contact live services are excluded by default. See the
[testing guide](testing.md) before running a credentialed or paid-tier test.

## Choose the relevant guide

- [Creating a provider](creating-a-provider.md) covers the provider contract, registry metadata,
  deterministic tests, and documentation required for a new adapter.
- [Architecture](architecture.md) explains the boundaries among providers, validation, storage,
  configuration, and orchestration.
- [Testing](testing.md) lists the offline, resource-warning, focused, and live-provider lanes.

The root [`AGENTS.md`](https://github.com/ml4t/data/blob/main/AGENTS.md) records the repository map,
change rules, and complete verification commands. More specific `AGENTS.md` files apply under the
provider, storage, and futures source directories.

## Pull requests

1. Open or reference one issue that defines the problem and acceptance criteria.
2. Make the smallest coherent change and add a test that fails for the behavior being corrected.
3. Run focused checks while editing, then the repository gates from `AGENTS.md`.
4. Complete the pull request template, including compatibility and release implications.

Use [private vulnerability reporting](https://github.com/ml4t/data/security/advisories/new) for a
suspected security issue. Use the [issue tracker](https://github.com/ml4t/data/issues) for other
bugs, documentation problems, and feature requests.
