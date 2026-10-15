# Contributing to ml4t-data

The [ML4T contribution policy](https://github.com/ml4t/ecosystem/blob/main/CONTRIBUTING.md)
defines the issue, pull-request, regression-evidence, and accountability requirements for this
repository. This guide adds the setup and verification steps specific to `ml4t-data`.

## Before opening an issue

Search the existing issues first. Each issue must describe one problem or coherent change. Open
separate issues for unrelated work.

A bug report must identify the affected package version and environment, state the observed and
expected behavior, and include the smallest complete reproduction through a public API or user
workflow. Report suspected vulnerabilities through
[private vulnerability reporting](https://github.com/ml4t/data/security/advisories/new), not a
public issue.

Discuss substantial changes in their owning issue before implementing them. A change to an intended
definition, numerical convention, default, or compatibility contract requires agreement on the
expected behavior and migration impact.

## Set up the repository

`ml4t-data` supports Python 3.12 through 3.14 and uses
[uv](https://docs.astral.sh/uv/) for environments and commands.

```bash
git clone https://github.com/ml4t/data.git ml4t-data
cd ml4t-data
uv sync --locked --all-extras --all-groups
uv run pre-commit install
```

Create a focused branch from the current `main` branch. Keep provider-specific dependencies
optional, do not require credentials or network access during import, and preserve the PEP 420
namespace layout under `src/ml4t/`. Use the repository's conventional commit prefixes, such as
`feat:`, `fix:`, `docs:`, `test:`, and `chore:`.

Use the test markers already declared in `pyproject.toml`. The default lane excludes live,
credentialed, paid-tier, and slow tests. The
[testing guide](https://www.ml4trading.io/docs/data/contributing/testing/) explains focused tests,
markers, optional dependencies, and live provider contracts. The
[provider guide](https://www.ml4trading.io/docs/data/contributing/creating-a-provider/) defines the
provider interfaces and required contract coverage.

## Verify the change

Run focused tests while editing, then run the repository checks before requesting review:

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check
uv run pytest tests -q -ra --cov=ml4t.data --cov-report=term-missing
uv run pytest tests -q -ra -W error::ResourceWarning
uv run mkdocs build --strict
uv build
actionlint .github/workflows/*.yml
pre-commit run --all-files
```

Bug fixes require executed regression evidence. Run the same command against the relevant base
revision and the proposed fix. Record the revisions, command, failing base result, and passing fix
result in the pull request. The regression must exercise the affected public API or user workflow.
Tests of copied logic or source text do not establish that behavior is corrected.

Correct verified numerical defects. Do not preserve an incorrect result solely for compatibility.
When a change intentionally alters a definition or convention, document the agreed behavior and
migration impact in the owning issue and user documentation.

## Open a pull request

Each pull request must address one issue and include a closing reference such as `Closes #123`.
Automated dependency updates and administrative metadata changes do not require a separate issue.

The pull request must describe the outcome and compatibility impact, report the commands and
results used for verification, and state any documentation, release-note, or patch-release work.
Disclose whether AI tools helped produce code or prose, identify the affected parts, and confirm
that you reviewed and tested the submission and can explain its evidence.

CI runs the repository and ecosystem qualification checks. Address review findings and failed
checks before merge. Maintainers squash-merge accepted pull requests.

## Get help

- Use [GitHub Issues](https://github.com/ml4t/data/issues) for reproducible defects and accepted
  changes.
- Read the [Data documentation](https://www.ml4trading.io/docs/data/) for supported workflows and
  provider guidance. File a documentation issue when that guidance is missing or incorrect.
