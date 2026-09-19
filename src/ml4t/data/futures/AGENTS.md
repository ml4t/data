# Futures subsystem guide

This directory owns futures symbology, Databento-backed acquisition, continuous-contract
construction, roll selection, and price adjustment. It is not the general provider registry or the
generic storage layer.

## Change locations

| File | Responsibility |
|---|---|
| `parser.py`, `databento_parser.py` | Contract symbols and Databento record conversion |
| `schema.py`, `definitions.py` | Contract specifications and definition data |
| `downloader.py` | Parent-symbol downloads |
| `continuous_downloader.py`, `individual_downloader.py` | Continuous and individual-contract acquisition |
| `continuous.py`, `roll.py`, `adjustment.py` | Continuous series, roll decisions, and price adjustment |
| `config.py` | Futures-specific configuration |
| `book_downloader.py` | Higher-level futures workflow used by the book examples |

## Invariants

- Keep symbol parsing and contract-calendar logic separate from network acquisition.
- Preserve the distinction between parent, individual, and continuous symbols in code and tests.
- Route filesystem defaults through the package data-root helpers; do not embed workstation paths.
- Keep Databento optional. Importing unrelated `ml4t.data` modules must not require the extra.
- Mock external calls in the default test suite. Credentialed or live checks use the established
  integration markers.
- Treat output schemas and roll or adjustment behavior as public contracts. Add regression tests for
  boundary dates and representative symbols when changing them.

Run `uv run pytest tests/futures -q -ra` for focused verification, then run the root quality gates.
User-facing behavior and examples belong in the futures API documentation and
`docs/book-guide/index.md`, not in this file.
