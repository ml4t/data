# Contributing

Welcome to ML4T Data! We appreciate your interest in contributing.

## Ways to Contribute

<div class="grid cards" markdown>

-   :material-source-pull:{ .lg .middle } __Create a Provider__

    ---

    Add support for a new data source.

    [:octicons-arrow-right-24: Provider Guide](creating-a-provider.md)

-   :material-bug:{ .lg .middle } __Fix Bugs__

    ---

    Help improve reliability and fix issues.

    [:octicons-arrow-right-24: Testing Guide](testing.md)

-   :material-file-document:{ .lg .middle } __Improve Docs__

    ---

    Enhance documentation and examples.

    [:octicons-arrow-right-24: Architecture](architecture.md)

</div>

## Quick Start

```bash
# Clone the repository
git clone https://github.com/stefan-jansen/ml4t-data.git
cd ml4t-data

# Install with dev dependencies
uv sync --all-extras

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Code Style

- **Formatter**: ruff (100 char line length)
- **Type checking**: mypy strict mode
- **Docstrings**: Google style
- **Tests**: pytest with 80%+ coverage

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `pre-commit run --all-files`
5. Submit a pull request

## Getting Help

- [GitHub Issues](https://github.com/stefan-jansen/ml4t-data/issues)
- [Discussions](https://github.com/stefan-jansen/ml4t-data/discussions)
