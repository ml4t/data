"""Execute the Python quick start published in README.md."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
QUICK_START = re.compile(
    r"^## Quick start\s*$\n(?P<body>.*?)(?=^## |\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
PYTHON_BLOCK = re.compile(r"^```python\s*$\n(?P<source>.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def extract_quick_start(readme: str) -> str:
    """Return the single Python block in the README quick-start section."""
    section = QUICK_START.search(readme)
    if section is None:
        raise ValueError("README.md has no '## Quick start' section")
    blocks = [match["source"] for match in PYTHON_BLOCK.finditer(section["body"])]
    if len(blocks) != 1:
        raise ValueError(f"README quick start must contain one Python block, found {len(blocks)}")
    return blocks[0]


def main() -> None:
    """Compile and execute the documented quick start."""
    source = extract_quick_start((ROOT / "README.md").read_text(encoding="utf-8"))
    exec(compile(source, "README.md quick start", "exec"), {"__name__": "__main__"})  # noqa: S102


if __name__ == "__main__":
    main()
