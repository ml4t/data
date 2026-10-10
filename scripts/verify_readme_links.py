"""Verify every local and external link published in README.md."""

from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlsplit

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)")
REMOTE_SCHEMES = {"http", "https"}
IGNORED_SCHEMES = {"mailto", "tel"}


def extract_markdown_targets(markdown: str) -> list[str]:
    """Return distinct Markdown link and image targets in source order."""
    targets: list[str] = []
    for match in MARKDOWN_LINK.finditer(markdown):
        target = match.group("target").removeprefix("<").removesuffix(">")
        if target not in targets:
            targets.append(target)
    return targets


def _probe_url(url: str, attempts: int = 3, delay: float = 1.0) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,image/*,*/*;q=0.8",
            "User-Agent": "ml4t-documentation-link-verifier",
        },
    )
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                if response.status >= 400:
                    raise ValueError(f"HTTP {response.status}")
                response.read(1)
            return
        except (OSError, urllib.error.URLError, ValueError) as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise ValueError(str(error))


def readme_link_failures(
    readme: Path,
    probe_url: Callable[[str], None] = _probe_url,
) -> list[str]:
    """Return missing local targets and unavailable external URLs."""
    failures: list[str] = []
    for target in extract_markdown_targets(readme.read_text(encoding="utf-8")):
        parsed = urlsplit(target)
        if parsed.scheme in REMOTE_SCHEMES:
            try:
                probe_url(target)
            except (OSError, urllib.error.URLError, ValueError) as error:
                failures.append(f"{target}: {error}")
            continue
        if parsed.scheme in IGNORED_SCHEMES or target.startswith("#"):
            continue
        if parsed.scheme:
            failures.append(f"{target}: unsupported link scheme {parsed.scheme!r}")
            continue
        destination = readme.parent / unquote(parsed.path)
        if not destination.exists():
            failures.append(f"{target}: local target does not exist")
    return failures


def main() -> int:
    """Validate the configured README and print one line per failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("readme", type=Path, nargs="?", default=Path("README.md"))
    args = parser.parse_args()

    failures = readme_link_failures(args.readme)
    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
