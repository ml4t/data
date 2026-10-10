"""Verify rendered and deployed documentation release identity."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

REQUIRED_META = ("ml4t-library", "ml4t-version", "ml4t-commit")


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = dict(attrs)
        name = values.get("name")
        content = values.get("content")
        if name in REQUIRED_META and content is not None:
            self.values[name] = content


class _ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        anchor = values.get("id") or (values.get("name") if tag == "a" else None)
        if anchor:
            self.anchors.add(anchor)
        attribute = {
            "a": "href",
            "img": "src",
            "link": "href",
            "script": "src",
        }.get(tag)
        reference = values.get(attribute) if attribute else None
        if reference:
            self.references.append(reference)


def identity_failures(
    html: str,
    *,
    expected_library: str,
    expected_version: str,
    expected_commit: str,
    source: str,
) -> list[str]:
    """Return mismatched or missing documentation identity fields."""
    parser = _MetadataParser()
    parser.feed(html)
    expected = {
        "ml4t-library": expected_library,
        "ml4t-version": expected_version,
        "ml4t-commit": expected_commit,
    }
    return [
        f"{source}: {name} is {parser.values.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if parser.values.get(name) != value
    ]


def _site_pages(site: Path) -> list[Path]:
    return sorted(
        path
        for path in site.rglob("*.html")
        if path.is_file() and "overrides" not in path.relative_to(site).parts
    )


def _read_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ml4t-release-verifier"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        if response.status != 200:
            raise ValueError(f"{url}: HTTP {response.status}")
        return response.read().decode("utf-8")


def _local_destination(
    site: Path,
    page: Path,
    reference: str,
    site_path: str,
) -> tuple[Path, str] | None:
    parsed = urllib.parse.urlsplit(reference)
    if parsed.path.startswith("/"):
        if not parsed.path.startswith(site_path):
            return None
        destination = site / urllib.parse.unquote(parsed.path.removeprefix(site_path))
    else:
        destination = page.parent / urllib.parse.unquote(parsed.path)
    if not parsed.path or parsed.path.endswith("/") or destination.is_dir():
        destination /= "index.html"
    return destination.resolve(), urllib.parse.unquote(parsed.fragment)


def site_link_failures(site: Path, site_path: str = "/docs/data/") -> list[str]:
    """Return broken internal links, assets, and fragments in a rendered site."""
    failures: list[str] = []
    site_root = site.resolve()
    anchor_cache: dict[Path, set[str]] = {}
    for page in _site_pages(site):
        parser = _ReferenceParser()
        parser.feed(page.read_text(encoding="utf-8"))
        anchor_cache[page.resolve()] = parser.anchors
        for reference in parser.references:
            parsed = urllib.parse.urlsplit(reference)
            if parsed.scheme or parsed.netloc or reference.startswith(("mailto:", "tel:", "data:")):
                continue
            resolved = _local_destination(site, page, reference, site_path)
            if resolved is None:
                continue
            destination, fragment = resolved
            if destination != site_root and site_root not in destination.parents:
                failures.append(f"{page}: {reference!r} escapes the rendered site")
                continue
            if not destination.exists():
                failures.append(f"{page}: {reference!r} does not resolve")
                continue
            if fragment and destination.suffix == ".html":
                anchors = anchor_cache.get(destination)
                if anchors is None:
                    target_parser = _ReferenceParser()
                    target_parser.feed(destination.read_text(encoding="utf-8"))
                    anchors = target_parser.anchors
                    anchor_cache[destination] = anchors
                if fragment not in anchors:
                    failures.append(f"{page}: {reference!r} has no matching anchor")
    return failures


def page_reference_urls(html: str, *, page_url: str, site_root: str) -> list[str]:
    """Return distinct same-site documentation references from one deployed page."""
    parser = _ReferenceParser()
    parser.feed(html)
    root = urllib.parse.urlsplit(site_root)
    references: list[str] = []
    for reference in parser.references:
        absolute = urllib.parse.urljoin(page_url, reference)
        parsed = urllib.parse.urlsplit(absolute)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != root.netloc:
            continue
        if not parsed.path.startswith(root.path):
            continue
        normalized = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
        )
        if normalized not in references:
            references.append(normalized)
    return references


def _probe_url(url: str, attempts: int = 3, delay: float = 1.0) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ml4t-release-verifier"})
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}")
                response.read(1)
            return
        except (OSError, urllib.error.URLError, ValueError) as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise ValueError(str(error))


def deployed_link_failures(urls: list[str]) -> list[str]:
    """Return unavailable navigation, asset, and internal-link targets."""
    site_root = urls[0]
    references: list[str] = []
    failures: list[str] = []
    for url in urls:
        try:
            html = _read_url(url)
            for reference in page_reference_urls(html, page_url=url, site_root=site_root):
                if reference not in references:
                    references.append(reference)
        except (OSError, UnicodeDecodeError, urllib.error.URLError, ValueError) as error:
            failures.append(f"{url}: {error}")
    for reference in references:
        try:
            _probe_url(reference)
        except (OSError, urllib.error.URLError, ValueError) as error:
            failures.append(f"{reference}: {error}")
    return failures


def deployed_identity_failures(
    urls: list[str],
    *,
    expected_library: str,
    expected_version: str,
    expected_commit: str,
    attempts: int,
    delay: float,
) -> list[str]:
    """Retry deployed pages until every URL reports the expected identity."""
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        failures = []
        try:
            for url in urls:
                failures.extend(
                    identity_failures(
                        _read_url(url),
                        expected_library=expected_library,
                        expected_version=expected_version,
                        expected_commit=expected_commit,
                        source=url,
                    )
                )
        except (OSError, UnicodeDecodeError, urllib.error.URLError, ValueError) as error:
            failures = [str(error)]
        if not failures:
            break
        if attempt < attempts:
            time.sleep(delay)
    return failures


def main() -> int:
    """Validate a rendered site or one or more deployed URLs."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--site", type=Path)
    source.add_argument("--url", action="append")
    parser.add_argument("--expected-library", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    if args.attempts < 1 or args.delay < 0:
        parser.error("--attempts must be positive and --delay must be non-negative")
    if len(args.expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.expected_commit
    ):
        parser.error("--expected-commit must be a full lowercase Git SHA")

    failures: list[str] = []
    try:
        if args.site is not None:
            pages = _site_pages(args.site)
            if not pages:
                failures.append(f"{args.site}: no rendered HTML pages found")
            for page in pages:
                failures.extend(
                    identity_failures(
                        page.read_text(encoding="utf-8"),
                        expected_library=args.expected_library,
                        expected_version=args.expected_version,
                        expected_commit=args.expected_commit,
                        source=str(page),
                    )
                )
            failures.extend(site_link_failures(args.site))
    except (OSError, UnicodeDecodeError, urllib.error.URLError, ValueError) as error:
        failures.append(str(error))
    if args.url is not None:
        failures.extend(
            deployed_identity_failures(
                args.url,
                expected_library=args.expected_library,
                expected_version=args.expected_version,
                expected_commit=args.expected_commit,
                attempts=args.attempts,
                delay=args.delay,
            )
        )
        if not failures:
            failures.extend(deployed_link_failures(args.url))

    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
