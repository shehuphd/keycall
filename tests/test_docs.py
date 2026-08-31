"""Docs hygiene, enforced mechanically rather than by review habit.

Two guarantees:

1. No public doc or shipped source file references an internal artifact —
   the gitignored planning folder, internal standards files, the roadmap.
   The public record states behavior directly; internal reasoning lives in
   files that never ship.
2. Every markdown link in README.md is absolute. PyPI resolves relative
   links against pypi.org and freezes each release's rendered README at
   build time, so a relative link becomes a permanently dead link on the
   project page for that release.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PUBLIC_DOCS = ("README.md", "USAGE.md", "ARCHITECTURE.md", "CHANGELOG.md", "MANIFEST.md")

# Each pattern names an internal artifact that must never appear on a
# public surface. `project/` allows pypi.org/project/<name> URLs, the one
# legitimate public string containing it.
INTERNAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!pypi\.org/)\bproject/"), "the internal project/ folder"),
    (re.compile(r"CODING\.md"), "the internal coding-standards file"),
    (re.compile(r"ROADMAP"), "the internal roadmap"),
    (re.compile(r"\.claude"), "assistant tool state"),
)


def _shipped_sources() -> list[Path]:
    package = ROOT / "src" / "keycall"
    return sorted(
        path
        for pattern in ("*.py", "*.html", "*.js", "*.css", "*.json")
        for path in package.rglob(pattern)
        if "__pycache__" not in path.parts
    )


def _internal_hits(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, label in INTERNAL_PATTERNS:
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno} mentions {label}: {line.strip()!r}")
    return hits


@pytest.mark.parametrize("name", PUBLIC_DOCS)
def test_public_docs_never_reference_internal_artifacts(name):
    hits = _internal_hits(ROOT / name)
    assert not hits, (
        "public docs must state behavior directly, never point at internal files; "
        "reword these lines:\n" + "\n".join(hits)
    )


def test_shipped_source_never_references_internal_artifacts():
    hits = [hit for path in _shipped_sources() for hit in _internal_hits(path)]
    assert not hits, (
        "shipped source must not name internal files; reword these lines:\n" + "\n".join(hits)
    )


def test_readme_links_are_absolute():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    # Inline markdown links only; bare URLs and code spans don't resolve as
    # links on PyPI in the first place.
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    relative = [t for t in targets if not t.startswith(("http://", "https://", "#"))]
    assert not relative, (
        "PyPI resolves relative README links against pypi.org and freezes the rendered "
        "page per release; make these absolute GitHub URLs: " + ", ".join(relative)
    )
