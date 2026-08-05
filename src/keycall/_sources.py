"""Smoke-test target sources for `keycall verify` (PRD sections 14.2.2-3).

Every loader produces the same Target type. Parsing never uses shell
evaluation or interpolation; key values are opaque and never appear in
diagnostics or parse errors — errors reference field names and line
numbers only.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from ._sanitize import safe_display_name

__all__ = ["SourceWarning", "Target", "load_targets"]

_ALLOWED_FIELDS = {"protocol", "provider", "key", "name", "base_url"}
_REQUIRED_FIELDS = {"provider", "key"}

# key=value tokens; values may be single- or double-quoted with escaped
# quotes, or bare (no whitespace).
_TXT_TOKEN = re.compile(
    r"""(?P<field>[A-Za-z_]+)=(?P<value>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\S+)"""
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Target:
    provider: str
    key: str
    protocol: str | None = None
    name: str | None = None
    base_url: str | None = None

    @property
    def display_name(self) -> str:
        return safe_display_name(self.name or self.provider)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceWarning:
    message: str


class SourceError(ValueError):
    """Parse or validation failure. Never contains a key value."""


def _unquote(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        quote = raw[0]
        inner = raw[1:-1]
        return inner.replace(f"\\{quote}", quote).replace("\\\\", "\\")
    return raw


def _target_from_mapping(fields: dict[str, str], *, where: str) -> Target:
    unknown = set(fields) - _ALLOWED_FIELDS
    if unknown:
        raise SourceError(f"{where}: unknown field(s) {sorted(unknown)}")
    missing = _REQUIRED_FIELDS - set(fields)
    if missing:
        raise SourceError(f"{where}: missing required field(s) {sorted(missing)}")
    if not fields["key"].strip():
        raise SourceError(f"{where}: key is empty")
    return Target(
        provider=fields["provider"].strip().lower(),
        key=fields["key"],
        protocol=fields.get("protocol", "").strip().lower() or None,
        name=fields.get("name"),
        base_url=fields.get("base_url") or None,
    )


def _parse_txt(text: str) -> list[Target]:
    targets = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields: dict[str, str] = {}
        matched_spans = []
        for match in _TXT_TOKEN.finditer(stripped):
            field = match.group("field").lower()
            if field in fields:
                raise SourceError(f"line {line_number}: duplicate field {field!r}")
            fields[field] = _unquote(match.group("value"))
            matched_spans.append(match.span())
        # Anything outside matched tokens (other than whitespace) is a
        # malformed record — refuse rather than guess.
        leftover = stripped
        for start, end in reversed(matched_spans):
            leftover = leftover[:start] + leftover[end:]
        if leftover.strip():
            raise SourceError(f"line {line_number}: malformed record")
        targets.append(_target_from_mapping(fields, where=f"line {line_number}"))
    return targets


def _parse_json(text: str) -> list[Target]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SourceError(f"invalid JSON: line {exc.lineno}") from None
    entries = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise SourceError("JSON source must be a list of targets or {\"targets\": [...]}")
    targets = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SourceError(f"target {index}: not an object")
        fields = {str(k): str(v) for k, v in entry.items()}
        targets.append(_target_from_mapping(fields, where=f"target {index}"))
    return targets


def _parse_toml(text: str) -> list[Target]:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        raise SourceError("invalid TOML") from None
    entries = payload.get("targets")
    if not isinstance(entries, list):
        raise SourceError("TOML source must define [[targets]] tables")
    targets = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SourceError(f"target {index}: not a table")
        fields = {str(k): str(v) for k, v in entry.items()}
        targets.append(_target_from_mapping(fields, where=f"target {index}"))
    return targets


def _file_warnings(path: Path) -> list[SourceWarning]:
    warnings = []
    try:
        mode = path.stat().st_mode
    except OSError:
        return warnings
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        warnings.append(
            SourceWarning(
                message=f"{path.name} is readable by other users; consider chmod 600"
            )
        )
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            warnings.append(
                SourceWarning(
                    message=f"{path.name} sits inside a git repository; "
                    "keep credential files out of version control"
                )
            )
            break
    return warnings


def load_targets(
    source: str,
    *,
    provider: str | None = None,
    protocol: str | None = None,
    base_url: str | None = None,
) -> tuple[list[Target], list[SourceWarning]]:
    """Load targets from a file path, ``env:VAR_NAME``, or ``-`` (interactive).

    env: and interactive sources describe one target and take provider /
    protocol / base_url from the accompanying CLI arguments.
    """
    if source.startswith("env:"):
        variable = source[4:]
        if not variable:
            raise SourceError("env: source needs a variable name, e.g. env:MY_TEST_KEY")
        value = os.environ.get(variable)
        if value is None or not value.strip():
            raise SourceError(f"environment variable {variable} is not set or empty")
        if not provider:
            raise SourceError("--provider is required with an env: source")
        return (
            [
                Target(
                    provider=provider,
                    key=value,
                    protocol=protocol,
                    base_url=base_url,
                    name=variable,
                )
            ],
            [],
        )

    if source == "-":
        import getpass

        prompt_provider = provider or input("Provider: ").strip().lower()
        key = getpass.getpass("API key: ")
        if not key.strip():
            raise SourceError("no key entered")
        return (
            [Target(provider=prompt_provider, key=key, protocol=protocol, base_url=base_url)],
            [],
        )

    path = Path(source)
    if not path.is_file():
        raise SourceError(f"source file not found: {path}")
    text = path.read_text("utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        targets = _parse_json(text)
    elif suffix == ".toml":
        targets = _parse_toml(text)
    else:
        targets = _parse_txt(text)
    if not targets:
        raise SourceError("source contains no targets")
    return targets, _file_warnings(path)


def remind_deletion(source: str) -> str | None:
    if source.startswith("env:") or source == "-":
        return None
    return (
        f"reminder: {Path(source).name} contains credentials; delete it when "
        "you no longer need it (KeyCall never deletes user files)"
    )
