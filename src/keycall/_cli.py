"""keycall CLI: live credential verification and the local viewer.

`verify` walks filtered text models in provider order and reports the
outcome of every attempt until one succeeds or the attempt budget runs out
(the walk itself lives in `_verify_core.py`, shared with the viewer). This
is *reported* fallthrough, never silent: each skipped model is printed
with the reason it was skipped, so provider drift stays visible instead
of being masked.

Keys never appear in output.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from typing import TextIO

from ._sanitize import safe_display_name
from ._sources import SourceError, load_targets, remind_deletion
from ._verify_core import DEFAULT_ATTEMPTS, VerifyResult, run_verify

USAGE_URL = "https://github.com/shehuphd/keycall/blob/main/USAGE.md"

_COMMANDS = ("verify", "view")

# Color marks a fixed category and nothing else: green for a pass, red for
# a failure, yellow for a warning or an unverified outcome. Codes wrap the
# finished string last, so width and alignment are computed on plain text.
_GREEN, _RED, _YELLOW = "32", "31", "33"


def _color_on(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _paint(text: str, code: str, stream: TextIO) -> str:
    if not _color_on(stream):
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


_MARKER_COLORS = {"✓": _GREEN, "✗": _RED, "!": _YELLOW}


def _emit(line: str, stream: TextIO) -> None:
    """Print one report line, coloring only its leading ✓/✗/! marker."""
    stripped = line.lstrip()
    marker = stripped[:1]
    code = _MARKER_COLORS.get(marker)
    if code:
        indent = line[: len(line) - len(stripped)]
        line = indent + _paint(marker, code, stream) + stripped[1:]
    print(line, file=stream)


def _warn(message: str) -> None:
    print(f"{_paint('warning:', _YELLOW, sys.stderr)} {message}", file=sys.stderr)


def _fail(message: str) -> None:
    """One failure sentence to stderr, blank-line spaced so it stands apart."""
    print(file=sys.stderr)
    _emit(f"✗ {message}", sys.stderr)
    print(file=sys.stderr)


_KEY_SHAPE = re.compile(r"[A-Za-z0-9._\-]{16,}")


def _looks_like_key(token: str) -> bool:
    """A long unbroken token with digits reads as a pasted credential.
    Better to hide a harmless value than to echo a live key back into
    the terminal a second time."""
    if token.startswith("-"):
        return False
    return bool(_KEY_SHAPE.fullmatch(token)) and any(c.isdigit() for c in token)


def _suggest(token: str, candidates: list[str]) -> str | None:
    matches = difflib.get_close_matches(token, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


_KEY_GUIDANCE = (
    (
        "Anything typed on the command line is saved in your shell history, so "
        "treat that key as exposed: clear the history line, and rotate the key "
        "if it's live."
    ),
    (
        "keycall asks for keys in safer ways. Run `keycall verify` and paste the "
        "key at the hidden prompt, point at a file with "
        "`keycall verify --source keys.toml`, or name an environment variable "
        "with `keycall verify --source env:MY_KEY`."
    ),
)


class _UsageError(Exception):
    def __init__(self, message: str, parser: argparse.ArgumentParser) -> None:
        super().__init__(message)
        self.message = message
        self.parser = parser


class _Parser(argparse.ArgumentParser):
    """Raises instead of printing argparse's raw usage dump, so every
    parse failure goes through the plain-language renderer below."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(message, self)


def _option_names(parser: argparse.ArgumentParser) -> list[str]:
    """Every --flag this parser or its subcommands accept. An unrecognized
    flag surfaces on the top-level parser even when it was meant for a
    subcommand, so the suggestion pool has to span all of them."""
    names: list[str] = []
    for action in parser._actions:  # argparse offers no public listing
        names.extend(s for s in action.option_strings if s.startswith("--"))
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                names.extend(_option_names(sub))
    return sorted(set(names))


def _describe_unrecognized(tokens: list[str], parser: argparse.ArgumentParser) -> list[str]:
    """Plain-language pieces for tokens argparse couldn't place. The first
    bad token gets at most one confident guess; a pasted key gets hidden
    and answered with the safe ways in."""
    shown = []
    key_seen = False
    for token in tokens:
        if _looks_like_key(token):
            shown.append(f'"{token[:3]}…" (hidden — it looks like an API key)')
            key_seen = True
        else:
            shown.append(f'"{token}"')
    pieces = [f"keycall doesn't recognize {', '.join(shown)}."]
    if key_seen:
        pieces.extend(_KEY_GUIDANCE)
    else:
        first = tokens[0]
        candidates = _option_names(parser) if first.startswith("-") else list(_COMMANDS)
        guess = _suggest(first, candidates)
        if guess:
            pieces.append(f"Perhaps you meant `{guess}`?")
    return pieces


def _usage_error(error: _UsageError) -> int:
    """Translate argparse's message into sentences, block-spaced on stderr."""
    message, parser = error.message, error.parser
    pieces: list[str]

    unrecognized = re.fullmatch(r"unrecognized arguments: (.+)", message)
    bad_choice = re.search(r"invalid choice: '(.+?)'", message)
    bad_int = re.search(r"argument (\S+): invalid int value: '(.+?)'", message)
    needs_value = re.search(r"argument (\S+?)(?:/\S+)*: expected one argument", message)

    if unrecognized:
        pieces = _describe_unrecognized(unrecognized.group(1).split(), parser)
    elif bad_choice:
        token = bad_choice.group(1)
        hidden = _looks_like_key(token)
        display = f'"{token[:3]}…" (hidden — it looks like an API key)' if hidden else f'"{token}"'
        pieces = [
            (
                f"{display} isn't a keycall command. keycall knows two: "
                "`verify` checks keys against their live providers, and `view` "
                "opens the local viewer in your browser."
            )
        ]
        if hidden:
            pieces.extend(_KEY_GUIDANCE)
        else:
            guess = _suggest(token, list(_COMMANDS))
            if guess:
                pieces.append(f"Perhaps you meant `keycall {guess}`?")
    elif bad_int:
        name, value = bad_int.groups()
        example = {"--port": "8823"}.get(name, "4")
        pieces = [
            (
                f'{name} needs a whole number, and "{value}" isn\'t one. '
                f"For example: {name} {example}."
            )
        ]
    elif needs_value:
        name = needs_value.group(1)
        pieces = [f"{name} needs a value right after it. For example: {name} keys.toml."]
    else:
        pieces = [f"keycall couldn't read that command ({message})."]

    prog = parser.prog if parser.prog.startswith("keycall") else "keycall"
    pieces.append(f"Run `keycall` alone to see what it can do, or `{prog} --help` for every option.")

    print(file=sys.stderr)
    _emit(f"✗ {pieces[0]}", sys.stderr)
    for piece in pieces[1:]:
        print(file=sys.stderr)
        print(piece, file=sys.stderr)
    print(file=sys.stderr)
    return 2


def _welcome() -> int:
    from . import __version__

    print(f"keycall {__version__} — checks AI-provider API keys, lists their models, and makes live calls.")
    print()
    print("Try one of these:")
    print()
    print("  keycall verify               check the keys in a file, or paste one at a hidden prompt")
    print("  keycall verify --generate    also make one small billable call per key")
    print("  keycall view                 open the local viewer in your browser")
    print()
    print(f"Every option: keycall verify --help, keycall view --help. Full manual: {USAGE_URL}")
    return 0


def _render(result: VerifyResult) -> list[str]:
    lines: list[str] = []
    if not result.listed_ok:
        lines.append(
            f"✗ {result.label} ({result.provider}): "
            f"{result.list_error_code} — {result.list_error_message}"
        )
        return lines

    lines.append(
        f"✓ {result.label} ({result.provider}): key accepted, "
        f"{result.text_model_count} text model(s), "
        f"list digest {result.model_list_digest}, "
        f"selection rule v{result.selection_rule_version}"
    )
    if not result.generate_requested:
        return lines
    if result.outcome == "no_text_models":
        lines.append(f"✗ {result.label}: no text models available to generate with")
        return lines

    for attempt in result.attempts:
        if attempt.ok:
            skipped = f", {attempt.position} advertised model(s) skipped" if attempt.position else ""
            usage = attempt.total_tokens if attempt.total_tokens is not None else "unreported"
            lines.append(
                f"✓ {result.label}: generated with {attempt.model_id} "
                f"(filtered position {attempt.position}, "
                f"provider-list position {attempt.raw_position}{skipped}, "
                f"{attempt.round_trip_duration_ms:.0f} ms, total tokens: {usage})"
            )
        else:
            lines.append(
                f"  ✗ {attempt.model_id} (filtered position {attempt.position}, "
                f"provider-list position {attempt.raw_position}): "
                f"{attempt.error_code} — {attempt.error_message}"
            )

    if result.outcome == "credential_rejected":
        lines.append(f"✗ {result.label}: credential rejected")
    elif result.outcome == "rate_limited_unverified":
        tried = len(result.attempts)
        lines.append(
            f"! {result.label}: generation unverified — quota/rate limited "
            f"({tried} attempted of {result.text_model_count})"
        )
    elif result.outcome == "no_model_invocable":
        tried = len(result.attempts)
        lines.append(
            f"✗ {result.label}: no advertised text model was invocable "
            f"({tried} attempted of {result.text_model_count})"
        )
    return lines


def _run_verify(args: argparse.Namespace) -> int:
    try:
        targets, warnings = load_targets(
            args.source or "-",
            provider=args.provider,
            protocol=args.protocol,
            base_url=args.base_url,
        )
    except SourceError as error:
        _fail(str(error))
        return 2

    for warning in warnings:
        if args.strict_credentials:
            _fail(f"{warning.message} (--strict-credentials treats this as an error)")
            return 2
        _warn(warning.message)

    all_ok = True
    for target in targets:
        result = run_verify(target, generate=args.generate, attempts=args.attempts)
        all_ok = all_ok and (result.generate_ok if args.generate else result.listed_ok)
        for line in _render(result):
            _emit(safe_display_name(line, max_length=300), sys.stdout)

    if args.source:
        reminder = remind_deletion(args.source)
        if reminder:
            print(reminder, file=sys.stderr)

    return 0 if all_ok else 1


def _run_view(args: argparse.Namespace) -> int:
    if args.source:
        try:
            targets, warnings = load_targets(
                args.source,
                provider=args.provider,
                protocol=args.protocol,
                base_url=args.base_url,
            )
        except SourceError as error:
            _fail(str(error))
            return 2
        for warning in warnings:
            _warn(warning.message)
    else:
        # No source: start empty — the viewer prompts for a key file.
        targets = []

    try:
        from .viewer import run as run_viewer
    except ImportError:
        _fail(
            "this install is missing the viewer. Reinstall with "
            "`python3 -m pip install --force-reinstall keycall` and try again."
        )
        return 2

    return run_viewer(
        targets,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        reload=args.reload,
    )


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        "-s",
        help="TXT/JSON/TOML target file, env:VAR_NAME, or omit for interactive prompt",
    )
    parser.add_argument("--provider", help="provider name (env:/interactive sources)")
    parser.add_argument("--protocol", help="protocol override (custom targets)")
    parser.add_argument("--base-url", dest="base_url", help="base URL (custom targets)")


def main(argv: list[str] | None = None) -> int:
    from . import __version__

    parser = _Parser(
        prog="keycall",
        description="Checks AI-provider API keys, lists their models, and makes live calls.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"keycall {__version__}")
    subparsers = parser.add_subparsers(dest="command", parser_class=_Parser)

    verify = subparsers.add_parser(
        "verify",
        help="check keys against their live providers",
        description="Checks each key against its live provider and reports the outcome.",
        allow_abbrev=False,
    )
    _add_source_args(verify)
    verify.add_argument(
        "--generate", action="store_true", help="also make one bounded text generation per target"
    )
    verify.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help=f"max models to try per target with --generate (default {DEFAULT_ATTEMPTS})",
    )
    verify.add_argument(
        "--strict-credentials", action="store_true", help="treat credential-file warnings as errors"
    )

    view = subparsers.add_parser(
        "view",
        help="open the local viewer in your browser",
        description="Starts the local viewer and opens it in your browser.",
        allow_abbrev=False,
    )
    _add_source_args(view)
    view.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    view.add_argument("--port", type=int, default=0, help="port (default: pick a free one)")
    view.add_argument(
        "--no-open", action="store_true", help="don't open a browser tab automatically"
    )
    view.add_argument(
        "--reload",
        action="store_true",
        help="restart when keycall's source changes, keeping the address and "
        "token, so a hard reload in the browser picks up server-side edits",
    )

    try:
        args = parser.parse_args(argv)
    except _UsageError as error:
        return _usage_error(error)
    if args.command == "verify":
        return _run_verify(args)
    if args.command == "view":
        return _run_view(args)
    return _welcome()


if __name__ == "__main__":
    raise SystemExit(main())
