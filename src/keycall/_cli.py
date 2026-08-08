"""keycall CLI: live credential verification and the local viewer.

`verify` walks filtered text models in provider order and reports the
outcome of every attempt until one succeeds or the attempt budget runs out
(the walk itself lives in `_verify_core.py`, shared with the viewer). This
is *reported* fallthrough, not the silent fallthrough PRD section 14.3
forbids: each skipped model is printed with the reason it was skipped, so
provider drift stays visible instead of being masked.

Keys never appear in output.
"""

from __future__ import annotations

import argparse
import sys

from ._sanitize import safe_display_name
from ._sources import SourceError, load_targets, remind_deletion
from ._verify_core import DEFAULT_ATTEMPTS, VerifyResult, run_verify


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
        print(f"error: {error}", file=sys.stderr)
        return 2

    for warning in warnings:
        if args.strict_credentials:
            print(f"error (strict): {warning.message}", file=sys.stderr)
            return 2
        print(f"warning: {warning.message}", file=sys.stderr)

    all_ok = True
    for target in targets:
        result = run_verify(target, generate=args.generate, attempts=args.attempts)
        all_ok = all_ok and (result.generate_ok if args.generate else result.listed_ok)
        for line in _render(result):
            print(safe_display_name(line, max_length=300))

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
            print(f"error: {error}", file=sys.stderr)
            return 2
        for warning in warnings:
            print(f"warning: {warning.message}", file=sys.stderr)
    else:
        # No source: start empty — the viewer prompts for a key file.
        targets = []

    try:
        from .viewer import run as run_viewer
    except ImportError:
        print("keycall view is under construction and not functional yet", file=sys.stderr)
        return 2

    return run_viewer(targets, host=args.host, port=args.port, open_browser=not args.no_open)


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
    parser = argparse.ArgumentParser(prog="keycall")
    subparsers = parser.add_subparsers(dest="command")

    verify = subparsers.add_parser("verify", help="verify credentials against live providers")
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

    view = subparsers.add_parser("view", help="open the local web viewer for a target source")
    _add_source_args(view)
    view.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    view.add_argument("--port", type=int, default=0, help="port (default: pick a free one)")
    view.add_argument(
        "--no-open", action="store_true", help="don't open a browser tab automatically"
    )

    args = parser.parse_args(argv)
    if args.command == "verify":
        return _run_verify(args)
    if args.command == "view":
        return _run_view(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
