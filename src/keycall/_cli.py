"""keycall CLI: live credential verification (PRD section 14.2).

Each target gets one model-list call. With --generate, KeyCall walks the
filtered text models in provider order and reports the outcome of every
attempt until one succeeds or the attempt budget runs out.

This is *reported* fallthrough, not the silent fallthrough PRD section 14.3
forbids: each skipped model is printed with the reason it was skipped, so
provider drift (retired models still advertised, modality mismatches, quota
walls) stays visible instead of being masked. A credential failure stops
immediately — no point trying more models with a key the provider rejected.

Keys never appear in output.
"""

from __future__ import annotations

import argparse
import sys

from ._enums import ModelCategory
from ._errors import ErrorCode, KeyCallError
from ._sanitize import safe_display_name
from ._sources import SourceError, Target, load_targets, remind_deletion

_GENERATION_PROMPT = "Reply with the single word: ok"
_GENERATION_MAX_TOKENS = 16
_DEFAULT_ATTEMPTS = 8

# The credential itself is the problem: stop immediately, since no other
# model will fare better with a key the provider has rejected.
_CREDENTIAL_FAILURES = frozenset(
    {ErrorCode.INVALID_API_KEY, ErrorCode.PERMISSION_DENIED}
)
# Everything else is model-scoped and worth trying the next candidate for.
# Rate limits included, deliberately: providers meter per model and tier, so
# a 429 on one model says nothing about the next (Gemini free tier gives
# 2.5-pro zero quota while flash-latest answers fine). Exhausting the budget
# on rate limits is reported as unverified, never as a failed adapter
# (PRD 14.2: rate limits are distinct from adapter incompatibility).


def _verify_target(
    target: Target, *, generate: bool, attempts: int = _DEFAULT_ATTEMPTS
) -> tuple[bool, list[str]]:
    """Returns (ok, report_lines). Never raises for provider failures."""
    from ._client import KeyCall
    from ._types import Message, TextInput

    label = target.display_name
    lines = []
    try:
        with KeyCall(
            provider=target.provider,
            api_key=target.key,
            protocol=target.protocol,
            base_url=target.base_url,
        ) as client:
            # Verification must hit the live provider, never cached data.
            discovery = client.list_models(
                categories={ModelCategory.TEXT_GENERATION}, refresh=True
            )
            text_models = discovery.models
            lines.append(
                f"✓ {label} ({client.provider}): key accepted, "
                f"{len(text_models)} text model(s)"
            )
            if not generate:
                return True, lines
            if not text_models:
                lines.append(f"✗ {label}: no text models available to generate with")
                return False, lines

            messages = [Message(role="user", content=[TextInput(text=_GENERATION_PROMPT)])]
            rate_limited = False
            for position, candidate in enumerate(text_models[:attempts]):
                try:
                    result = client.generate_text(
                        model=candidate.id,
                        messages=messages,
                        max_output_tokens=_GENERATION_MAX_TOKENS,
                    )
                except KeyCallError as error:
                    lines.append(
                        f"  ✗ {candidate.id} (position {position}): "
                        f"{error.code.value} — {error.message}"
                    )
                    if error.code in _CREDENTIAL_FAILURES:
                        lines.append(f"✗ {label}: credential rejected")
                        return False, lines
                    if error.code is ErrorCode.RATE_LIMITED:
                        rate_limited = True
                    continue
                usage = result.usage.total_tokens
                skipped = f", {position} advertised model(s) skipped" if position else ""
                lines.append(
                    f"✓ {label}: generated with {candidate.id} "
                    f"(filtered position {position}{skipped}, "
                    f"{result.round_trip_duration_ms:.0f} ms, "
                    f"total tokens: {usage if usage is not None else 'unreported'})"
                )
                return True, lines

            tried = min(attempts, len(text_models))
            if rate_limited:
                lines.append(
                    f"! {label}: generation unverified — quota/rate limited "
                    f"({tried} attempted of {len(text_models)})"
                )
            else:
                lines.append(
                    f"✗ {label}: no advertised text model was invocable "
                    f"({tried} attempted of {len(text_models)})"
                )
            return False, lines
    except KeyCallError as error:
        lines.append(
            f"✗ {label} ({target.provider}): {error.code.value} — {error.message}"
            + (" [retryable]" if error.retryable else "")
        )
        return False, lines


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
        ok, lines = _verify_target(target, generate=args.generate, attempts=args.attempts)
        all_ok = all_ok and ok
        for line in lines:
            print(safe_display_name(line, max_length=300))

    if args.source:
        reminder = remind_deletion(args.source)
        if reminder:
            print(reminder, file=sys.stderr)

    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="keycall")
    subparsers = parser.add_subparsers(dest="command")

    verify = subparsers.add_parser(
        "verify", help="verify credentials against live providers"
    )
    verify.add_argument(
        "--source",
        "-s",
        help="TXT/JSON/TOML target file, env:VAR_NAME, or omit for interactive prompt",
    )
    verify.add_argument("--provider", help="provider name (env:/interactive sources)")
    verify.add_argument("--protocol", help="protocol override (custom targets)")
    verify.add_argument("--base-url", dest="base_url", help="base URL (custom targets)")
    verify.add_argument(
        "--generate",
        action="store_true",
        help="also make one bounded text generation per target",
    )
    verify.add_argument(
        "--attempts",
        type=int,
        default=_DEFAULT_ATTEMPTS,
        help=f"max models to try per target with --generate (default {_DEFAULT_ATTEMPTS})",
    )
    verify.add_argument(
        "--strict-credentials",
        action="store_true",
        help="treat credential-file warnings as errors",
    )

    args = parser.parse_args(argv)
    if args.command == "verify":
        return _run_verify(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
