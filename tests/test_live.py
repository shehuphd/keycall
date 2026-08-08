"""Live provider smoke tests (PRD 14.2.1).

Deselected by default; select with `pytest -m live`. Credentials load only
when this module's test actually runs, from the target file named by
KEYCALL_LIVE_SOURCE, so ordinary runs never touch the environment for
key-like values.

Mode is a CI concern, not a test concern: `warn` runs this job with
continue-on-error, `strict` blocks the release on any failure. The test
itself distinguishes rate-limit outcomes (a verification-environment
failure: the release stays unverified, but the adapter is not implicated)
from adapter or credential failures.
"""

from __future__ import annotations

import os

import pytest

from keycall._sources import load_targets
from keycall._verify_core import run_verify

pytestmark = pytest.mark.live


def test_live_smoke_every_target_generates():
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        pytest.skip("KEYCALL_LIVE_SOURCE not set; live verification needs a target file")
    targets, _ = load_targets(source)

    verified: list[str] = []
    rate_limited: list[str] = []
    failed: list[str] = []
    for target in targets:
        result = run_verify(target, generate=True)
        summary = (
            f"{result.label} ({result.provider}): {result.outcome}, "
            f"digest {result.model_list_digest}, rule v{result.selection_rule_version}, "
            f"{len(result.attempts)} attempt(s)"
        )
        for attempt in result.attempts:
            summary += (
                f"\n    {'ok' if attempt.ok else attempt.error_code}: {attempt.model_id} "
                f"(filtered {attempt.position}, raw {attempt.raw_position}, "
                f"{attempt.classification_source})"
            )
        print(summary)
        if result.generate_ok:
            verified.append(result.label)
        elif result.outcome == "rate_limited_unverified":
            rate_limited.append(summary)
        else:
            failed.append(summary)

    report = []
    if failed:
        report.append("adapter/credential failures:\n" + "\n".join(failed))
    if rate_limited:
        report.append(
            "verification-environment failures (rate limited, provider not "
            "implicated, release still unverified):\n" + "\n".join(rate_limited)
        )
    assert not report, "\n\n".join(report)
    assert verified, "no targets were verified"
