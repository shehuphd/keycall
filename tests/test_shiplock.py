"""The docs-vs-code release gate, run inside the suite so a drift that would
block a tag surfaces here first, off the same shiplock.toml that `shiplock
check` and the CI gate read. The deterministic layer only; the semantic audit
runs in CI, where its API key lives."""

from pathlib import Path

from shiplock import load_config, run_checks

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_docs_match_code():
    report = run_checks(load_config(REPO_ROOT))
    assert report.ok, [f.message for f in report.findings]
