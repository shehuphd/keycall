import os

import pytest


def pytest_collection_finish(session):
    """Pre-flight for the live suite: before any test spends a billable call
    (batch submissions happen at session setup, then generates, streams, and
    tool rounds), confirm every live target's key is usable. A provider
    that's unreachable now — invalid key, no credits, rate limited, or down —
    is named here, up front, and the run stops in seconds with the cause
    instead of crashing deep inside a fixture or leaving it to a later test.

    The check does a minimal generate per target, because an out-of-credits
    or rate-limited key still lists models fine; only an inference call
    surfaces it. It runs only when live targets are configured and live
    tests are actually selected, so ordinary mocked runs never touch it.
    """
    source = os.environ.get("KEYCALL_LIVE_SOURCE")
    if not source:
        return
    live_items = [item for item in session.items if "test_live" in item.nodeid]
    if not live_items:
        return

    from keycall._sources import load_targets
    from keycall._verify_core import run_verify

    targets, _ = load_targets(source)
    unavailable: list[str] = []
    for target in targets:
        result = run_verify(target, generate=True)
        reachable = result.generate_ok or result.outcome in ("services_probed", "no_text_models")
        if reachable:
            continue
        detail = result.attempts[-1].error_message if result.attempts else result.outcome
        unavailable.append(f"{target.display_name} ({result.provider}): {result.outcome}: {detail}")

    if unavailable:
        pytest.exit(
            "live pre-flight: these targets are unreachable, so the billable "
            "suite was not run (fix the provider, then re-run):\n  "
            + "\n  ".join(unavailable),
            returncode=1,
        )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_availability_cache():
    """Each test gets a clean process-global availability cache."""
    from keycall import _cache

    original = _cache.shared_cache
    _cache.shared_cache = _cache.AvailabilityCache()
    try:
        yield
    finally:
        _cache.shared_cache = original
