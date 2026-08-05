import pytest


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
