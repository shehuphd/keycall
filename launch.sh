#!/bin/sh
# KeyCall viewer launcher (Linux/macOS). Path-safe and interpreter-safe:
# works from a fresh clone in any directory, on any account.
set -e
cd "$(dirname "$0")" || exit 1

# Resolve the interpreter explicitly — never trust a bare `python`.
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "error: python3 not found. Install Python 3.10+ and retry." >&2
    exit 1
fi

# Validate an existing venv by running its own interpreter, not by
# checking the directory exists (a synced/stale venv fails downstream).
if [ -x .venv/bin/python ] && .venv/bin/python --version >/dev/null 2>&1; then
    :
else
    echo "creating virtual environment…"
    rm -rf .venv
    "$PY" -m venv .venv
    .venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
fi

# Install/refresh keycall into the venv's own interpreter (-m pip, never bare pip).
.venv/bin/python -m pip install -q -e . || {
    echo "error: install failed" >&2
    exit 1
}

# Find a key source: explicit argument wins, then conventional locations.
SOURCE="$1"
if [ -z "$SOURCE" ]; then
    for candidate in keycall-keys.toml keycall-test-keys.toml project/keycall-test-keys.toml; do
        if [ -f "$candidate" ]; then
            SOURCE="$candidate"
            break
        fi
    done
fi
if [ -z "$SOURCE" ]; then
    # No key file found — the viewer opens with a prompt to load one.
    exec .venv/bin/python -m keycall._cli view
fi

exec .venv/bin/python -m keycall._cli view --source "$SOURCE"
