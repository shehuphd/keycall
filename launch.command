#!/bin/sh
# macOS double-click launcher: delegates to launch.sh in its own directory.
cd "$(dirname "$0")" || exit 1
exec ./launch.sh "$@"
