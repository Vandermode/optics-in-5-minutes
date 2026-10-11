#!/usr/bin/env bash
# Full render of this episode (see ../common/render.sh):  ./render.sh [--skip-audio]   env WORKERS (default 6)
cd "$(dirname "$0")" && WORKERS=${WORKERS:-6} exec ../common/render.sh . "$@"
