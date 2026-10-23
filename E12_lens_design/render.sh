#!/usr/bin/env bash
# Full render of this episode (see ../common/render.sh):  ./render.sh [--skip-audio]   env WORKERS (default 3)
cd "$(dirname "$0")" && WORKERS=${WORKERS:-3} exec ../common/render.sh . "$@"
