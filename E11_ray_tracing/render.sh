#!/usr/bin/env bash
# Render only this episode; cap workers while another episode is in production.
cd "$(dirname "$0")" && WORKERS=${WORKERS:-3} exec ../common/render.sh . "$@"
