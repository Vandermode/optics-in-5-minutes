#!/usr/bin/env bash
# Low-resolution chapter previews, one local worker while E12 is also in production.
cd "$(dirname "$0")" && PAR=${PAR:-1} exec ../common/preview.sh . "$@"
