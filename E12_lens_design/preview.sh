#!/usr/bin/env bash
# Low-res chapter previews (see ../common/preview.sh):  ./preview.sh [Chapter ...]
cd "$(dirname "$0")" && PAR=${PAR:-1} exec ../common/preview.sh . "$@"
