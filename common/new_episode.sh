#!/usr/bin/env bash
# Scaffold a new episode:  common/new_episode.sh <FolderName e.g. E02_lens_fourier> "<Title>"
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT=$O5_ROOT
F=$1; TITLE=$2
D="$ROOT/$F"
SLUG=$(echo "$F" | tr 'A-Z' 'a-z')
mkdir -p "$D/data" "$D/.cache/Tex"
[ -f "$D/episode.json" ] || cat > "$D/episode.json" <<J
{"slug": "$SLUG", "title": "$TITLE", "chapters": ["Ch0Hook"]}
J
cat > "$D/render.sh" <<'R'
#!/usr/bin/env bash
# Full render of this episode (see ../common/render.sh):  ./render.sh [--skip-audio]   env WORKERS (default 6)
cd "$(dirname "$0")" && WORKERS=${WORKERS:-6} exec ../common/render.sh . "$@"
R
cat > "$D/preview.sh" <<'R'
#!/usr/bin/env bash
# Low-res chapter previews (see ../common/preview.sh):  ./preview.sh [Chapter ...]
cd "$(dirname "$0")" && exec ../common/preview.sh . "$@"
R
chmod +x "$D/render.sh" "$D/preview.sh"
[ -f "$D/scenes.py" ] || cat > "$D/scenes.py" <<'S'
"""<Title> -- chapter scenes (Optics in 5 Minutes).  Render: ./render.sh ; preview: ./preview.sh Ch0Hook"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import o5common  # noqa: E402

o5common.load_timing(HERE)
from o5common import *  # noqa: E402,F401,F403

S
[ -f "$D/physics.py" ] || cat > "$D/physics.py" <<'P'
"""Precompute every physical field shown in this episode -> data/*.npz (run once; scenes only load).
    python physics.py          (or: make physics E=<episode>)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
TAU = 2 * np.pi
P
echo "scaffolded $D (slug $SLUG)"
