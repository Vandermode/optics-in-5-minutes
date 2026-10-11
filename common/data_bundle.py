"""Precomputed physics data as release assets, so machines without a GPU never recompute it.

    python common/data_bundle.py pack <episode_dir> [out_dir]   -> <out_dir>/<slug>_data.tar.gz   (make pack-data)
    python common/data_bundle.py fetch <episode_dir>            -> <episode_dir>/data/            (make fetch-data)

O5_DATA_URL (config.local.sh) says where the bundles live:
    https://…/download/<tag>        a public URL prefix (e.g. a public GitHub release's download path)
    file:///path/to/dir            a local or mounted folder
    gh:<owner>/<repo>@<tag>        a GitHub release, private ones included (uses the authenticated `gh` CLI)
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

import o5env  # noqa: F401  (loads config defaults for consistency with the other tools)


def url_configured():
    return bool(os.environ.get("O5_DATA_URL", "").strip())


def stamp(ep, how):
    with open(os.path.join(ep, "data", ".done"), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {how}\n")


def pack(ep, out_dir):
    slug = json.load(open(os.path.join(ep, "episode.json")))["slug"]
    data = os.path.join(ep, "data")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{slug}_data.tar.gz")
    with tarfile.open(dst, "w:gz", compresslevel=6) as tar:
        for f in sorted(os.listdir(data)):
            if not f.startswith("."):
                tar.add(os.path.join(data, f), arcname=f)
    print(f"{dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
    return dst


def fetch(ep, slug=None):
    slug = slug or json.load(open(os.path.join(ep, "episode.json")))["slug"]
    base = os.environ.get("O5_DATA_URL", "").strip()
    if not base:
        sys.exit("O5_DATA_URL is not set (see config.local.sh.example)")
    name = f"{slug}_data.tar.gz"
    tmp = tempfile.mkdtemp(prefix="o5data_")
    try:
        path = os.path.join(tmp, name)
        if base.startswith("gh:"):
            repo, _, tag = base[3:].partition("@")
            subprocess.run(["gh", "release", "download", tag, "-R", repo, "-p", name, "-D", tmp], check=True)
        else:
            print(f"download {base.rstrip('/')}/{name}", flush=True)
            urllib.request.urlretrieve(f"{base.rstrip('/')}/{name}", path)
        data = os.path.join(ep, "data")
        os.makedirs(data, exist_ok=True)
        with tarfile.open(path) as tar:
            tar.extractall(data, filter="data")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    cmd, ep = sys.argv[1], os.path.abspath(sys.argv[2])
    if cmd == "pack":
        pack(ep, sys.argv[3] if len(sys.argv) > 3 else os.path.join(o5env.ROOT, "dist", "data"))
    elif cmd == "fetch":
        fetch(ep)
        stamp(ep, "bundle")
    else:
        sys.exit(__doc__)
