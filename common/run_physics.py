"""Produce an episode's data/ by the fastest available route, then stamp data/.done (make's up-to-date marker).

    python common/run_physics.py <episode_dir> [--cpu]        (normally via: make physics E=E0N)

Routes, in order of preference:
  1. physics.py uses only numpy/scipy  -> run it here (seconds to ~2 min).
  2. physics.py uses JAX and a GPU is usable (nvidia-smi + CUDA-enabled jaxlib) -> run it on the least-busy GPU.
     A CUDA_VISIBLE_DEVICES you set yourself is respected.
  3. JAX physics, no usable GPU, O5_DATA_URL set -> download the precomputed bundle <slug>_data.tar.gz.
  4. Otherwise -> run on the CPU, with a warning (E02 takes about an hour; the others minutes).
--cpu forces route 1/4 (compute here, no download). Bundles are made with `make pack-data`.
"""
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time

import data_bundle


def pick_gpu():
    """Index of the least-busy NVIDIA GPU with free memory, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    best = None
    for line in out.strip().splitlines():
        try:
            i, util, used, total = (float(v) for v in line.split(","))
        except ValueError:
            continue
        if used / max(total, 1.0) > 0.85:
            continue
        key = (util, used)
        if best is None or key < best[0]:
            best = (key, int(i))
    return None if best is None else best[1]


def jax_has_cuda():
    for dist in ("jax-cuda12-plugin", "jax-cuda13-plugin", "jaxlib-cuda"):
        try:
            importlib.metadata.version(dist)
            return True
        except importlib.metadata.PackageNotFoundError:
            pass
    return False


def main():
    ep = os.path.abspath(sys.argv[1])
    force_cpu = "--cpu" in sys.argv
    slug = json.load(open(os.path.join(ep, "episode.json")))["slug"]
    uses_jax = "import jax" in open(os.path.join(ep, "physics.py")).read()
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    # one BLAS/OpenMP thread unless you say otherwise: the physics does many small products, where all-core BLAS
    # is slower (E02: 37 s and 3554 CPU-s with 112 threads vs 23 s and 20 CPU-s with one) and floods a shared server
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(v, "1")
    route = "cpu"
    if uses_jax and not force_cpu:
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu is None:
            g = pick_gpu()
            gpu = "" if g is None else str(g)
        if gpu and gpu != "-1" and jax_has_cuda():
            env["CUDA_VISIBLE_DEVICES"] = gpu
            route = f"gpu {gpu}"
        elif data_bundle.url_configured():
            data_bundle.fetch(ep, slug)
            data_bundle.stamp(ep, "bundle")
            print(f"[{slug}] data: downloaded the precomputed bundle")
            return
        else:
            print(f"[{slug}] WARNING: no usable GPU and no O5_DATA_URL: computing on the CPU (E02 takes about an hour).\n"
                  f"          Faster: install CUDA JAX (setup_env.sh does it when it sees a GPU), or set O5_DATA_URL "
                  f"to the release holding the data bundles.", flush=True)
    if not uses_jax:
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    t = time.time()
    print(f"[{slug}] data: physics.py on {route}", flush=True)
    r = subprocess.run(["nice", "-n", "5", sys.executable, "physics.py"], cwd=ep, env=env)
    if r.returncode:
        sys.exit(r.returncode)
    data_bundle.stamp(ep, f"physics.py on {route}, {time.time() - t:.0f} s")
    print(f"[{slug}] data: done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
