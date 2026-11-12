"""Build-once wheel cache for CUDA extensions (SPEC §11.2).

AIR's `databricks_ai_v6` has nvcc 13 but no prebuilt wheels for some CUDA extensions (nemotron-ocr's C++
extension, causal-conv1d). Build them once with `pip wheel --no-deps --no-build-isolation`, cache the
wheel on a Volume keyed by package/revision/torch version, and install from the cache afterwards.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def torch_tag() -> str:
    import torch

    return "torch" + torch.__version__.replace("+", "_")


def cached_wheel(cache_dir: Path, prefix: str = "") -> Path | None:
    wheels = sorted(cache_dir.glob(f"{prefix}*.whl"))
    return wheels[-1] if wheels else None


def ensure_wheel(
    source_dir: str | Path, cache_dir: str | Path, env: dict[str, str] | None = None, prefix: str = ""
) -> Path:
    """Return a cached wheel for `source_dir` (a directory or a pip requirement such as `pkg==1.2`), building it
    into `cache_dir` if missing. `prefix` (e.g. `causal_conv1d-1.6.2.post1-`) restricts cache hits to one version."""
    cache_dir = Path(cache_dir)
    hit = cached_wheel(cache_dir, prefix)
    if hit:
        print(f"[wheels] cache hit: {hit}", flush=True)
        return hit
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wheel-build-") as out:
        t0 = time.monotonic()
        cmd = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", out, str(source_dir)]
        print(f"[wheels] building {source_dir}: {' '.join(cmd)}", flush=True)
        import os

        subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})
        built = sorted(Path(out).glob("*.whl"))
        if not built:
            raise RuntimeError(f"pip wheel produced no wheel for {source_dir}")
        tmp = cache_dir / (built[-1].name + ".tmp")
        shutil.copy2(built[-1], tmp)
        tmp.replace(cache_dir / built[-1].name)
        print(f"[wheels] built {built[-1].name} in {time.monotonic() - t0:.0f}s", flush=True)
    return cache_dir / built[-1].name


def pip_install(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)
