"""Training-only dependencies for Nano VL on stock `databricks_ai_v6` (SPEC §6.5.1, §11.3 patches 1-4).

vLLM (eval and serving) needs none of this: it has its own Mamba kernels and Nano VL implementation.

- Patch 1: the HF remote code imports `mamba_ssm.ops.triton.*` unconditionally. Install mamba-ssm without
  its CUDA build (Triton ops only) and without deps (they would change torch/transformers).
- Patch 2: `mamba_ssm/__init__.py` eagerly imports the Mamba-1 CUDA extension, which that build lacks.
  Replace it with a version-only stub.
- Patch 3: the fused Mamba-2 path needs `causal-conv1d`'s CUDA extension (the torch fallback OOMs). There's
  no prebuilt wheel for torch 2.11, so build it once (~11 min) and cache the wheel on a Volume.
- Patch 4: open_clip (RADIO) needs ftfy and regex. (zss is for the eval metrics.)
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from ..common.wheels import ensure_wheel, pip_install

MAMBA_SSM = "mamba-ssm==2.3.2.post1"
CAUSAL_CONV1D = "causal-conv1d==1.6.2.post1"
EXTRAS = ("einops==0.8.2", "timm==1.0.31", "open_clip_torch==4.2.1", "ftfy==6.3.1", "zss==1.2.0")


def _version(req: str) -> str:
    return req.split("==", 1)[1]


def causal_conv1d_cache(wheel_cache_root: str | Path) -> Path:
    import torch

    # Same layout as the M0 spike, so its cached wheel is reused: <root>/causal_conv1d/<torch version>/
    return Path(wheel_cache_root) / "causal_conv1d" / torch.__version__.replace("+", "_")


def stub_mamba_init() -> Path:
    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("mamba_ssm is not installed")
    init = Path(list(spec.submodule_search_locations)[0]) / "__init__.py"
    stub = f"__version__ = '{_version(MAMBA_SSM)}'\n"
    if init.read_text() != stub:
        init.write_text(stub)
    return init


def ensure_training_deps(wheel_cache_root: str | Path) -> dict[str, str]:
    """Install the Nano VL training deps (idempotent). Returns what was used, for logging."""
    t0 = time.monotonic()
    missing = [r for r in EXTRAS if importlib.util.find_spec(r.split("==")[0].replace("_torch", "")) is None]
    if missing:
        pip_install(*missing)
    if importlib.util.find_spec("regex") is None:
        pip_install("regex")
    if importlib.util.find_spec("mamba_ssm") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", MAMBA_SSM],
            check=True, env={**os.environ, "MAMBA_SKIP_CUDA_BUILD": "TRUE"},
        )
    stub_mamba_init()
    wheel = ensure_wheel(
        CAUSAL_CONV1D, causal_conv1d_cache(wheel_cache_root),
        env={"CAUSAL_CONV1D_FORCE_BUILD": "TRUE", "MAX_JOBS": os.environ.get("MAX_JOBS", "16")},
        prefix=f"causal_conv1d-{_version(CAUSAL_CONV1D)}-",
    )
    if importlib.util.find_spec("causal_conv1d") is None:
        pip_install("--no-deps", str(wheel))
    importlib.invalidate_caches()
    # Fail here rather than deep inside the model's forward.
    subprocess.run([sys.executable, "-c", (
        "import torch, causal_conv1d, causal_conv1d_cuda, open_clip, timm;"  # torch first: the ext links libc10
        "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn;"
        "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined"
    )], check=True)
    info = {"causal_conv1d_wheel": str(wheel), "setup_s": f"{time.monotonic() - t0:.0f}"}
    print(f"[env] Nano VL training deps ready: {info}", flush=True)
    return info
