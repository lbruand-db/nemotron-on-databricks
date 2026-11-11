"""Fetch the pinned nemotron-ocr-v1 repo and install its package (C++/CUDA extension via the wheel cache)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from ..common.hf import NEMOTRON_OCR_V1
from ..common.wheels import ensure_wheel, pip_install, torch_tag

# Architectures for the serving GPUs: T4 (7.5), A10G (8.6), L40S (8.9), H100 (9.0) + PTX.
DEFAULT_ARCHS = "7.5;8.6;8.9;9.0+PTX"
CHECKPOINTS = ("detector.pth", "recognizer.pth", "relational.pth", "charset.txt")


@dataclass
class OCRInstall:
    repo_dir: Path  # snapshot root (contains checkpoints/ and nemotron-ocr/)
    wheel: Path

    @property
    def checkpoints(self) -> Path:
        return self.repo_dir / "checkpoints"


def snapshot_repo(dest: str | Path) -> Path:
    return NEMOTRON_OCR_V1.snapshot(
        dest, allow_patterns=["checkpoints/*", "nemotron-ocr/**", "LICENSE", "THIRD_PARTY_NOTICES.md"]
    )


def ensure_installed(wheel_cache_root: str | Path, dest: str | Path, archs: str = DEFAULT_ARCHS) -> OCRInstall:
    """Snapshot the repo, build (or reuse) the wheel, install it with --no-deps (+ shapely if missing)."""
    repo = snapshot_repo(dest)
    if importlib.util.find_spec("hatchling") is None:  # build backend, needed with --no-build-isolation
        pip_install("hatchling")
    cache = Path(wheel_cache_root) / "nemotron-ocr" / f"{NEMOTRON_OCR_V1.revision[:8]}-{torch_tag()}"
    wheel = ensure_wheel(repo / "nemotron-ocr", cache, env={"TORCH_CUDA_ARCH_LIST": archs, "BUILD_CPP_FORCE": "1"})
    if importlib.util.find_spec("nemotron_ocr") is None:
        pip_install("--no-deps", str(wheel))
    if importlib.util.find_spec("shapely") is None:
        pip_install("shapely==2.1.2")
    importlib.invalidate_caches()
    return OCRInstall(repo_dir=repo, wheel=wheel)
