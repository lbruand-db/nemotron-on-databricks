"""Load NVIDIA's Nemotron YOLOX detectors from a pinned HF snapshot, with optional fine-tuned weights.

The NVIDIA repos are small Python packages (`<package>/{model.py, <exp>.py, yolox/...}`); `define_model`
downloads weights from the Hub at call time. We bypass it so training, evaluation and serving all load
from a local snapshot (no Hub access at serving time) and can swap in fine-tuned weights.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..common.hf import PAGE_ELEMENTS_V3, HFSource


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    source: HFSource
    package: str  # top-level python package inside the HF repo
    exp_module: str  # module defining `Exp`
    weights_file: str  # path of the pretrained weights inside the repo

    @property
    def code_patterns(self) -> list[str]:
        return [f"{self.package}/**/*.py", f"{self.package}/*.py", "LICENSE", "THIRD_PARTY_NOTICES.md"]


PAGE_ELEMENTS = DetectorSpec(
    name="page_elements_v3",
    source=PAGE_ELEMENTS_V3,
    package="nemotron_page_elements_v3",
    exp_module="page_element_v3",
    weights_file="nemotron_page_elements_v3/weights.pth",
)

SPECS = {s.name: s for s in (PAGE_ELEMENTS,)}


def snapshot(spec: DetectorSpec, dest: str | Path, with_weights: bool = True) -> Path:
    patterns = spec.code_patterns + ([spec.weights_file] if with_weights else [])
    return spec.source.snapshot(dest, allow_patterns=patterns)


def _import(repo_dir: Path, module: str):
    repo = str(repo_dir.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return importlib.import_module(module)


def load_exp(spec: DetectorSpec, repo_dir: str | Path, device: str | None = None):
    exp = _import(Path(repo_dir), f"{spec.package}.{spec.exp_module}").Exp()
    if device is not None:
        exp.device = device
    return exp


def load_yolox(spec: DetectorSpec, repo_dir: str | Path, weights: str | Path | None = None, device=None):
    """Return (bare YOLOX nn.Module with weights loaded strictly, Exp config)."""
    repo_dir = Path(repo_dir)
    exp = load_exp(spec, repo_dir, device)
    model = exp.get_model()
    weights = Path(weights) if weights else repo_dir / spec.weights_file
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    return model, exp


def wrap(spec: DetectorSpec, repo_dir: str | Path, model: torch.nn.Module, exp):
    """NVIDIA `YoloXWrapper` (NMS, rescale, normalized boxes) around `model`, in eval mode."""
    wrapper_cls = _import(Path(repo_dir), f"{spec.package}.model").YoloXWrapper
    return wrapper_cls(model, exp).eval().to(exp.device)


def save_weights(model: torch.nn.Module, path: str | Path) -> None:
    """Same format as NVIDIA's `weights.pth` ({"model": state_dict}) so `load_yolox` reads both."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


@torch.inference_mode()
def predict(wrapper, images: list[np.ndarray], batch_size: int = 8) -> list[dict[str, np.ndarray]]:
    """Run the NVIDIA wrapper on RGB uint8 HxWx3 arrays.

    Returns per image: boxes [N,4] normalized xyxy, labels [N] int, scores [N] (no per-class thresholds;
    the wrapper already applied NMS with `conf_thresh`/`iou_thresh`).
    """
    out: list[dict[str, np.ndarray]] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        x = torch.stack([wrapper.preprocess(img) for img in chunk])
        sizes = np.array([img.shape[:2] for img in chunk])
        for p in wrapper(x, sizes):
            out.append({
                "boxes": p["boxes"].float().cpu().numpy().reshape(-1, 4),
                "labels": p["labels"].cpu().numpy().astype(np.int64),
                "scores": p["scores"].float().cpu().numpy(),
            })
    return out


def apply_class_thresholds(
    pred: dict[str, np.ndarray], labels: list[str], thresholds: dict[str, float]
) -> dict[str, np.ndarray]:
    """NVIDIA's `postprocess_preds_page_element`, on numpy predictions."""
    if len(pred["labels"]) == 0:
        return pred
    thr = np.array([thresholds.get(labels[int(c)], 0.0) for c in pred["labels"]])
    keep = pred["scores"] > thr
    return {k: v[keep] for k, v in pred.items()}
