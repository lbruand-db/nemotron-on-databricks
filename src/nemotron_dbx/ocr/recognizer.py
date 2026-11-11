"""Recognizer-only fine-tuning of nemotron-ocr-v1 (SPEC §6.4.1).

Mirrors `nemotron_ocr.inference.pipeline.NemotronOCR._process_tensor` exactly, except that the word quads
come from the ground truth (optionally jittered) instead of the detector + NMS:

    image [0,1] -> pad_to_square(bottom_right, 1.0) -> interpolate_and_pad(1024, PAD_COLOR)
    -> frozen FOTSDetector -> det_feature_3
    -> QuadRectify(8, 32)(quads in 1024-px coords) -> IndirectGridSample -> [N, C, 8, 32]
    -> TransformerRecognizer (trainable) -> logits [N, 32, 858]
    -> masked per-position cross-entropy vs [c_1..c_L, EOS]

The detector and relational model stay frozen and no gradient flows through the custom CUDA ops.
Requires the `nemotron_ocr` package (built by `nemotron_dbx.ocr.build`) and a CUDA device.
"""

from __future__ import annotations

import io
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import IGNORE, SEQ_LEN, Charset
from .cord import OCRRecord
from .metrics import bag_of_words, recognition_metrics

INFER_LENGTH = 1024


def decode_rgb(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


@dataclass
class OCRTrainConfig:
    epochs: int = 10
    images_per_step: int = 4
    max_regions_per_step: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    quad_jitter: float = 0.06  # vertex noise, as a fraction of the word height (detector quads are noisy)
    brightness: float = 0.15
    contrast: float = 0.15
    log_every: int = 20
    seed: int = 0
    max_minutes: float | None = None


class OCRStack:
    """NVIDIA's detector (frozen) + recognizer (trainable) + rectification ops."""

    def __init__(self, checkpoints: str | Path, device: str = "cuda") -> None:
        from nemotron_ocr.inference.models.detector.fots_detector import FOTSDetector
        from nemotron_ocr.inference.models.recognizer import TransformerRecognizer
        from nemotron_ocr.inference.post_processing.indirect_grid_sample import IndirectGridSample
        from nemotron_ocr.inference.post_processing.quad_rectify import QuadRectify

        ckpt = Path(checkpoints)
        self.device = torch.device(device)
        self.charset = Charset.from_file(ckpt / "charset.txt")
        self.detector = FOTSDetector(coordinate_mode="RBOX", backbone="regnet_y_8gf", verbose=False)
        self.detector.load_state_dict(torch.load(ckpt / "detector.pth", weights_only=True), strict=True)
        self.detector.to(self.device).eval()
        for p in self.detector.parameters():
            p.requires_grad_(False)
        self.recognizer = TransformerRecognizer(
            nic=self.detector.num_features[-1], num_tokens=self.charset.num_tokens, max_width=SEQ_LEN
        )
        self.recognizer.load_state_dict(torch.load(ckpt / "recognizer.pth", weights_only=True), strict=True)
        self.recognizer.to(self.device).eval()
        self.rectifier = QuadRectify(8, 32)
        self.sampler = IndirectGridSample()

    # -- preprocessing (identical to the pipeline) ---------------------------------------------------
    def preprocess(self, rgb: np.ndarray, brightness: float = 0.0, contrast: float = 0.0):
        from nemotron_ocr.inference.pipeline import PAD_COLOR
        from nemotron_ocr.inference.pre_processing import interpolate_and_pad, pad_to_square

        img = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device).float() / 255.0
        if brightness or contrast:
            mean = img.mean()
            img = ((img - mean) * (1 + contrast) + mean) * (1 + brightness)
            img = img.clamp(0, 1)
        img = img.half()
        padded_length = max(img.shape[1:])
        padded = interpolate_and_pad(
            pad_to_square(img, padded_length, how="bottom_right").unsqueeze(0), PAD_COLOR.to(self.device), INFER_LENGTH
        )
        return padded, INFER_LENGTH / padded_length

    @torch.no_grad()
    def features(self, padded: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=self.device.type == "cuda"):
            _, _, _, feat = self.detector(padded)
        return feat

    @torch.no_grad()
    def rectify(self, feats: torch.Tensor, quads: torch.Tensor, region_counts: torch.Tensor) -> torch.Tensor:
        """quads: [N, 4, 2] in 1024-px coordinates of the padded input; region_counts: [B] int64."""
        from nemotron_ocr_cpp import region_counts_to_indices

        grid = self.rectifier(quads.float(), INFER_LENGTH, INFER_LENGTH)
        idx = region_counts_to_indices(region_counts, quads.shape[0])
        return self.sampler(feats.float(), grid.float(), idx)

    def logits(self, rect: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out, _ = self.recognizer(rect)
        return out.float()

    def save_recognizer(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.recognizer.state_dict().items()}, path)


def jitter_quads(quads: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    if amount <= 0 or len(quads) == 0:
        return quads
    heights = np.linalg.norm(quads[:, 3] - quads[:, 0], axis=1).clip(min=1.0)
    noise = rng.normal(0.0, 1.0, size=quads.shape) * (amount * heights)[:, None, None]
    return (quads + noise).astype(np.float32)


def _freeze_batchnorm(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def _lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def train_recognizer(
    stack: OCRStack,
    records: list[OCRRecord],
    cfg: OCRTrainConfig,
    log_fn: Callable[[dict[str, float], int], None] | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    images = [decode_rgb(r.image) for r in records]
    usable = [i for i, r in enumerate(records) if any(stack.charset.trainable(t) for t in r.texts)]
    steps_per_epoch = max(1, len(usable) // cfg.images_per_step)
    total = cfg.epochs * steps_per_epoch
    params = [p for p in stack.recognizer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    t0 = time.monotonic()
    step = 0
    stack.recognizer.train()
    _freeze_batchnorm(stack.recognizer)
    for _epoch in range(cfg.epochs):
        order = rng.permutation(usable)
        for s in range(steps_per_epoch):
            batch_idx = order[s * cfg.images_per_step : (s + 1) * cfg.images_per_step]
            padded, quads, texts, counts = [], [], [], []
            for i in batch_idx:
                rec = records[i]
                keep = [k for k, t in enumerate(rec.texts) if stack.charset.trainable(t)]
                b = rng.uniform(-cfg.brightness, cfg.brightness) if cfg.brightness else 0.0
                c = rng.uniform(-cfg.contrast, cfg.contrast) if cfg.contrast else 0.0
                img, scale = stack.preprocess(images[i], b, c)
                q = jitter_quads(rec.quads[keep], cfg.quad_jitter, rng) * scale
                padded.append(img)
                quads.append(q)
                texts.extend(rec.texts[k] for k in keep)
                counts.append(len(keep))
            q_all = np.concatenate(quads)
            if len(q_all) > cfg.max_regions_per_step:  # subsample regions, keeping per-image counts consistent
                sel = np.sort(rng.choice(len(q_all), cfg.max_regions_per_step, replace=False))
                owner = np.repeat(np.arange(len(counts)), counts)[sel]
                counts = np.bincount(owner, minlength=len(counts)).tolist()
                q_all = q_all[sel]
                texts = [texts[j] for j in sel]
            feats = stack.features(torch.cat(padded))
            rect = stack.rectify(
                feats,
                torch.from_numpy(q_all).to(stack.device),
                torch.tensor(counts, dtype=torch.int64, device=stack.device),
            )
            targets = torch.from_numpy(stack.charset.encode(texts)).to(stack.device)
            logits = stack.logits(rect)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE)
            f = _lr_factor(step, total, cfg.warmup_steps, cfg.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * f
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            step += 1
            if log_fn and (step == 1 or step % cfg.log_every == 0):
                with torch.no_grad():
                    valid = targets != IGNORE
                    tok_acc = ((logits.argmax(-1) == targets) & valid).sum() / valid.sum().clamp(min=1)
                log_fn({
                    "train/loss": loss.item(), "train/token_acc": tok_acc.item(),
                    "train/grad_norm": float(grad_norm), "train/lr": cfg.lr * f,
                    "train/regions": float(len(texts)), "train/elapsed_min": (time.monotonic() - t0) / 60,
                }, step)
            if cfg.max_minutes and (time.monotonic() - t0) / 60 > cfg.max_minutes:
                stack.recognizer.eval()
                return {"steps": step, "stopped_early": 1.0}
    stack.recognizer.eval()
    return {"steps": step, "stopped_early": 0.0}


@torch.no_grad()
def eval_gt_quads(stack: OCRStack, records: list[OCRRecord]) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """Recognition on ground-truth quads (isolates the recognizer). All GT words count, incl. OOV ones."""
    stack.recognizer.eval()
    preds, gts = [], []
    oov_chars = total_chars = 0
    for rec in records:
        if len(rec.texts) == 0:
            continue
        img, scale = stack.preprocess(decode_rgb(rec.image))
        feats = stack.features(img)
        rect = stack.rectify(
            feats,
            torch.from_numpy(rec.quads * scale).to(stack.device),
            torch.tensor([len(rec.texts)], dtype=torch.int64, device=stack.device),
        )
        ids = stack.logits(rect).argmax(-1).cpu().numpy()
        preds.extend(stack.charset.decode(row) for row in ids)
        gts.extend(rec.texts)
        for t in rec.texts:
            total_chars += len(t)
            oov_chars += len(stack.charset.oov_chars(t))
    m = recognition_metrics(preds, gts)
    m["oov_char_rate"] = oov_chars / max(1, total_chars)
    return m, list(zip(gts, preds, strict=True))


def eval_end_to_end(checkpoints: str | Path, records: list[OCRRecord]) -> dict[str, float]:
    """Full NVIDIA pipeline (detector + NMS + recognizer + relational) at word level -> bag-of-words F1."""
    from nemotron_ocr.inference.pipeline import NemotronOCR

    ocr = NemotronOCR(model_dir=str(checkpoints))
    pred_words, gt_words = [], []
    t0 = time.monotonic()
    for rec in records:
        out = ocr(decode_rgb(rec.image), merge_level="word")
        pred_words.append([p["text"] for p in out if isinstance(p.get("text"), str)])
        gt_words.append(rec.texts)
    m = bag_of_words(pred_words, gt_words)
    m["sec_per_image"] = (time.monotonic() - t0) / max(1, len(records))
    del ocr
    torch.cuda.empty_cache()
    return m


def make_checkpoint_dir(base: str | Path, recognizer: str | Path, dest: str | Path) -> Path:
    """Checkpoint dir for `NemotronOCR(model_dir=...)`: base detector/relational/charset + new recognizer."""
    import shutil

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("detector.pth", "relational.pth", "charset.txt"):
        shutil.copy2(Path(base) / name, dest / name)
    shutil.copy2(recognizer, dest / "recognizer.pth")
    return dest
