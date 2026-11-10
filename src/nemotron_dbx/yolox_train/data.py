"""Detection dataset for Nemotron YOLOX models.

Preprocessing matches NVIDIA's inference wrapper exactly (`resize_pad`): RGB, float 0..255, resize the
longest side to fit `size` with bilinear interpolation, pad bottom/right with 114. Boxes are scaled by
the same ratio, so training and serving see identical geometry.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps

PAD_VALUE = 114.0


@dataclass
class DetRecord:
    """One image with absolute-pixel `xyxy` boxes and integer labels (model class indices)."""

    image: bytes
    boxes: np.ndarray  # [N, 4] float, xyxy in image pixels
    labels: np.ndarray  # [N] int
    image_id: str = ""


@dataclass
class AugmentConfig:
    scale_min: float = 0.75  # content scale jitter inside the canvas (1.0 = no jitter)
    brightness: float = 0.15
    contrast: float = 0.15
    grayscale_p: float = 0.1
    jpeg_p: float = 0.3
    jpeg_quality: tuple[int, int] = (60, 95)


def letterbox(img: torch.Tensor, size: tuple[int, int], scale: float | None = None):
    """`resize_pad` from the NVIDIA repos, with an optional explicit `scale`. img: [3, H, W] float."""
    _, h, w = img.shape
    base = min(size[0] / h, size[1] / w)
    scale = base if scale is None else min(scale, base)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    out = F.interpolate(img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)
    out = torch.clamp(out, 0, 255)
    out = F.pad(out, (0, size[1] - nw, 0, size[0] - nh), value=PAD_VALUE)
    return out, scale


def decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _photometric(img: Image.Image, cfg: AugmentConfig, rng: random.Random) -> Image.Image:
    if cfg.brightness:
        img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-cfg.brightness, cfg.brightness))
    if cfg.contrast:
        img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-cfg.contrast, cfg.contrast))
    if rng.random() < cfg.grayscale_p:
        img = ImageOps.grayscale(img).convert("RGB")
    if rng.random() < cfg.jpeg_p:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(*cfg.jpeg_quality))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


class DetectionDataset(torch.utils.data.Dataset):
    """Returns (image [3,S,S] float 0..255, labels [max_labels, 5] = (cls, cx, cy, w, h) px, info)."""

    def __init__(
        self,
        records: list[DetRecord],
        size: tuple[int, int] = (1024, 1024),
        augment: AugmentConfig | None = None,
        max_labels: int = 300,
        min_box_px: float = 2.0,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.size = size
        self.augment = augment
        self.max_labels = max_labels
        self.min_box_px = min_box_px
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        rng = random.Random(hash((self.seed, self._epoch, idx)))
        img = decode_image(rec.image)
        w0, h0 = img.size
        scale = None
        if self.augment is not None:
            img = _photometric(img, self.augment, rng)
            base = min(self.size[0] / h0, self.size[1] / w0)
            scale = base * rng.uniform(self.augment.scale_min, 1.0)
        tensor = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float()
        tensor, used_scale = letterbox(tensor, self.size, scale)
        labels = encode_targets(rec.boxes, rec.labels, used_scale, self.size, self.max_labels, self.min_box_px)
        return tensor, labels, {"image_id": rec.image_id, "orig_hw": (h0, w0), "scale": used_scale}


def encode_targets(
    boxes: np.ndarray,
    labels: np.ndarray,
    scale: float,
    size: tuple[int, int],
    max_labels: int,
    min_box_px: float = 2.0,
) -> torch.Tensor:
    """xyxy image px -> padded [max_labels, 5] (cls, cx, cy, w, h) in letterboxed px."""
    out = torch.zeros((max_labels, 5), dtype=torch.float32)
    if len(boxes) == 0:
        return out
    b = np.asarray(boxes, dtype=np.float32) * scale
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, size[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, size[0])
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    keep = (w >= min_box_px) & (h >= min_box_px)
    b, w, h, lab = b[keep], w[keep], h[keep], np.asarray(labels)[keep]
    n = min(len(b), max_labels)
    if n == 0:
        return out
    out[:n, 0] = torch.from_numpy(lab[:n].astype(np.float32))
    out[:n, 1] = torch.from_numpy(b[:n, 0] + w[:n] / 2)
    out[:n, 2] = torch.from_numpy(b[:n, 1] + h[:n] / 2)
    out[:n, 3] = torch.from_numpy(w[:n])
    out[:n, 4] = torch.from_numpy(h[:n])
    return out


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    imgs, labels, infos = zip(*batch, strict=True)
    return torch.stack(imgs), torch.stack(labels), list(infos)
