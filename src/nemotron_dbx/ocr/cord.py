"""CORD-v2 receipts -> word-level OCR records (image bytes, quads, texts) (SPEC §6.4.2)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..common.hf import CORD_V2


@dataclass
class OCRRecord:
    image: bytes
    quads: np.ndarray  # [N, 4, 2] float32, pixel coords, vertex order TL, TR, BR, BL
    texts: list[str]
    image_id: str = ""
    size: tuple[int, int] = (0, 0)  # (height, width)
    extra: dict = field(default_factory=dict)


def parse_ground_truth(gt: dict, image_hw: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Word quads + texts from a CORD `ground_truth` dict, rescaled if `meta.image_size` differs."""
    h, w = image_hw
    meta = gt.get("meta", {}).get("image_size") or {}
    sx = w / meta["width"] if meta.get("width") else 1.0
    sy = h / meta["height"] if meta.get("height") else 1.0
    quads, texts = [], []
    for line in gt.get("valid_line", []):
        for word in line.get("words", []):
            text = (word.get("text") or "").strip()
            q = word.get("quad")
            if not text or not q:
                continue
            pts = [[q[f"x{i}"] * sx, q[f"y{i}"] * sy] for i in (1, 2, 3, 4)]
            quads.append(pts)
            texts.append(text)
    arr = np.asarray(quads, dtype=np.float32).reshape(-1, 4, 2)
    arr[..., 0] = arr[..., 0].clip(0, w)
    arr[..., 1] = arr[..., 1].clip(0, h)
    return arr, texts


def load_split(split: str, max_items: int | None = None, cache_dir: str | Path | None = None) -> list[OCRRecord]:
    """Load one CORD-v2 split ("train" | "validation" | "test") from the pinned parquet files."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from PIL import Image

    files = HfApi().list_repo_files(CORD_V2.repo_id, repo_type="dataset", revision=CORD_V2.revision)
    shards = sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    records: list[OCRRecord] = []
    for shard in shards:
        local = CORD_V2.file(shard, local_dir=cache_dir)
        for row in pq.read_table(local, columns=["image", "ground_truth"]).to_pylist():
            data = row["image"]["bytes"]
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
            gt = json.loads(row["ground_truth"])
            quads, texts = parse_ground_truth(gt, (h, w))
            image_id = str(gt.get("meta", {}).get("image_id", len(records)))
            records.append(OCRRecord(data, quads, texts, f"{split}-{image_id}", (h, w)))
            if max_items and len(records) >= max_items:
                return records
    return records
