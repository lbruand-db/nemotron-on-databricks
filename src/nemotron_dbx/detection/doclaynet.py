"""DocLayNet v1.2 -> page-elements-v3 training data (SPEC §6.1).

The HF dataset is ~40 GB of parquet because every row embeds the source PDF. We download shards one by
one, read only the needed columns, filter by document category, map the 11 DocLayNet classes to the 6
page-elements classes, and write a compact parquet (image bytes + xyxy boxes + labels) that is cached on
the Volume and reused by later runs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..common.hf import DOCLAYNET_V1_2, HFSource
from ..common.paths import scratch_dir
from ..yolox_train.data import DetRecord

# DocLayNet category ids (1-based, see dataset card).
DOCLAYNET_CLASSES = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item", 5: "Page-footer", 6: "Page-header",
    7: "Picture", 8: "Section-header", 9: "Table", 10: "Text", 11: "Title",
}
PICTURE_ID = 7

# page-elements-v3 label order, from nemotron_page_elements_v3/page_element_v3.py (Exp.labels).
PAGE_ELEMENTS_LABELS = ["table", "chart", "title", "infographic", "text", "header_footer"]

# DocLayNet -> page-elements. `Picture` is ambiguous (chart vs infographic vs photo): None = unmapped.
DOCLAYNET_TO_PAGE_ELEMENTS: dict[int, str | None] = {
    1: "text",  # Caption
    2: "text",  # Footnote
    3: "text",  # Formula
    4: "text",  # List-item
    5: "header_footer",  # Page-footer
    6: "header_footer",  # Page-header
    7: None,  # Picture
    8: "title",  # Section-header
    9: "table",  # Table
    10: "text",  # Text
    11: "title",  # Title
}


@dataclass
class SubsetConfig:
    doc_categories: list[str] = field(default_factory=lambda: ["financial_reports", "manuals"])
    max_pages: dict[str, int] = field(default_factory=lambda: {"train": 2000, "validation": 200, "test": 300})
    drop_picture_pages: bool = True  # SPEC OQ-2 default

    def key(self) -> str:
        cats = "+".join(sorted(self.doc_categories))
        sizes = "-".join(f"{s}{n}" for s, n in sorted(self.max_pages.items()))
        return f"doclaynet-v1.2@{DOCLAYNET_V1_2.revision[:8]}_{cats}_{sizes}_pic{int(self.drop_picture_pages)}"


def map_page(
    bboxes_xywh: Sequence[Sequence[float]],
    category_ids: Sequence[int],
    width: float,
    height: float,
    drop_picture_pages: bool,
    labels: Sequence[str] = PAGE_ELEMENTS_LABELS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map one DocLayNet page. Returns (xyxy boxes clipped to the page, label indices) or None to skip."""
    if drop_picture_pages and PICTURE_ID in category_ids:
        return None
    index = {name: i for i, name in enumerate(labels)}
    boxes, out_labels = [], []
    for (x, y, w, h), cid in zip(bboxes_xywh, category_ids, strict=True):
        name = DOCLAYNET_TO_PAGE_ELEMENTS.get(int(cid))
        if name is None:
            continue
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append([x1, y1, x2, y2])
        out_labels.append(index[name])
    if not boxes:
        return None
    return np.asarray(boxes, dtype=np.float32), np.asarray(out_labels, dtype=np.int64)


def _split_shards(source: HFSource, split: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(source.repo_id, repo_type=source.repo_type, revision=source.revision)
    return sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))


def _iter_rows(parquet_path: str) -> Iterable[dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=64, columns=["image", "bboxes", "category_id", "metadata"]):
        yield from batch.to_pylist()


def prepare_split(split: str, cfg: SubsetConfig, out_dir: Path, source: HFSource = DOCLAYNET_V1_2) -> dict:
    """Write `<out_dir>/<split>.parquet` and return a manifest (idempotent: reuses an existing file)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.parquet"
    manifest_path = out_dir / f"{split}.manifest.json"
    if out_path.is_file() and manifest_path.is_file():
        return json.loads(manifest_path.read_text())

    limit = cfg.max_pages[split]
    cats = set(cfg.doc_categories)
    rows: dict[str, list] = {"image": [], "boxes": [], "labels": [], "image_id": [], "doc_category": []}
    seen = skipped_picture = skipped_empty = 0
    used_shards = []
    for shard in _split_shards(source, split):
        if len(rows["image"]) >= limit:
            break
        local = source.file(shard, local_dir=scratch_dir("doclaynet-shards"))
        used_shards.append(shard)
        for r in _iter_rows(str(local)):
            meta = r["metadata"]
            if meta["doc_category"] not in cats:
                continue
            seen += 1
            if cfg.drop_picture_pages and PICTURE_ID in r["category_id"]:
                skipped_picture += 1
                continue
            mapped = map_page(r["bboxes"], r["category_id"], meta["coco_width"], meta["coco_height"],
                              drop_picture_pages=False)
            if mapped is None:
                skipped_empty += 1
                continue
            boxes, labels = mapped
            rows["image"].append(r["image"]["bytes"])
            rows["boxes"].append(boxes.reshape(-1).tolist())
            rows["labels"].append(labels.tolist())
            rows["image_id"].append(meta["page_hash"])
            rows["doc_category"].append(meta["doc_category"])
            if len(rows["image"]) >= limit:
                break
        if os.environ.get("NEMOTRON_KEEP_SHARDS") != "1":
            Path(local).unlink(missing_ok=True)

    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.table(rows), tmp)
    tmp.replace(out_path)
    counts = np.bincount(np.concatenate([np.asarray(x) for x in rows["labels"]]) if rows["labels"] else [],
                         minlength=len(PAGE_ELEMENTS_LABELS))
    manifest = {
        "split": split,
        "pages": len(rows["image"]),
        "pages_seen_in_categories": seen,
        "skipped_picture_pages": skipped_picture,
        "skipped_empty_pages": skipped_empty,
        "boxes_per_class": dict(zip(PAGE_ELEMENTS_LABELS, counts.tolist(), strict=True)),
        "shards": used_shards,
        "source": f"{source.repo_id}@{source.revision}",
        "config": asdict(cfg),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_records(parquet_path: str | Path) -> list[DetRecord]:
    import pyarrow.parquet as pq

    t = pq.read_table(parquet_path, columns=["image", "boxes", "labels", "image_id"]).to_pydict()
    return [
        DetRecord(image=img, boxes=np.asarray(b, dtype=np.float32).reshape(-1, 4),
                  labels=np.asarray(lab, dtype=np.int64), image_id=iid)
        for img, b, lab, iid in zip(t["image"], t["boxes"], t["labels"], t["image_id"], strict=True)
    ]
