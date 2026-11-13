"""PubTables-1M structure split -> nemotron-table-structure-v1 training records (SPEC §6.2).

PubTables-1M (CDLA-Permissive-2.0) ships table crops with PASCAL-VOC XML. Mapping to NVIDIA's classes:
- `table row` -> row, `table column` -> column;
- `cell` = row x column intersections, with merged cells (`table spanning cell`, and `table projected
  row header`, which is a full-width merged row) replacing the grid cells whose centres they cover.
  NVIDIA trained on Azure Document Intelligence cells, which are grid cells with merged cells as one box.
- `table`, `table column header` are dropped (NVIDIA's `border`/`header` slots are marked "not used").

Only the structure *Val* and *Test* archives are used (2.7 GB of images each): Val is split into our train
and val by a hash of the file name; a hash-ordered subset of Test is the held-out test set.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..common.hf import PUBTABLES_1M, HFSource
from ..common.paths import scratch_dir
from .doclaynet import load_records  # noqa: F401 - same parquet layout (image, boxes, labels, image_id)

# Indices in nemotron-table-structure-v1's `Exp.labels` = [border, cell, row, column, header].
CELL, ROW, COLUMN = 1, 2, 3
SPANNING = ("table spanning cell", "table projected row header")


@dataclass
class PubTablesSubset:
    train: int = 3000
    val: int = 300
    test: int = 300
    max_boxes: int = 1000  # tables with more cells+rows+columns are skipped (dataset max_labels)

    def key(self) -> str:
        return f"pubtables-structure_tr{self.train}_va{self.val}_te{self.test}_mb{self.max_boxes}"


def parse_voc(xml_bytes: bytes) -> tuple[int, int, list[tuple[str, list[float]]]]:
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    objs = []
    for o in root.iter("object"):
        b = o.find("bndbox")
        objs.append((o.findtext("name"), [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]))
    return w, h, objs


def derive_cells(rows: np.ndarray, cols: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Grid cells (row x column intersections) with merged cells replacing the ones they cover."""
    cells = []
    for r in rows:
        for c in cols:
            x0, y0, x1, y1 = max(r[0], c[0]), max(r[1], c[1]), min(r[2], c[2]), min(r[3], c[3])
            if x1 > x0 and y1 > y0:
                cells.append([x0, y0, x1, y1])
    cells = np.asarray(cells, dtype=np.float32).reshape(-1, 4)
    if len(spans) and len(cells):
        cx, cy = (cells[:, 0] + cells[:, 2]) / 2, (cells[:, 1] + cells[:, 3]) / 2
        covered = np.zeros(len(cells), dtype=bool)
        for s in spans:
            covered |= (cx >= s[0]) & (cx <= s[2]) & (cy >= s[1]) & (cy <= s[3])
        cells = np.concatenate([cells[~covered], spans.astype(np.float32)])
    return cells


def _dedupe(boxes: np.ndarray, iou_thr: float = 0.9) -> np.ndarray:
    """Spanning cells and projected row headers often coincide; keep one box per merged cell."""
    keep: list[np.ndarray] = []
    for b in boxes:
        if not any(_iou(b, k) > iou_thr for k in keep):
            keep.append(b)
    return np.asarray(keep, dtype=np.float32).reshape(-1, 4)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def map_table(objs: list[tuple[str, list[float]]]) -> tuple[np.ndarray, np.ndarray] | None:
    """VOC objects -> (boxes [N,4] xyxy px, labels [N]) in table-structure-v1 class indices, or None."""
    def boxes(*names):
        return np.asarray([b for n, b in objs if n in names], dtype=np.float32).reshape(-1, 4)

    rows, cols = boxes("table row"), boxes("table column")
    if not len(rows) or not len(cols):
        return None
    cells = derive_cells(rows, cols, _dedupe(boxes(*SPANNING)))
    out = np.concatenate([cells, rows, cols])
    labels = np.concatenate([np.full(len(cells), CELL), np.full(len(rows), ROW), np.full(len(cols), COLUMN)])
    return out, labels.astype(np.int64)


def _hash_order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: hashlib.md5(n.encode()).hexdigest())


def _read_annotations(tar_path: Path, wanted: set[str] | None = None) -> dict[str, bytes]:
    out = {}
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".xml"):
                name = Path(m.name).name
                if wanted is None:
                    out[name] = b""
                elif name in wanted:
                    out[name] = tf.extractfile(m).read()
    return out


def _read_images(tar_path: Path, wanted: set[str]) -> dict[str, bytes]:
    out = {}
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            name = Path(m.name).name
            if m.isfile() and name in wanted:
                out[name] = tf.extractfile(m).read()
                if len(out) == len(wanted):
                    break
    return out


def _select(source: HFSource, part: str, counts: list[int], max_boxes: int, scratch: Path):
    """Hash-ordered tables from one archive pair ("Val" | "Test"), split into consecutive groups of `counts`."""
    t0 = time.monotonic()
    ann_tar = source.file(f"PubTables-1M-Structure_Annotations_{part}.tar.gz", local_dir=scratch)
    names = _hash_order(list(_read_annotations(ann_tar)))
    # Parse a margin of extra candidates: some tables are skipped (no rows/columns, or too many boxes).
    need = sum(counts)
    candidates = names[: int(need * 1.3) + 50]
    xmls = _read_annotations(ann_tar, set(candidates))
    picked, skipped = [], {"no_grid": 0, "too_many_boxes": 0}
    for name in candidates:
        w, h, objs = parse_voc(xmls[name])
        mapped = map_table(objs)
        if mapped is None:
            skipped["no_grid"] += 1
            continue
        if len(mapped[0]) > max_boxes:
            skipped["too_many_boxes"] += 1
            continue
        picked.append((name, w, h, *mapped))
        if len(picked) == need:
            break
    print(f"[data] {part}: picked {len(picked)} tables of {len(names)} (skipped {skipped}) "
          f"at {time.monotonic() - t0:.0f}s; reading images", flush=True)
    img_tar = source.file(f"PubTables-1M-Structure_Images_{part}.tar.gz", local_dir=scratch)
    images = _read_images(img_tar, {n.replace(".xml", ".jpg") for n, *_ in picked})
    img_tar.unlink(missing_ok=True)
    print(f"[data] {part}: {len(images)} images read at {time.monotonic() - t0:.0f}s", flush=True)
    groups, i = [], 0
    for c in counts:
        groups.append([(images[n.replace(".xml", ".jpg")], *rest) for n, *rest in picked[i:i + c]
                       if n.replace(".xml", ".jpg") in images])
        i += c
    return groups, skipped, len(names)


def prepare(cfg: PubTablesSubset, out_dir: Path, source: HFSource = PUBTABLES_1M) -> dict:
    """Write `<out_dir>/{train,validation,test}.parquet` + manifest (idempotent)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file() and all((out_dir / f"{s}.parquet").is_file() for s in ("train", "validation", "test")):
        return json.loads(manifest_path.read_text())
    scratch = scratch_dir("pubtables")
    (train, val), skipped_val, n_val_src = _select(source, "Val", [cfg.train, cfg.val], cfg.max_boxes, scratch)
    (test,), skipped_test, n_test_src = _select(source, "Test", [cfg.test], cfg.max_boxes, scratch)
    manifest = {"source": f"{source.repo_id}@{source.revision}", "config": asdict(cfg),
                "skipped": {"Val": skipped_val, "Test": skipped_test},
                "source_tables": {"Val": n_val_src, "Test": n_test_src}, "splits": {}}
    for split, rows in (("train", train), ("validation", val), ("test", test)):
        cols = {"image": [r[0] for r in rows], "image_id": [f"{split}-{i}" for i in range(len(rows))],
                "boxes": [r[3].reshape(-1).tolist() for r in rows], "labels": [r[4].tolist() for r in rows]}
        tmp = out_dir / f"{split}.parquet.tmp"
        pq.write_table(pa.table(cols), tmp)
        tmp.replace(out_dir / f"{split}.parquet")
        counts = np.bincount(np.concatenate([r[4] for r in rows]) if rows else [], minlength=5)
        manifest["splits"][split] = {"tables": len(rows), "cells": int(counts[CELL]), "rows": int(counts[ROW]),
                                     "columns": int(counts[COLUMN])}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
