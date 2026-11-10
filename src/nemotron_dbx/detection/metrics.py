"""COCO-style detection metrics (pycocotools) for normalized predictions vs pixel ground truth."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence

import numpy as np


def coco_map(
    gts: Sequence[tuple[np.ndarray, np.ndarray]],
    preds: Sequence[dict[str, np.ndarray]],
    image_sizes: Sequence[tuple[int, int]],
    class_names: Sequence[str],
    max_dets: int = 300,
) -> dict[str, float]:
    """mAP@[.5:.95], mAP@.5 and per-class AP.

    Args:
        gts: per image (boxes [N,4] xyxy pixels, labels [N] int).
        preds: per image {"boxes": [M,4] normalized xyxy, "labels": [M], "scores": [M]}.
        image_sizes: per image (height, width) in pixels (to un-normalize predictions).
        class_names: index -> name.
        max_dets: detections per image considered (documents often exceed COCO's default 100).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    ann_id = 1
    for img_id, ((gb, gl), pred, (h, w)) in enumerate(zip(gts, preds, image_sizes, strict=True), start=1):
        images.append({"id": img_id, "height": int(h), "width": int(w)})
        for box, lab in zip(np.asarray(gb).reshape(-1, 4), np.asarray(gl).reshape(-1), strict=True):
            x1, y1, x2, y2 = (float(v) for v in box)
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            anns.append({"id": ann_id, "image_id": img_id, "category_id": int(lab) + 1,
                         "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1
        pb = np.asarray(pred["boxes"], dtype=np.float64).reshape(-1, 4) * np.array([w, h, w, h])
        for box, lab, score in zip(pb, pred["labels"], pred["scores"], strict=True):
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append({"image_id": img_id, "category_id": int(lab) + 1,
                         "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": float(score)})

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": anns, "categories": categories}
        coco_gt.createIndex()
        if not dets:
            return {"map": 0.0, "map50": 0.0, **{f"ap/{n}": 0.0 for n in class_names}}
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate()
        ev.accumulate()

    # precision: [T iou thresholds, R recall points, K classes, A area ranges, M max dets].
    # Computed directly: COCOeval.summarize() hard-codes maxDets=100 and returns -1 for other values.
    precision = ev.eval["precision"]

    def _mean(p: np.ndarray) -> float:
        valid = p[p > -1]
        return float(valid.mean()) if valid.size else 0.0

    out = {"map": _mean(precision[:, :, :, 0, -1]), "map50": _mean(precision[0, :, :, 0, -1])}
    present = {a["category_id"] for a in anns}
    for k, name in enumerate(class_names):
        p = precision[:, :, k, 0, -1]
        valid = p[p > -1]
        out[f"ap/{name}"] = float(valid.mean()) if (k + 1) in present and valid.size else float("nan")
    return out
