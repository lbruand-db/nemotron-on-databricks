"""OCR metrics: character error rate, word accuracy, page-level bag-of-words F1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def recognition_metrics(preds: Sequence[str], gts: Sequence[str]) -> dict[str, float]:
    """Region-level recognition (same regions for pred and GT): CER over GT characters, exact word acc."""
    if len(preds) != len(gts):
        raise ValueError("preds and gts must align")
    chars = sum(len(g) for g in gts)
    errors = sum(edit_distance(p, g) for p, g in zip(preds, gts, strict=True))
    exact = sum(p == g for p, g in zip(preds, gts, strict=True))
    return {
        "cer": errors / max(1, chars),
        "word_acc": exact / max(1, len(gts)),
        "regions": float(len(gts)),
    }


def bag_of_words(pred_words: Sequence[Sequence[str]], gt_words: Sequence[Sequence[str]]) -> dict[str, float]:
    """End-to-end, order- and box-free: multiset precision / recall / F1 of words per page, micro-averaged."""
    tp = n_pred = n_gt = 0
    for p, g in zip(pred_words, gt_words, strict=True):
        cp, cg = Counter(p), Counter(g)
        tp += sum((cp & cg).values())
        n_pred += sum(cp.values())
        n_gt += sum(cg.values())
    precision = tp / max(1, n_pred)
    recall = tp / max(1, n_gt)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"bow_precision": precision, "bow_recall": recall, "bow_f1": f1}
