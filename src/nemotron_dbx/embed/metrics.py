"""Retrieval metrics over a dense score matrix (queries x documents), binary relevance."""

from __future__ import annotations

import numpy as np


def retrieval_metrics(scores: np.ndarray, relevant: list[set[int]], ks: tuple[int, ...] = (1, 5, 10, 100)) -> dict:
    """NDCG@10, MRR@10 and recall@k. `relevant[i]` holds the relevant document indices of query i."""
    n_q, n_d = scores.shape
    if n_q != len(relevant):
        raise ValueError(f"{n_q} score rows for {len(relevant)} queries")
    depth = min(max(*ks, 10), n_d)
    top = np.argsort(-scores, axis=1)[:, :depth]
    discounts = 1.0 / np.log2(np.arange(2, 12))
    out = {f"recall@{k}": 0.0 for k in ks} | {"ndcg@10": 0.0, "mrr@10": 0.0}
    for ranked, rel in zip(top, relevant, strict=True):
        hits = np.array([d in rel for d in ranked])
        for k in ks:
            out[f"recall@{k}"] += hits[:k].sum() / len(rel)
        gains = hits[:10].astype(float)
        ideal = discounts[: min(len(rel), 10)].sum()
        out["ndcg@10"] += float((gains * discounts[: len(gains)]).sum() / ideal)
        first = np.flatnonzero(hits[:10])
        out["mrr@10"] += 1.0 / (first[0] + 1) if len(first) else 0.0
    return {k: round(float(v) / n_q, 5) for k, v in out.items()}
