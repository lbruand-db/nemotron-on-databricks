"""Alloprof French retrieval set for the embedding eval (SPEC §6.6, OQ-19).

Source: `antoinelb7/alloprof` (MIT on the Hub), French Q&A from a Quebec school help forum. Each question links
the reference pages that teachers cited in their answers (`relevant`). The eval corpus is every French reference
page; queries are French questions with at least one relevant page in that corpus, hash-ordered by id so the
sample is stable. Alloprof is not in Nemotron-3-Embed's published training-data list, unlike MIRACL or MLDR.
Only used for evaluation, never for training.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..common.hf import HFSource

ALLOPROF = HFSource("antoinelb7/alloprof", "0faa90fee1ad1a6e3e461d7be49abf71488e6687", repo_type="dataset")
MAX_QUERY_CHARS = 2000  # a few forum posts paste whole essays


@dataclass
class RetrievalSet:
    doc_ids: list[str]
    docs: list[str]
    query_ids: list[str]
    queries: list[str]
    relevant: list[set[int]]  # per query: indices into docs


def build_retrieval_set(rows: list[dict], n_queries: int | None = 1000, min_query_chars: int = 10) -> RetrievalSet:
    """`rows` are alloprof.csv records (strings). Pure function, unit-tested without the download."""
    docs = [r for r in rows if r.get("is_query", "").lower() == "false" and r.get("language") == "fr"
            and r.get("text", "").strip()]
    index = {r["id"]: i for i, r in enumerate(docs)}
    queries = []
    for r in rows:
        if r.get("is_query", "").lower() != "true" or r.get("language") != "fr":
            continue
        text = r.get("text", "").strip()
        rel = {index[x] for x in r.get("relevant", "").split(";") if x in index}
        if rel and len(text) >= min_query_chars:
            queries.append((r["id"], text[:MAX_QUERY_CHARS], rel))
    queries.sort(key=lambda q: hashlib.md5(q[0].encode()).hexdigest())
    if n_queries:
        queries = queries[:n_queries]
    return RetrievalSet([r["id"] for r in docs], [r["text"] for r in docs], [q[0] for q in queries],
                        [q[1] for q in queries], [q[2] for q in queries])


def load_alloprof(cache_dir: str | Path, n_queries: int | None = 1000) -> RetrievalSet:
    import csv
    import sys

    csv.field_size_limit(sys.maxsize)
    path = ALLOPROF.file("data/alloprof.csv", local_dir=cache_dir)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return build_retrieval_set(rows, n_queries)
