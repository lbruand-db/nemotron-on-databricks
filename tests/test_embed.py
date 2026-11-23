import numpy as np
import pytest

from nemotron_dbx.embed.data import build_retrieval_set
from nemotron_dbx.embed.metrics import retrieval_metrics, top_k_indices
from nemotron_dbx.embed.server import (
    CARD_SCORES,
    EMBED_VLLM,
    PREFIXES,
    compare_to_card,
    parse_dims_list,
    retryable,
    truncate_normalize,
    vllm_embed_command,
)
from nemotron_dbx.embed.serving import parse_dimensions, parse_records
from nemotron_dbx.lifecycle.query import build_embed_request, summarize_embeddings


def test_truncate_normalize():
    x = np.array([[3.0, 4.0, 12.0], [1.0, 0.0, 0.0]])
    full = truncate_normalize(x, None)
    np.testing.assert_allclose(np.linalg.norm(full, axis=1), 1.0, rtol=1e-6)
    two = truncate_normalize(x, 2)
    np.testing.assert_allclose(two[0], [0.6, 0.8], rtol=1e-6)
    for bad in (4, 0, -1):
        with pytest.raises(ValueError):
            truncate_normalize(x, bad)


def test_retrieval_metrics_perfect_and_partial():
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.2, 0.9]])
    m = retrieval_metrics(scores, [{0}, {1}], ks=(1, 2))
    assert m["recall@1"] == 0.5 and m["recall@2"] == 1.0
    # query 0: rank 1 -> ndcg 1, mrr 1; query 1: relevant at rank 2 -> ndcg 1/log2(3), mrr 1/2
    assert m["ndcg@10"] == pytest.approx((1 + 1 / np.log2(3)) / 2, abs=1e-5)
    assert m["mrr@10"] == pytest.approx(0.75, abs=1e-5)
    assert all(type(v) is float for v in m.values())  # plain floats (clean logs and JSON)


def test_build_retrieval_set_filters_and_is_stable():
    rows = [
        {"id": "d1", "is_query": "false", "language": "fr", "text": "Les fractions"},
        {"id": "d2", "is_query": "false", "language": "fr", "text": "La photosynthèse"},
        {"id": "de", "is_query": "false", "language": "en", "text": "Fractions"},
        {"id": "q1", "is_query": "true", "language": "fr", "text": "Comment additionner des fractions ?",
         "relevant": "d1;de"},
        {"id": "q2", "is_query": "true", "language": "fr", "text": "court", "relevant": "d2"},  # too short
        {"id": "q3", "is_query": "true", "language": "fr", "text": "Question sans lien utile", "relevant": "de"},
        {"id": "q4", "is_query": "true", "language": "fr", "text": "Qu'est-ce que la photosynthèse ?",
         "relevant": "d2"},
    ]
    rs = build_retrieval_set(rows, n_queries=None)
    assert rs.doc_ids == ["d1", "d2"] and sorted(rs.query_ids) == ["q1", "q4"]
    assert rs.relevant[rs.query_ids.index("q1")] == {0}
    assert build_retrieval_set(list(reversed(rows)), n_queries=None).query_ids == rs.query_ids


def test_embed_command_and_records():
    cmd = vllm_embed_command("m", "served", 3081, EMBED_VLLM)
    assert cmd[:2] == ["env", "HF_HUB_OFFLINE=1"] and "--runner" in cmd and cmd[cmd.index("--runner") + 1] == "pooling"
    assert "--video-pruning-rate" not in cmd and cmd[cmd.index("--max-model-len") + 1] == "8192"
    assert PREFIXES == {"query": "query: ", "document": "passage: "}
    recs = parse_records([{"text": "a"}, {"text": "b", "input_type": "QUERY", "id": "x"}])
    assert recs == [(0, "document", "a"), ("x", "query", "b")]
    for bad in ({"text": ""}, {"text": "a", "input_type": "passage"}):
        with pytest.raises(ValueError):
            parse_records([bad])
    assert np.argmax(CARD_SCORES, axis=1).tolist() == [0, 1, 2, 3]


def test_embed_request_and_summary():
    body = build_embed_request(["a", "b"], "query", 1024)
    assert body["params"] == {"dimensions": 1024}
    assert [r["input_type"] for r in body["dataframe_records"]] == ["query", "query"]
    preds = [{"id": 0, "input_type": "query", "embedding": [1.0, 0.0]},
             {"id": 1, "input_type": "document", "embedding": [0.0, 1.0]}]
    out = summarize_embeddings(preds)
    assert "dims=2" in out and "similarity:" in out


def test_model_signature_accepts_documented_requests():
    pytest.importorskip("mlflow")
    import pandas as pd
    from mlflow.models.utils import _enforce_schema

    from nemotron_dbx.embed.serving import model_signature

    sig = model_signature()
    for req in ([{"text": "x"}], [{"text": "x", "id": 3, "input_type": "query"}], [{"text": "x", "id": "d1"}]):
        _enforce_schema(pd.DataFrame(req), sig.inputs)  # text only, integer ids and string ids all pass
    assert sig.params.to_dict() == [{"name": "dimensions", "type": "long", "default": 4096, "shape": None}]
    # Unity Catalog refuses input-only signatures ("must be logged with ... both input and output").
    assert sig.outputs is not None and {c.name for c in sig.outputs.inputs} == {"id", "input_type", "dimensions",
                                                                                "embedding"}


def test_parse_dimensions():
    assert parse_dimensions(None) == 4096 and parse_dimensions({"dimensions": 1024}) == 1024
    assert parse_dimensions({"dimensions": "2048"}) == 2048 and parse_dimensions({"dimensions": 512.0}) == 512
    for bad in (0, -1, 5000, 1.5, True, "abc", None, [1024]):
        with pytest.raises(ValueError):
            parse_dimensions({"dimensions": bad})


def test_retryable_only_transient_errors():
    import requests

    def http(code):
        r = requests.Response()
        r.status_code = code
        return requests.HTTPError(response=r)

    assert retryable(requests.ConnectionError()) and retryable(requests.Timeout())
    assert retryable(http(503)) and retryable(http(429))
    assert not retryable(http(400)) and not retryable(http(404)) and not retryable(ValueError())


def test_parse_dims_list():
    assert parse_dims_list("4096,2048,1024") == [4096, 2048, 1024] and parse_dims_list(" 512 ") == [512]
    for bad in ("4096,8192", "0", "", "4096,abc"):
        with pytest.raises(ValueError):
            parse_dims_list(bad)


def test_compare_to_card():
    # Orthonormal-ish vectors whose dot products reproduce the card's score matrix exactly.
    u, sv, vt = np.linalg.svd(CARD_SCORES.astype(np.float64))
    q = (u * np.sqrt(sv)).astype(np.float32)
    d = (vt.T * np.sqrt(sv)).astype(np.float32)
    assert compare_to_card(q, d)["max_abs_err"] < 1e-4
    with pytest.raises(RuntimeError):
        compare_to_card(q, d[[1, 0, 2, 3]])  # documents swapped: query 0 no longer ranks its own first


def test_top_k_indices_matches_full_sort():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((20, 500)).astype(np.float32)
    np.testing.assert_array_equal(top_k_indices(scores, 10), np.argsort(-scores, axis=1)[:, :10])
    assert top_k_indices(scores[:, :3], 10).shape == (20, 3)
