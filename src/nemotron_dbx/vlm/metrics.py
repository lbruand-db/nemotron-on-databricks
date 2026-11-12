"""CORD key-information extraction metrics (SPEC §6.5.2): the Donut evaluation, plus JSON validity.

`normalize`, `flatten`, `field_f1` and `ted_accuracy` port `JSONParseEvaluator` from Donut
(https://github.com/clovaai/donut, donut/util.py; Copyright (c) 2022-present NAVER Corp., MIT License):
- field F1: micro F1 over (key path, value) leaf pairs, after normalization;
- TED accuracy: max(0, 1 - TED(pred, gt) / TED({}, gt)), with zss tree edit distance and Donut's costs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..ocr.metrics import edit_distance


def parse_json_output(text: str) -> dict | None:
    """The first JSON object in a model answer (tolerates ``` fences and surrounding text); None if invalid."""
    if not text:
        return None
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize(data: Any) -> Any:
    """Donut `normalize_dict`: sort keys, wrap leaves in lists, drop empty values, stringify leaves."""
    if not data:
        return {}
    if isinstance(data, dict):
        new: Any = {}
        for key in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = normalize(data[key])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new[key] = value
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            new = [n for n in (normalize(item) for item in data) if n]
        else:
            new = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
    else:
        new = [str(data).strip()]
    return new


def flatten(data: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    def _flatten(value, key=""):
        if isinstance(value, dict):
            for child_key, child in value.items():
                _flatten(child, f"{key}.{child_key}" if key else child_key)
        elif isinstance(value, list):
            for item in value:
                _flatten(item, key)
        else:
            out.append((key, value))

    _flatten(data)
    return out


def field_f1(preds: list[dict], answers: list[dict]) -> float:
    tp, fn_or_fp = 0, 0
    for pred, answer in zip(preds, answers, strict=True):
        p, a = flatten(normalize(pred or {})), flatten(normalize(answer))
        for field in p:
            if field in a:
                tp += 1
                a.remove(field)
            else:
                fn_or_fp += 1
        fn_or_fp += len(a)
    denom = tp + fn_or_fp / 2
    return tp / denom if denom else 0.0


def _tree(data: Any, name: str = "<root>"):
    from zss import Node

    node = Node(name)
    if isinstance(data, dict):
        for key, value in data.items():
            node.addkid(_tree(value, key))
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            for item in data:
                node.addkid(_tree(item, "<subtree>"))
        else:
            for item in data:
                node.addkid(Node(f"<leaf>{item}"))
    else:
        raise ValueError(f"Unexpected value {data!r} under {name}")
    return node


def _update_cost(a, b) -> int:
    la, lb = a.label, b.label
    a_leaf, b_leaf = "<leaf>" in la, "<leaf>" in lb
    if a_leaf and b_leaf:
        return edit_distance(la.replace("<leaf>", ""), lb.replace("<leaf>", ""))
    if not a_leaf and b_leaf:
        return 1 + len(lb.replace("<leaf>", ""))
    if a_leaf and not b_leaf:
        return 1 + len(la.replace("<leaf>", ""))
    return int(la != lb)


def _insert_remove_cost(node) -> int:
    return len(node.label.replace("<leaf>", "")) if "<leaf>" in node.label else 1


def _ted(a, b) -> float:
    import zss

    return zss.distance(
        a, b, get_children=zss.Node.get_children, insert_cost=_insert_remove_cost,
        remove_cost=_insert_remove_cost, update_cost=_update_cost,
    )


def ted_accuracy(pred: dict | None, answer: dict) -> float:
    gt = _tree(normalize(answer))
    denom = _ted(_tree(normalize({})), gt)
    if denom == 0:
        return 1.0 if not normalize(pred or {}) else 0.0
    return float(max(0.0, 1 - _ted(_tree(normalize(pred or {})), gt) / denom))  # zss returns numpy floats


def extraction_metrics(texts: list[str], answers: list[dict]) -> dict[str, float]:
    """Model answers (raw text) vs CORD `gt_parse` dicts. Invalid JSON counts as an empty prediction."""
    preds = [parse_json_output(t) for t in texts]
    n = max(len(answers), 1)
    return {
        "json_valid": sum(p is not None for p in preds) / n,
        "field_f1": field_f1([p or {} for p in preds], answers),
        "ted_acc": sum(ted_accuracy(p, a) for p, a in zip(preds, answers, strict=True)) / n,
        "exact_match": sum(normalize(p or {}) == normalize(a) for p, a in zip(preds, answers, strict=True)) / n,
    }
