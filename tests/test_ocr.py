import numpy as np
import pytest

from nemotron_dbx.ocr.charset import BLANK, EOS, IGNORE, RARE, Charset
from nemotron_dbx.ocr.cord import parse_ground_truth
from nemotron_dbx.ocr.metrics import bag_of_words, edit_distance, recognition_metrics
from nemotron_dbx.ocr.serving import format_regions

CS = Charset([" ", "0", "1", "A", "B", "é"])  # ids: " "=3, "0"=4, "1"=5, "A"=6, "B"=7, "é"=8


def test_charset_encode_layout():
    t = CS.encode(["AB", "1"], seq_len=6)
    assert t.tolist() == [[6, 7, EOS, IGNORE, IGNORE, IGNORE], [5, EOS, IGNORE, IGNORE, IGNORE, IGNORE]]


def test_charset_trainable_rules():
    assert CS.trainable("AB é")
    assert not CS.trainable("ABC")  # C not in charset
    assert not CS.trainable("")
    assert not CS.trainable("A" * 32)  # needs room for EOS within 32 positions
    assert CS.oov_chars("A€B") == ["€"]
    with pytest.raises(ValueError):
        CS.encode(["C"])


def test_charset_decode_matches_nvidia_rules():
    # blanks skipped, no duplicate merging, stop at EOS, rare -> '^'
    assert CS.decode([6, BLANK, 6, 7, RARE, EOS, 6, 6]) == "AAB^"
    assert CS.decode(CS.encode(["AB é"], seq_len=8)[0].clip(min=0)) == "AB é"


def test_metrics():
    assert edit_distance("kitten", "sitting") == 3
    m = recognition_metrics(["AB", "1"], ["AB", "10"])
    assert m["cer"] == pytest.approx(1 / 4) and m["word_acc"] == 0.5
    b = bag_of_words([["A", "A", "B"]], [["A", "B", "C"]])
    assert b["bow_precision"] == pytest.approx(2 / 3) and b["bow_recall"] == pytest.approx(2 / 3)


def test_parse_cord_ground_truth_rescales_and_orders_vertices():
    gt = {
        "meta": {"image_size": {"width": 200, "height": 100}},
        "valid_line": [
            {"words": [
                {"quad": {"x1": 10, "y1": 20, "x2": 50, "y2": 20, "x3": 50, "y3": 40, "x4": 10, "y4": 40},
                 "text": "AB"},
                {"quad": {"x1": 0, "y1": 0, "x2": 1, "y2": 0, "x3": 1, "y3": 1, "x4": 0, "y4": 1}, "text": "  "},
            ]},
        ],
    }
    quads, texts = parse_ground_truth(gt, image_hw=(200, 400))  # image is 2x the annotated size
    assert texts == ["AB"]
    np.testing.assert_allclose(quads[0], [[20, 40], [100, 40], [100, 80], [20, 80]])


def test_format_regions_fixes_flipped_y_and_drops_nan():
    preds = [
        {"text": "TOTAL", "confidence": 0.9, "left": 0.1, "upper": 0.5, "right": 0.3, "lower": 0.45},
        {"text": "x", "confidence": 0.5, "left": float("nan"), "upper": 0, "right": 0, "lower": 0},
    ]
    out = format_regions(preds)
    assert len(out) == 1
    assert out[0]["y_min"] == 0.45 and out[0]["y_max"] == 0.5 and out[0]["x_min"] == 0.1
