"""Run nemotron-ocr-v1 on the bundled example and a CORD-v2 receipt; report timing and output."""

import json
import sys
import tempfile
import time
from pathlib import Path

import torch
from datasets import load_dataset
from nemotron_ocr.inference.pipeline import NemotronOCR

repo = Path(sys.argv[1])
t = time.monotonic()
try:
    ocr = NemotronOCR(model_dir=str(repo / "checkpoints"))
except TypeError:
    ocr = NemotronOCR()
report = {"load_s": round(time.monotonic() - t, 1), "gpu": torch.cuda.get_device_name(0)}

images = [repo / "ocr-example-input-1.png"]
row = load_dataset("naver-clova-ix/cord-v2", split="test")[0]
cord = Path(tempfile.mkdtemp()) / "cord0.png"
row["image"].convert("RGB").save(cord)
images.append(cord)

for img in images:
    for level in ("word", "paragraph"):
        t = time.monotonic()
        try:
            preds = ocr(str(img), merge_level=level)
        except TypeError:
            preds = ocr(str(img))
        dt = time.monotonic() - t
        report[f"{img.name}:{level}"] = {
            "latency_s": round(dt, 3),
            "n_regions": len(preds),
            "sample": [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}
                for p in preds[:6]
            ],
        }
report["cord_gt_words"] = [
    w["text"] for line in json.loads(row["ground_truth"])["valid_line"] for w in line["words"]
][:30]
print("SPIKE_RESULT=" + json.dumps(report, indent=2, ensure_ascii=False, default=str))
