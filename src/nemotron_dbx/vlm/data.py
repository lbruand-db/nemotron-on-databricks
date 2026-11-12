"""CORD-v2 receipts -> chat samples for Nano VL key-information extraction (SPEC §6.5.2).

Training, vLLM eval and the serving client all build the same conversation:
    system: "/no_think"
    user:   [image, INSTRUCTION]
    assistant: canonical JSON of the receipt's `gt_parse`
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

from ..common.hf import CORD_V2

SYSTEM = "/no_think"
INSTRUCTION = (
    "Extract the key information from this receipt as JSON, using the CORD schema. Use only the keys that "
    "apply: \"menu\" (a list of items, each with nm, num, unitprice, cnt, discountprice, price, itemsubtotal, "
    "vatyn, etc, and \"sub\" for sub-items with sub_nm, sub_unitprice, sub_cnt, sub_price, sub_etc), "
    "\"void_menu\" (nm, price), \"sub_total\" (subtotal_price, discount_price, service_price, othersvc_price, "
    "tax_price, etc) and \"total\" (total_price, total_etc, cashprice, changeprice, creditcardprice, "
    "emoneyprice, menutype_cnt, menuqty_cnt). Copy every value exactly as printed. Output only the JSON."
)
IGNORE_INDEX = -100


@dataclass
class VLSample:
    image: bytes
    target: dict  # gt_parse
    sample_id: str = ""

    @property
    def mime(self) -> str:
        from PIL import Image

        with Image.open(io.BytesIO(self.image)) as im:
            fmt = (im.format or "PNG").lower()
        return "image/jpeg" if fmt in ("jpeg", "jpg", "mpo") else f"image/{fmt}"

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{base64.b64encode(self.image).decode()}"


def target_text(gt_parse: dict) -> str:
    return json.dumps(gt_parse, ensure_ascii=False)


def hf_messages(instruction: str = INSTRUCTION) -> list[dict]:
    """Messages for the HF chat template (the processor inserts the image tiles at `<image>`)."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "image", "image": ""}, {"type": "text", "text": instruction}]},
    ]


def openai_messages(data_url: str, instruction: str = INSTRUCTION) -> list[dict]:
    """The same conversation in OpenAI chat format (vLLM eval and the serving endpoint)."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": instruction},
        ]},
    ]


def load_cord(split: str, max_items: int | None = None, cache_dir: str | Path | None = None) -> list[VLSample]:
    """One CORD-v2 split ("train" | "validation" | "test") from the pinned parquet files."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(CORD_V2.repo_id, repo_type="dataset", revision=CORD_V2.revision)
    shards = sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    out: list[VLSample] = []
    for shard in shards:
        local = CORD_V2.file(shard, local_dir=cache_dir)
        for row in pq.read_table(local, columns=["image", "ground_truth"]).to_pylist():
            gt = json.loads(row["ground_truth"])
            image_id = gt.get("meta", {}).get("image_id", len(out))
            out.append(VLSample(row["image"]["bytes"], gt["gt_parse"], f"{split}-{image_id}"))
            if max_items and len(out) >= max_items:
                return out
    return out


def check_pixel_values(pixel_values) -> None:
    """Fail fast on unnormalized tiles. With CLIP mean/std, [0, 1] inputs give values in about [-1.8, 2.2];
    [0, 255] inputs (SPEC §11.3 patch 11) give values in the hundreds, and the model silently learns to ignore
    the image."""
    peak = float(pixel_values.abs().max())
    if peak > 10:
        raise ValueError(f"pixel_values look unnormalized (max |x| = {peak:.0f}); is compat.fix_image_rescale applied?")


def encode_example(processor, tokenizer, sample: VLSample, max_len: int, eos_token_id: int):
    """input_ids / labels / pixel_values for one sample; labels are -100 on the prompt (and image) tokens.

    The prompt goes through the processor (which expands `<image>` into the tile tokens); the target is
    tokenized separately and appended, followed by a newline and the end-of-turn token (eos). That is what the
    model sees at generation time after vLLM applies the same chat template with `add_generation_prompt=True`.
    Returns None if the sequence would exceed `max_len`.
    """
    import torch
    from PIL import Image

    prompt = tokenizer.apply_chat_template(hf_messages(), tokenize=False, add_generation_prompt=True)
    with Image.open(io.BytesIO(sample.image)) as im:
        image = im.convert("RGB")
    enc = processor(text=[prompt], images=[image], return_tensors="pt")
    check_pixel_values(enc.pixel_values)
    prompt_ids = enc.input_ids[0]
    # The chat template renders an assistant turn as "<content>\n<SPECIAL_12>" (eos), so train on that.
    target = target_text(sample.target) + "\n"
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids[0]
    target_ids = torch.cat([target_ids, torch.tensor([eos_token_id])])
    if prompt_ids.numel() + target_ids.numel() > max_len:
        return None
    input_ids = torch.cat([prompt_ids, target_ids])
    labels = torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), target_ids])
    return {
        "input_ids": input_ids[None],
        "labels": labels[None],
        "attention_mask": torch.ones_like(input_ids)[None],
        "pixel_values": enc.pixel_values,
    }
