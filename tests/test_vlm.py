import io
import json

import pytest
import torch
from PIL import Image

from nemotron_dbx.vlm import compat
from nemotron_dbx.vlm.data import INSTRUCTION, SYSTEM, VLSample, openai_messages, target_text
from nemotron_dbx.vlm.lora import lr_factor, rank_slice
from nemotron_dbx.vlm.metrics import extraction_metrics, field_f1, normalize, parse_json_output, ted_accuracy
from nemotron_dbx.vlm.vllm_server import (
    A10G_BUDGET_GIB,
    VLLMConfig,
    vllm_command,
    vllm_config_from_json,
    with_gpu_budget,
)

GT = {
    "menu": [{"nm": "ICED TEA", "cnt": "1", "price": "20,000"}, {"nm": "NASI", "cnt": "2", "price": "50,000"}],
    "total": {"total_price": "70,000", "cashprice": "100,000"},
}


def _png(fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format=fmt)
    return buf.getvalue()


# --- metrics (Donut JSONParseEvaluator semantics) ---

def test_parse_json_output_tolerates_fences_and_text():
    assert parse_json_output('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_output('Here you go: {"a": {"b": "x"}} done') == {"a": {"b": "x"}}
    assert parse_json_output("not json") is None
    assert parse_json_output("[1, 2]") is None


def test_normalize_sorts_wraps_and_drops_empty():
    assert normalize({"b": "1", "a": "", "cc": ["x", " "]}) == {"b": ["1"], "cc": ["x"]}
    assert normalize({"menu": {"nm": "A"}}) == {"menu": [{"nm": ["A"]}]}  # single item dict == list of one


def test_field_f1_and_ted():
    assert field_f1([GT], [GT]) == 1.0
    assert ted_accuracy(GT, GT) == 1.0
    assert ted_accuracy(None, GT) == 0.0
    pred = json.loads(json.dumps(GT))
    pred["total"]["cashprice"] = "100.000"  # one wrong leaf out of 8
    assert field_f1([pred], [GT]) == pytest.approx(7 / 8)
    assert 0.9 < ted_accuracy(pred, GT) < 1.0
    # menu order matters for TED but not for F1
    swapped = {**GT, "menu": GT["menu"][::-1]}
    assert field_f1([swapped], [GT]) == 1.0
    assert ted_accuracy(swapped, GT) < 1.0


def test_extraction_metrics_counts_invalid_json_as_empty():
    m = extraction_metrics([json.dumps(GT), "oops"], [GT, GT])
    assert m["json_valid"] == 0.5 and m["exact_match"] == 0.5
    assert m["field_f1"] == pytest.approx(8 / (8 + 8 / 2))
    assert m["ted_acc"] == pytest.approx(0.5)


# --- data / request format ---

def test_conversation_format_matches_training():
    s = VLSample(_png("JPEG"), GT)
    assert s.mime == "image/jpeg" and VLSample(_png(), GT).mime == "image/png"
    msgs = openai_messages(s.data_url())
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    assert msgs[1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert msgs[1]["content"][1]["text"] == INSTRUCTION
    assert json.loads(target_text(GT)) == GT and "ICED TEA" in target_text(GT)


def test_build_chat_request(tmp_path):
    from nemotron_dbx.lifecycle.query import build_chat_request

    img = tmp_path / "r.png"
    img.write_bytes(_png())
    body = build_chat_request(str(img))
    assert body["temperature"] == 0 and body["messages"][1]["content"][1]["text"] == INSTRUCTION


def test_vllm_command_has_serving_flags():
    cmd = vllm_command("nano_vl_ft", "m", 8080, VLLMConfig(max_model_len=8192))
    s = " ".join(cmd)
    assert s.startswith("env VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 python -u -m vllm")
    for flag in ("--model nano_vl_ft", "--port 8080", "--host 0.0.0.0", "--trust-remote-code",
                 "--max-model-len 8192", "--video-pruning-rate 0", "--dtype bfloat16"):
        assert flag in s


def test_vllm_command_env_and_extra_args():
    cfg = VLLMConfig(extra_args=("--quantization", "fp8"), env=("VLLM_TEST_FORCE_FP8_MARLIN=1",))
    cmd = vllm_command("m", "m", 8080, cfg)
    assert cmd[:4] == ["env", "VLLM_USE_FLASHINFER_SAMPLER=0", "HF_HUB_OFFLINE=1", "VLLM_TEST_FORCE_FP8_MARLIN=1"]
    assert cmd[-2:] == ["--quantization", "fp8"]
    for bad in ("VLLM_TEST_FORCE_FP8_MARLIN", "-i", "1X=2", "A=b c"):
        with pytest.raises(ValueError, match="NAME=VALUE"):
            VLLMConfig(env=(bad,))



def test_vllm_config_json_round_trip():
    import dataclasses
    import json

    cfg = VLLMConfig(max_num_seqs=8, extra_args=("--quantization", "fp8"), env=("A=1",))
    assert vllm_config_from_json(json.loads(json.dumps(dataclasses.asdict(cfg)))) == cfg



def test_gpu_budget_caps_utilization_only():
    cfg = VLLMConfig(extra_args=("--quantization", "fp8"))
    capped = with_gpu_budget(cfg, A10G_BUDGET_GIB, total_gib=79.2)
    assert capped.gpu_memory_utilization == pytest.approx(0.2551, abs=1e-4)
    assert capped.extra_args == cfg.extra_args
    assert with_gpu_budget(cfg, 0) is cfg
    assert with_gpu_budget(cfg, 40, total_gib=22.4).gpu_memory_utilization == cfg.gpu_memory_utilization  # never above


# --- training schedule ---

def test_lr_schedule_and_rank_slices():
    assert lr_factor(0, 100, 5) == pytest.approx(0.2)
    assert lr_factor(4, 100, 5) == pytest.approx(1.0)
    assert lr_factor(5, 100, 5) == pytest.approx(1.0)
    assert lr_factor(99, 100, 5) < 0.01
    order = list(range(40))
    parts = [rank_slice(order, 1, 16, r, 4) for r in range(4)]
    assert sorted(sum(parts, [])) == list(range(16, 32)) and all(len(p) == 4 for p in parts)


# --- compat patches ---

def test_pin_remote_code_injects_revision(monkeypatch):
    from transformers.models.auto import auto_factory

    calls = []

    def fake(class_ref, repo, *args, **kwargs):
        calls.append((repo, kwargs.get("code_revision")))

    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", fake)
    compat.pin_remote_code({"org/pinned": "abc123"})
    compat.pin_remote_code({"org/pinned": "abc123"})  # idempotent: wraps the original, not the wrapper
    auto_factory.get_class_from_dynamic_module("m.C", "org/pinned")
    auto_factory.get_class_from_dynamic_module("m.C", "org/other")
    auto_factory.get_class_from_dynamic_module("m.C", "org/pinned", code_revision="explicit")
    assert calls == [("org/pinned", "abc123"), ("org/other", None), ("org/pinned", "explicit")]


def test_patch_post_init_and_hooks():
    class Fake:
        def __init__(self):
            self.inited = True

        def post_init(self):
            self.all_tied_weights_keys = {}

    assert compat.patch_post_init(Fake) and not compat.patch_post_init(Fake)
    assert Fake().all_tied_weights_keys == {}

    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    m[0]._require_grads_hook = m[0].register_forward_hook(lambda *a: None)
    assert compat.remove_require_grads_hooks(m) == 1 and not m[0]._forward_hooks
    teachers = [{"name": "a"}, {"name": "b", "use_summary": False}, {"name": "c", "use_summary": True}]
    assert compat.expected_summary_idxs(teachers) == [0, 2]


# --- adapter merge == PEFT merge_and_unload, with the checkpoint's names and shards ---

class _Mixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Linear(6, 10, bias=False)
        self.out_proj = torch.nn.Linear(10, 6, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = _Mixer()
        self.norm = torch.nn.LayerNorm(6)


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])


class _LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.lm_head = torch.nn.Linear(6, 4, bias=False)


class _VL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LM()
        self.mlp1 = torch.nn.Sequential(torch.nn.Linear(6, 6))  # frozen projector (excluded)

    def forward(self, x):
        h = x
        for layer in self.language_model.backbone.layers:
            h = h + layer.mixer.out_proj(layer.mixer.in_proj(layer.norm(h)))
        return self.language_model.lm_head(self.mlp1(h))


def test_merge_adapter_matches_peft(tmp_path):
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    from nemotron_dbx.vlm.lora import FROZEN_MODULES
    from nemotron_dbx.vlm.merge import merge_adapter

    torch.manual_seed(0)
    base = _VL()
    sd = {k: v.clone().to(torch.bfloat16) for k, v in base.state_dict().items()}
    keys = sorted(sd)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    shards = {"model-00001-of-00002.safetensors": keys[: len(keys) // 2],
              "model-00002-of-00002.safetensors": keys[len(keys) // 2:]}
    for name, ks in shards.items():
        save_file({k: sd[k] for k in ks}, str(base_dir / name), metadata={"format": "pt"})
    index = {"metadata": {}, "weight_map": {k: n for n, ks in shards.items() for k in ks}}
    (base_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    (base_dir / "modeling.py").write_text("# remote code\n")

    model = _VL().to(torch.bfloat16)
    model.load_state_dict(sd)
    lcfg = LoraConfig(r=4, lora_alpha=8, target_modules=["in_proj", "out_proj"], exclude_modules=FROZEN_MODULES,
                      init_lora_weights=False)  # random A and B, so the delta is non-zero
    peft_model = get_peft_model(model, lcfg)
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    expected = {k: v.float() for k, v in peft_model.merge_and_unload().state_dict().items()}

    info = merge_adapter(base_dir, adapter_dir, tmp_path / "merged")
    assert info["merged_modules"] == 4 and info["scale"] == 2.0
    merged_dir = tmp_path / "merged"
    assert (merged_dir / "modeling.py").exists() and (merged_dir / "model.safetensors.index.json").exists()
    from safetensors.torch import load_file

    merged = {}
    for name in shards:
        part = load_file(str(merged_dir / name))
        assert sorted(part) == sorted(shards[name])  # same tensors in the same shards
        merged |= part
    assert all(v.dtype == torch.bfloat16 for v in merged.values())
    for k, v in expected.items():
        torch.testing.assert_close(merged[k].float(), v, atol=2e-2, rtol=2e-2)
    changed = [k for k in keys if not torch.equal(merged[k], sd[k])]
    assert sorted(changed) == sorted(k for k in keys if "_proj" in k)


# --- patch 11: image rescale (needs the Nano VL processor code; download with the snippet in the README) ---

NANO_VL_CODE = __import__("os").environ.get("NEMOTRON_NANO_VL_CODE", "/tmp/hf_nanovl")


@pytest.mark.skipif(not __import__("os").path.exists(f"{NANO_VL_CODE}/image_processing.py"),
                    reason="Nano VL processor code not available")
def test_image_processor_rescale_patch():
    from transformers import AutoProcessor

    from nemotron_dbx.vlm.data import check_pixel_values

    proc = AutoProcessor.from_pretrained(NANO_VL_CODE, trust_remote_code=True)
    img = Image.new("RGB", (600, 400), (200, 120, 40))
    raw = proc.image_processor(images=[img], return_tensors="pt").pixel_values
    patched = compat.fix_image_rescale(proc.image_processor)
    fixed = proc.image_processor(images=[img], return_tensors="pt").pixel_values
    if patched:  # transformers 5: the remote `_process_image` hook is dead code
        with pytest.raises(ValueError):
            check_pixel_values(raw)
        mean = torch.tensor(proc.image_processor.norm_mean).view(1, 3, 1, 1)
        std = torch.tensor(proc.image_processor.norm_std).view(1, 3, 1, 1)
        # only the input scale differs: de-normalized raw == 255 x de-normalized fixed
        torch.testing.assert_close(raw * std + mean, 255 * (fixed * std + mean), rtol=1e-3, atol=1.0)
    check_pixel_values(fixed)
    # (200/255 - 0.48145466) / 0.26862954 for the red channel of a flat image
    assert fixed[:, 0].mean().item() == pytest.approx((200 / 255 - 0.48145466) / 0.26862954, abs=1e-2)
    assert not compat.fix_image_rescale(proc.image_processor)  # idempotent


def test_vllm_pyfunc_predict_contract():
    import base64

    import pandas as pd

    from nemotron_dbx.vlm.serving import VLLMChatModel

    class FakeServer:
        def batch_chat(self, convs, concurrency, max_tokens):
            self.convs, self.max_tokens = convs, max_tokens
            return [{"text": json.dumps(GT), "usage": {"completion_tokens": 9}, "latency_s": 1.0},
                    {"text": "not json", "latency_s": 0.5, "error": "boom"}]

    m = VLLMChatModel()
    m.server = FakeServer()
    img = base64.b64encode(_png()).decode()
    df = pd.DataFrame([{"image": img, "id": "a"}, {"image": img, "id": "b", "instruction": "Only the total."}])
    out = m.predict(None, df, params={"max_tokens": 64})
    assert [o["id"] for o in out] == ["a", "b"] and m.server.max_tokens == 64
    assert out[0]["fields"] == GT and out[0]["json_valid"] and out[0]["usage"] == {"completion_tokens": 9}
    assert out[1]["fields"] is None and not out[1]["json_valid"] and out[1]["error"] == "boom"
    texts = [c[1]["content"][1]["text"] for c in m.server.convs]
    assert texts == [INSTRUCTION, "Only the total."]  # NaN instruction in row a falls back to the default
