"""M0 spike (e): 10 LoRA steps on Nemotron Nano 12B v2 VL (HF remote code + PEFT) on CORD-v2.

Reports: kernel availability, load success/failure with the installed transformers, trainable
params, per-step loss/time, peak memory. Run under torchrun (the model's forward calls
torch.distributed.get_rank()).
"""

import importlib.util
import json
import os
import time
import traceback

import torch
import torch.distributed as dist
import transformers

MODEL = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
REVISION = os.environ.get("MODEL_REVISION") or None
STEPS = int(os.environ.get("STEPS", "10"))
# The remote code declares the transformers-4 attribute `_supports_flash_attn_2`, so transformers 5
# rejects FA2 for the top-level class; eager only affects the 6 attention layers (56 are Mamba/MLP).
ATTN = os.environ.get("ATTN_IMPL", "eager")
report = {"transformers": transformers.__version__, "torch": torch.__version__, "attn_impl": os.environ.get("ATTN_IMPL", "eager")}
for m in ("mamba_ssm", "causal_conv1d", "causal_conv1d_cuda", "flash_attn", "peft", "timm", "open_clip"):
    report[f"has:{m}"] = importlib.util.find_spec(m) is not None


def emit():
    if int(os.environ.get("RANK", "0")) == 0:
        print("SPIKE_RESULT=" + json.dumps(report, indent=2, default=str), flush=True)


dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
dev = torch.device("cuda")

try:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # transformers>=5 expects PreTrainedModel.__init__ to end with post_init() (sets e.g.
    # all_tied_weights_keys); the Nano VL remote code predates that. Patch the top-level class.
    vl_cls = get_class_from_dynamic_module("modeling.NemotronH_Nano_VL_V2", MODEL, revision=REVISION)
    if int(transformers.__version__.split(".")[0]) >= 5 and not getattr(vl_cls, "_nemotron_dbx_patched", False):
        orig_init = vl_cls.__init__

        def patched_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

        vl_cls.__init__ = patched_init
        vl_cls._nemotron_dbx_patched = True
        report["patch"] = "post_init"

    t = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation=ATTN,
        output_loading_info=True,
    )
    model, loading_info = model
    model = model.to(dev)

    # RADIO's `summary_idxs` buffer is absent from the checkpoint; transformers>=5 re-initialises
    # missing keys (garbage indices -> device-side assert), so recompute it from the RADIO config.
    rm = model.vision_model.radio_model
    if getattr(rm, "summary_idxs", None) is not None:
        teachers = model.vision_model.config.args["teachers"]
        rm.summary_idxs = torch.tensor(
            [i for i, t in enumerate(teachers) if t.get("use_summary", True)], dtype=torch.int64, device=dev
        )
        report["summary_idxs"] = rm.summary_idxs.tolist()

    # The VL wrapper reads outputs.past_key_values, but NemotronHCausalLMOutput only has cache_params.
    import sys as _sys

    out_cls = getattr(_sys.modules[type(model.language_model).__module__], "NemotronHCausalLMOutput", None)
    if out_cls is not None and not hasattr(out_cls, "past_key_values"):
        out_cls.past_key_values = None
        report["patch_output"] = "past_key_values=None"
    report["loading_info"] = {
        k: {"count": len(v), "sample": sorted(map(str, v))[:12]} for k, v in loading_info.items()
    }
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    proc = AutoProcessor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    report["load_s"] = round(time.monotonic() - t)
    mh = _sys_mod = __import__("sys").modules[type(model.language_model).__module__]
    report["mamba_fast_path"] = bool(getattr(mh, "is_fast_path_available", False))
    report["load"] = "ok"

    # Module-name inventory for choosing LoRA targets.
    names = {n.split(".")[-1] for n, m in model.language_model.named_modules() if isinstance(m, torch.nn.Linear)}
    report["llm_linear_leaf_names"] = sorted(names)

    for p in model.parameters():
        p.requires_grad_(False)
    targets = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "in_proj", "out_proj") if n in names]
    lcfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, target_modules=targets,
        # Restrict to the language model; vision tower + projector stay frozen.
        exclude_modules=r".*(vision_model|mlp1).*",
    )
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["lora_targets"] = targets
    report["trainable_params"] = trainable
    try:
        model.base_model.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Remove every input-require-grads hook (PEFT and transformers add them for checkpointing):
        # with non-reentrant checkpointing they are unnecessary, and they turn inputs_embeds into a
        # grad-requiring leaf that the VL wrapper then writes in-place -> RuntimeError.
        removed = 0
        for mod in model.modules():
            hook = getattr(mod, "_require_grads_hook", None)
            if hook is not None:
                hook.remove()
                del mod._require_grads_hook
                removed += 1
        report["removed_require_grads_hooks"] = removed
        report["grad_ckpt"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["grad_ckpt"] = f"failed: {exc!r}"

    ds = load_dataset("naver-clova-ix/cord-v2", split="train")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    instr = "Extract the receipt fields as JSON."
    steps = []
    model.train()
    for step in range(STEPS):
        row = ds[step]
        img = row["image"].convert("RGB")
        target = json.dumps(json.loads(row["ground_truth"])["gt_parse"], ensure_ascii=False)
        msgs = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": [{"type": "image", "image": ""}, {"type": "text", "text": instr}]},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + target + tok.eos_token
        enc_p = proc(text=[prompt], images=[img], return_tensors="pt")
        enc = proc(text=[full], images=[img], return_tensors="pt")
        input_ids = enc.input_ids.to(dev)
        labels = input_ids.clone()
        labels[:, : enc_p.input_ids.shape[1]] = -100
        pixel_values = enc.pixel_values.to(dev, torch.bfloat16)
        image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=dev)
        torch.cuda.synchronize()
        t = time.monotonic()
        out = model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=enc.attention_mask.to(dev),
            image_flags=image_flags, labels=labels, use_cache=False,
        )
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        steps.append({
            "step": step, "loss": round(loss.item(), 4), "seq_len": input_ids.shape[1],
            "tiles": pixel_values.shape[0], "step_s": round(time.monotonic() - t, 2),
        })
        print(f"STEP {steps[-1]}", flush=True)
    report["steps"] = steps
    report["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    report["status"] = "ok"
except Exception as exc:  # noqa: BLE001 - spike reports the failure mode
    report["status"] = "failed"
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()[-6000:]
finally:
    emit()
    dist.destroy_process_group()
