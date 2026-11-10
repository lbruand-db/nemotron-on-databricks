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
report = {"transformers": transformers.__version__, "torch": torch.__version__}
for m in ("mamba_ssm", "causal_conv1d", "flash_attn", "peft", "timm", "open_clip"):
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

    t = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(dev)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    proc = AutoProcessor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    report["load_s"] = round(time.monotonic() - t)
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
        model.base_model.model.language_model.enable_input_require_grads()
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
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        steps.append({
            "step": step, "loss": round(out.loss.item(), 4), "seq_len": input_ids.shape[1],
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
