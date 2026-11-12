"""Compatibility patches for the Nano VL remote code under transformers 5 (SPEC §11.3, patches 5-11).

The remote code was written for transformers 4.5x. Every patch checks for the symptom first, so it turns into
a no-op once upstream fixes it.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Mapping

ATTN_IMPLEMENTATION = "eager"  # patch 5: the remote code only declares the transformers-4 FA2 flag


def pin_remote_code(pins: Mapping[str, str]) -> None:
    """Pin cross-repo remote code (`"repo--module.Class"` auto_map refs) to fixed revisions.

    `AutoModel*.from_config` resolves those refs at the upstream repo's `main` branch. The Nano VL modeling
    code calls it for its LLM (Nano v2 Base repo) and vision tower (C-RADIOv2-H repo), so without this the
    training code could change under a pinned model revision.
    """
    from transformers.models.auto import auto_factory

    current = auto_factory.get_class_from_dynamic_module
    original = getattr(current, "_nemotron_dbx_original", current)

    @functools.wraps(original)
    def pinned(class_reference, pretrained_model_name_or_path, *args, **kwargs):
        repo = str(pretrained_model_name_or_path)
        if repo in pins and not kwargs.get("code_revision") and not kwargs.get("revision"):
            kwargs["code_revision"] = pins[repo]
        return original(class_reference, pretrained_model_name_or_path, *args, **kwargs)

    pinned._nemotron_dbx_original = original
    auto_factory.get_class_from_dynamic_module = pinned


def patch_post_init(vl_cls: type) -> bool:
    """Patch 6: transformers 5 expects `__init__` to end with `post_init()` (sets `all_tied_weights_keys`)."""
    if getattr(vl_cls, "_nemotron_dbx_post_init", False):
        return False
    orig_init = vl_cls.__init__

    @functools.wraps(orig_init)
    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()

    vl_cls.__init__ = init
    vl_cls._nemotron_dbx_post_init = True
    return True


def expected_summary_idxs(teachers: list[dict]) -> list[int]:
    return [i for i, t in enumerate(teachers) if t.get("use_summary", True)]


def fix_radio_summary_idxs(model) -> list[int] | None:
    """Patch 7: RADIO's `summary_idxs` buffer is not in the checkpoint; transformers 5 re-initialises it with
    garbage (-> CUDA device-side assert). Recompute it from the RADIO config."""
    import torch

    rm = model.vision_model.radio_model
    current = getattr(rm, "summary_idxs", None)
    if current is None:
        return None
    want = expected_summary_idxs(model.vision_model.config.args["teachers"])
    if current.tolist() != want:
        rm.summary_idxs = torch.tensor(want, dtype=torch.int64, device=current.device)
    return want


def patch_llm_output(model) -> bool:
    """Patch 8: the VL wrapper reads `outputs.past_key_values`; `NemotronHCausalLMOutput` only has `cache_params`."""
    module = sys.modules[type(model.language_model).__module__]
    out_cls = getattr(module, "NemotronHCausalLMOutput", None)
    if out_cls is None or hasattr(out_cls, "past_key_values"):
        return False
    out_cls.past_key_values = None
    return True


def remove_require_grads_hooks(model) -> int:
    """Patch 9: PEFT/transformers add input-require-grads hooks for checkpointing. With non-reentrant
    checkpointing they're unnecessary, and they turn `inputs_embeds` into a grad-requiring leaf that the VL
    wrapper then writes in place -> RuntimeError."""
    removed = 0
    for mod in model.modules():
        hook = getattr(mod, "_require_grads_hook", None)
        if hook is not None:
            hook.remove()
            del mod._require_grads_hook
            removed += 1
    return removed


def mamba_fast_path(model) -> bool:
    """True if the remote code found the fused Mamba kernels (patch 3); False means the OOM-prone torch path."""
    return bool(getattr(sys.modules[type(model.language_model).__module__], "is_fast_path_available", False))


def fix_image_rescale(image_processor) -> bool:
    """Patch 11: tiles must be in [0, 1] before the processor's mean/std normalization.

    The remote `NemotronNanoVLV2ImageProcessor` converts PIL images with `T.ToTensor()` (-> float [0, 1]) in a
    `_process_image` hook, then normalizes without rescaling. transformers 5 renamed that hook to `process_image`,
    whose default returns uint8 [0, 255], so the override is never called and the vision tower gets inputs
    255x too large. The model then learns to ignore the image. vLLM preprocesses separately and is unaffected.
    Wrap the instance's `process_image` to rescale uint8 output, which is exactly what `ToTensor` does.
    """
    cls = type(image_processor)
    if "_process_image" not in vars(cls) or "process_image" in vars(cls) or not hasattr(cls, "process_image"):
        return False
    if getattr(image_processor, "_nemotron_dbx_rescale", False):
        return False
    import torch

    original = image_processor.process_image

    @functools.wraps(original)
    def process_image(image, *args, **kwargs):
        out = original(image, *args, **kwargs)
        return out.float().div_(255.0) if isinstance(out, torch.Tensor) and out.dtype == torch.uint8 else out

    image_processor.process_image = process_image
    image_processor._nemotron_dbx_rescale = True
    return True
