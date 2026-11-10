#!/usr/bin/env bash
# Phase A: AIR's transformers. Phase B (only if A fails): the model card's transformers pin.
# mamba-ssm is installed without its CUDA extension: the Nano VL remote code only imports
# mamba_ssm.ops.triton.* (Triton kernels); without causal_conv1d it uses the torch Mamba path.
set -uo pipefail
cd "$CODE_SOURCE_PATH"
P=recipes/spikes/nano_vl_lora/probe_lora.py
LOGDIR=$(mktemp -d)
t0=$(date +%s)
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -q --no-build-isolation --no-deps "mamba-ssm==2.2.5" einops timm open_clip_torch 2>&1 | tail -5
echo "mamba-ssm install took $(( $(date +%s) - t0 ))s"
python -c "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; print('mamba_ssm triton ops import OK')"
echo "=== PHASE A ==="
torchrun --nproc-per-node=1 "$P" | tee "$LOGDIR/phase_a.log"
if grep -q '"status": "ok"' "$LOGDIR/phase_a.log"; then exit 0; fi
echo "=== PHASE B: transformers==4.53.3 ==="
pip install -q "transformers==4.53.3" 2>&1 | tail -5
torchrun --nproc-per-node=1 "$P"
