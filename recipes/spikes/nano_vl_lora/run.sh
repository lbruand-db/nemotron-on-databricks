#!/usr/bin/env bash
# Phase A: AIR's transformers. Phase B (only if A fails): the model card's transformers pin.
# mamba-ssm is installed without its CUDA extension: the Nano VL remote code only imports
# mamba_ssm.ops.triton.* (Triton kernels); without causal_conv1d it uses the torch Mamba path.
set -uo pipefail
cd "$CODE_SOURCE_PATH"
P=recipes/spikes/nano_vl_lora/probe_lora.py
LOGDIR=$(mktemp -d)
t0=$(date +%s)
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -q --no-build-isolation --no-deps "mamba-ssm==${MAMBA_SSM_VERSION}" einops timm open_clip_torch ftfy regex 2>&1 | tail -5
python -c "import open_clip, timm; print('open_clip', open_clip.__version__, 'timm', timm.__version__)"
# mamba_ssm/__init__.py eagerly imports the Mamba-1 CUDA extension (selective_scan_cuda), which a
# CUDA-less build lacks. Nano VL only needs the Mamba-2 Triton ops, so blank the package __init__.
MAMBA_INIT=$(python -c "import importlib.util as u; print(u.find_spec('mamba_ssm').submodule_search_locations[0])")/__init__.py
echo "__version__ = '${MAMBA_SSM_VERSION}'" > "$MAMBA_INIT"
echo "mamba-ssm install took $(( $(date +%s) - t0 ))s"
# causal-conv1d CUDA extension: required for the fused Mamba-2 path (the torch fallback materialises
# a ~17 GiB tensor per layer at 2k tokens). Build once for sm_90 and cache the wheel on the Volume.
CC_CACHE="$WHEEL_CACHE/causal_conv1d/$(python -c 'import torch; print(torch.__version__.replace("+", "_"))')"
if ls "$CC_CACHE"/*.whl >/dev/null 2>&1; then
  echo "causal-conv1d: using cached wheel $(ls "$CC_CACHE"/*.whl)"
else
  t1=$(date +%s)
  mkdir -p "$CC_CACHE" /tmp/ccwheel
  CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16 \
    pip wheel --no-deps --no-build-isolation -w /tmp/ccwheel "causal-conv1d==${CAUSAL_CONV1D_VERSION}" 2>&1 | tail -5
  cp /tmp/ccwheel/*.whl "$CC_CACHE/"
  echo "causal-conv1d build took $(( $(date +%s) - t1 ))s"
fi
pip install -q --no-deps "$CC_CACHE"/*.whl
python -c "import causal_conv1d, causal_conv1d_cuda; print('causal_conv1d', causal_conv1d.__version__, 'CUDA ext OK')"
python -c "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; print('mamba_ssm triton ops import OK')"
echo "=== PHASE A ==="
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node=1 "$P" | tee "$LOGDIR/phase_a.log"
if grep -q '"status": "ok"' "$LOGDIR/phase_a.log"; then exit 0; fi
echo "=== PHASE B: transformers==4.53.3 ==="
pip install -q "transformers==4.53.3" 2>&1 | tail -5
python -c "import open_clip; print('open_clip', open_clip.__version__)" 2>&1 | tail -3
torchrun --nproc-per-node=1 "$P"
