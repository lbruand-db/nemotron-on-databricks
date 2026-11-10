#!/usr/bin/env bash
# M0 spike (d): build the nemotron-ocr C++/CUDA extension in databricks_ai_v6, run inference,
# and persist the wheel for the Model Serving load test (OQ-9).
set -euo pipefail
cd "$CODE_SOURCE_PATH"
WORK=$(mktemp -d)
echo "nvcc: $(command -v nvcc || ls /usr/local/cuda/bin/nvcc)"; /usr/local/cuda/bin/nvcc --version | tail -2
gcc --version | head -1
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("nvidia/nemotron-ocr-v1", revision="${OCR_REVISION:-main}", local_dir="$WORK/nemotron-ocr-v1")
print("snapshot:", p)
PY
cd "$WORK/nemotron-ocr-v1/nemotron-ocr"
pip install -q hatchling 2>&1 | tail -2
t0=$(date +%s)
BUILD_CPP_FORCE=1 pip wheel --no-deps --no-build-isolation -w "$WORK/wheels" . 2>&1 | tail -15
echo "BUILD_SECONDS=$(( $(date +%s) - t0 ))"
ls -la "$WORK/wheels"
python - <<'PY'
import importlib.util
for m in ("pandas", "PIL", "sklearn", "shapely", "torchvision"):
    print(m, importlib.util.find_spec(m) is not None)
PY
pip install -q --no-deps "$WORK"/wheels/*.whl
python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('shapely') else 1)" || pip install -q "shapely==2.1.2"
cd "$WORK/nemotron-ocr-v1"
python "$CODE_SOURCE_PATH/recipes/spikes/ocr_build/probe_ocr.py" "$WORK/nemotron-ocr-v1"
DEST="$OUTPUT_ROOT/${MLFLOW_RUN_ID}"
mkdir -p "$DEST" && cp "$WORK"/wheels/*.whl "$DEST/" && echo "WHEEL_SAVED=$(ls "$DEST"/*.whl)"
