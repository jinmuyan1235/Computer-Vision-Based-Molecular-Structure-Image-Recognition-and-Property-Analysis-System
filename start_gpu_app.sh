#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MOLECULE_VISION_GPU_VENV:-$HOME/.venvs/molecule-vision-gpu}"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

export OCSR_DEVICE="${OCSR_DEVICE:-cuda:0}"
export DECIMER_DEVICE="${DECIMER_DEVICE:-gpu}"
export OCSR_STRICT_MODE="${OCSR_STRICT_MODE:-true}"
export DECIMER_STRICT_MODE="${DECIMER_STRICT_MODE:-true}"
export MOLSCRIBE_ISOLATED_SUBPROCESS="${MOLSCRIBE_ISOLATED_SUBPROCESS:-true}"
export DECIMER_ISOLATED_SUBPROCESS="${DECIMER_ISOLATED_SUBPROCESS:-true}"
export OCSR_GPU_REQUIRED="${OCSR_GPU_REQUIRED:-true}"
export OCSR_GPU_MAX_CONCURRENT_INFERENCE="${OCSR_GPU_MAX_CONCURRENT_INFERENCE:-1}"
export OCSR_ENSEMBLE_PARALLEL="${OCSR_ENSEMBLE_PARALLEL:-false}"
export MOLSCRIBE_MODEL_PATH="${MOLSCRIBE_MODEL_PATH:-models/molscribe/swin_base_char_aux_1m.pth}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/curand/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

python scripts/verify_gpu_environment.py --no-strict
python -m streamlit run app.py \
  --server.runOnSave=false \
  --server.fileWatcherType=none \
  --server.websocketPingInterval=20
