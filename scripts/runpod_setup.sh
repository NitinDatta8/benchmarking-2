#!/usr/bin/env bash
# Setup a RunPod pod for quantization benchmarking.
# Python and PyTorch are pre-installed. This script installs vllm + llmcompressor
# and downloads the base model.
#
# Usage: bash scripts/runpod_setup.sh [GPU_TYPE]
#   GPU_TYPE: A100_SXM | L4 | RTX_5090 (default: A100_SXM)
#
# Reads HF_TOKEN and OPENAI_API_KEY from .env in the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env if present
if [ -f "$REPO_DIR/.env" ]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
  echo "Loaded .env from $REPO_DIR/.env"
else
  echo "WARNING: No .env file found at $REPO_DIR/.env — copy .env.example and fill in your keys."
fi

GPU_TYPE="${1:-A100_SXM}"
BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
BASE_MODEL_LOCAL="/workspace/models/base"

echo "=== RunPod Setup (GPU: $GPU_TYPE) ==="

# Validate GPU type
case "$GPU_TYPE" in
  A100_SXM|L4|RTX_5090) ;;
  *) echo "Unknown GPU: $GPU_TYPE. Valid: A100_SXM, L4, RTX_5090"; exit 1 ;;
esac

# GPU check
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Install uv if not present
pip install uv --quiet --break-system-packages 2>/dev/null || true

# Create venv and install packages
VENV_DIR="/workspace/.venv"
if [ ! -d "$VENV_DIR" ]; then
  uv venv "$VENV_DIR" --python python3
fi
source "$VENV_DIR/bin/activate"

uv pip install vllm llmcompressor datasets transformers pyyaml numpy loguru openai python-dotenv --quiet

# Print versions
echo "Installed versions:"
uv pip list | grep -E "vllm|torch |transformers|llmcompressor"

# Download base model
export HF_HOME=/workspace/.hf_cache
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "WARNING: HF_TOKEN not set (check .env). Gated repos will fail."
fi

mkdir -p "$(dirname "$BASE_MODEL_LOCAL")"
python3 -c "
from huggingface_hub import snapshot_download
import os, shutil

cache_path = snapshot_download(
    repo_id='$BASE_MODEL_ID',
    ignore_patterns=['*.msgpack', '*.h5', 'original/*', 'consolidated.safetensors'],
)
target = '$BASE_MODEL_LOCAL'
if os.path.islink(target):
    os.remove(target)
elif os.path.isdir(target):
    shutil.rmtree(target)
os.symlink(cache_path, target)
print(f'Model ready: {target} -> {cache_path}')
"

# Smoke test
python3 -c "
import torch, vllm
print(f'torch={torch.__version__} vllm={vllm.__version__} cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
"

echo ""
echo "Setup complete. Next: bash scripts/run_benchmark.sh $GPU_TYPE"
