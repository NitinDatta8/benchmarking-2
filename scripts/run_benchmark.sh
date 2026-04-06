#!/usr/bin/env bash
# Quantize (if needed) then benchmark for a given GPU type.
#
# Usage: bash scripts/run_benchmark.sh [GPU_TYPE] [SKIP_QUANT] [METHOD] [PROFILE]
#   GPU_TYPE:   A100_SXM | L4 | RTX_5090 (default: A100_SXM)
#   SKIP_QUANT: true to skip quantization (default: false)
#   METHOD:     run only this method (default: all for GPU type)
#   PROFILE:    true to enable vLLM torch profiler (default: false)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env (OPENAI_API_KEY, HF_TOKEN, etc.)
if [ -f "$REPO_DIR/.env" ]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi

# Activate venv
VENV_DIR="/workspace/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi

GPU_TYPE="${1:-A100_SXM}"
SKIP_QUANT="${2:-false}"
METHOD_FILTER="${3:-}"
PROFILE="${4:-false}"
BASE_MODEL="/workspace/models/base"
CONFIG="scripts/benchmark_config.yaml"
PROFILE_FLAG=""
if [ "$PROFILE" = "true" ]; then
  PROFILE_FLAG="--profile"
fi

cd "$REPO_DIR"

# method:quant_method:output_path (none = no quant step needed)
case "$GPU_TYPE" in
  A100_SXM) ENTRIES=(
              "baseline_fp16:none:$BASE_MODEL"
              "gptq_w4a16:gptq_w4a16:/workspace/models/gptq_w4a16"
              "awq_w4a16:awq_w4a16:/workspace/models/awq_w4a16"
              "fp8_dynamic:fp8_dynamic:/workspace/models/fp8_dynamic"
            ) ;;
  L4)       ENTRIES=(
              "baseline_fp16:none:$BASE_MODEL"
              "gptq_w4a16:gptq_w4a16:/workspace/models/gptq_w4a16"
              "awq_w4a16:awq_w4a16:/workspace/models/awq_w4a16"
              "fp8_dynamic:fp8_dynamic:/workspace/models/fp8_dynamic"
            ) ;;
  RTX_5090) ENTRIES=(
              "baseline_fp16:none:$BASE_MODEL"
              "gptq_w4a16:gptq_w4a16:/workspace/models/gptq_w4a16"
              "awq_w4a16:awq_w4a16:/workspace/models/awq_w4a16"
              "fp8_dynamic:fp8_dynamic:/workspace/models/fp8_dynamic"
              "nvfp4:nvfp4:/workspace/models/nvfp4"
            ) ;;
  *)        echo "Unknown GPU_TYPE: $GPU_TYPE"; exit 1 ;;
esac

for ENTRY in "${ENTRIES[@]}"; do
  METHOD=$(echo "$ENTRY" | cut -d: -f1)
  QUANT_METHOD=$(echo "$ENTRY" | cut -d: -f2)
  OUTPUT_PATH=$(echo "$ENTRY" | cut -d: -f3)

  [ -n "$METHOD_FILTER" ] && [ "$METHOD" != "$METHOD_FILTER" ] && continue

  echo "--- $METHOD ---"

  # Quantize if needed
  if [ "$QUANT_METHOD" != "none" ] && [ "$SKIP_QUANT" != "true" ]; then
    if [ -d "$OUTPUT_PATH" ] && [ "$(ls -A "$OUTPUT_PATH")" ]; then
      echo "Quantized model exists at $OUTPUT_PATH, skipping."
    else
      # Try downloading pre-quantized model from HF first
      HF_REPO=$(python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg['methods'].get('$QUANT_METHOD', {}).get('hf_repo_id') or '')
")
      if [ -n "$HF_REPO" ]; then
        echo "Downloading pre-quantized model from $HF_REPO ..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', local_dir='$OUTPUT_PATH')
"
      else
        python scripts/quantize.py --method "$QUANT_METHOD" --base_model "$BASE_MODEL" --output_path "$OUTPUT_PATH" --config "$CONFIG"
      fi
    fi
  fi

  # Benchmark
  python benchmark/runner.py --method "$METHOD" --gpu "$GPU_TYPE" --config "$CONFIG" $PROFILE_FLAG
done

echo "Done. Results in results/"
