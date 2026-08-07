#!/usr/bin/env bash
set -euo pipefail

# Edit these two paths.
INPUT_DIR="${INPUT_DIR:-/path/to/GLM-5.2-LoRA-Merged-BF16}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/GLM-5.2-LoRA-Merged-FP8-BLOCK}"

# balanced is the recommended first run.
PROFILE="${PROFILE:-balanced}"

# Restrict to one node's visible GPUs. llmcompressor 0.12.0 distributes shards
# across all devices exposed by CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/inspect_glm52_checkpoint.py" \
  "${INPUT_DIR}" \
  --profile "${PROFILE}"

python "${SCRIPT_DIR}/quantize_glm52_fp8.py" \
  --input "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}" \
  --profile "${PROFILE}" \
  --devices auto

python "${SCRIPT_DIR}/verify_glm52_fp8.py" \
  --source "${INPUT_DIR}" \
  --quantized "${OUTPUT_DIR}" \
  --profile "${PROFILE}"
