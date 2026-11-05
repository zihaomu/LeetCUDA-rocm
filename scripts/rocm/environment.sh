#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
# Verify the project lock before importing/compiling native extensions.
identity=$(python -m leetcuda_rocm.environment --cache-key)
export TORCH_EXTENSIONS_DIR="$project_root/.cache/rocm10/$identity/torch"
export TRITON_CACHE_DIR="$project_root/.cache/rocm10/$identity/triton"
export PYTORCH_ROCM_ARCH=gfx1201
export MAX_JOBS=${MAX_JOBS:-2}
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
exec "$@"
