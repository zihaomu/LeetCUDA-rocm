#!/usr/bin/env bash
# Run on nvidia-5090x2, from the dedicated LeetCUDA comparison directory.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cuda_image=sha256:42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee
exec docker run --rm --device "nvidia.com/gpu=${LEETCUDA_CUDA_GPU:-0}" \
  --user "$(id -u):$(id -g)" --shm-size 2g \
  --mount "type=bind,src=$project_root,dst=/work" --workdir /work \
  -e "LEETCUDA_CUDA_IMAGE=$cuda_image" -e PYTHONPATH=/work \
  -e TORCH_EXTENSIONS_DIR=/work/.cache/cuda_comparison \
  -e TRITON_CACHE_DIR=/work/.cache/triton -e MAX_JOBS=2 \
  --entrypoint bash "$cuda_image" -c 'exec "$@"' bash "$@"
