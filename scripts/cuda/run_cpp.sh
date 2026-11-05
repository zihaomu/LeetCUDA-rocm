#!/usr/bin/env bash
# Standalone C++ benchmark profile on nvidia-5090x2. No Torch is required.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cuda_cpp_toolkit=/usr/local/cuda-13.4
cuda_cpp_image=sha256:42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee
printf '%s  %s\n' \
  b6c72825585cb3d64d8cead9b43c165afed0f0f672319416261a0df76e6d600c "$cuda_cpp_toolkit/bin/nvcc" \
  6f6a701573ac9f5c2409708b3b700cb9b2a0f11b13c20468153c13b597b2d6d3 "$cuda_cpp_toolkit/bin/ptxas" |
  sha256sum --check --status
exec docker run --rm --device "nvidia.com/gpu=${LEETCUDA_CUDA_GPU:-0}" \
  --user "$(id -u):$(id -g)" --shm-size 2g \
  --mount "type=bind,src=$project_root,dst=/work" \
  --mount "type=bind,src=$cuda_cpp_toolkit,dst=/usr/local/cuda,readonly" \
  --workdir /work -e "LEETCUDA_CUDA_IMAGE=$cuda_cpp_image" \
  -e "LEETCUDA_CUDA_GPU=${LEETCUDA_CUDA_GPU:-0}" \
  --entrypoint bash "$cuda_cpp_image" -c 'exec "$@"' bash "$@"
