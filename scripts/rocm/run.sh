#!/usr/bin/env bash
# Always enter the workspace's immutable ROCm 10 foundation first.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
launcher=${ROCM10_LAUNCHER:-/home/zihaomu/zihaomu_workspace/rocm10-launch.sh}
cd "$project_root"
if (( $# == 0 )); then set -- python -m leetcuda_rocm.smoke; fi
exec "$launcher" bash "$project_root/scripts/rocm/environment.sh" "$@"
