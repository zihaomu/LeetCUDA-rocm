#!/usr/bin/env bash
# Select the physical card through HIP_VISIBLE_DEVICES; suites use logical GPU0.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$project_root/scripts/rocm/run.sh" python -m leetcuda_rocm.build_all_bench "$@"
