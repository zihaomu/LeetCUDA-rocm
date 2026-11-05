#!/usr/bin/env bash
# Standalone HIP benchmarks; always build and run inside the locked foundation.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$project_root/scripts/rocm/run.sh" python -m leetcuda_rocm.build_bench "$@"
