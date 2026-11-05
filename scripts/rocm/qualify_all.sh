#!/usr/bin/env bash
# Select two permitted GPUs explicitly, for example HIP_VISIBLE_DEVICES=6,2.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$project_root/scripts/rocm/run.sh" python -m leetcuda_rocm.qualify_all "$@"
