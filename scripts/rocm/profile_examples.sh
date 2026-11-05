#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
output=${1:-artifacts/rocm/profile_examples}
mkdir -p "$output"
profile_mode=${2:-trace}
case "$profile_mode" in
  trace) profile_options=(--kernel-trace --stats) ;;
  counters) profile_options=(--pmc SQ_WAVES --kernel-include-regex "^(elementwise_add|relu)_.*") ;;
  *) echo "mode must be trace or counters" >&2; exit 2 ;;
esac
# Select and inspect the GPU before invocation, as for all ROCm experiments.
exec scripts/rocm/run.sh env PYTHONPATH="$project_root" rocprofv3 \
  "${profile_options[@]}" --output-format csv --output-directory "$output" \
  -- python -m leetcuda_rocm.profile_examples --output "$output/validation.json"
