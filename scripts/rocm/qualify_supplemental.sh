#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# The locked launcher must wrap this driver; it does not select physical GPUs.
python -m leetcuda_rocm.supplemental_check run --backend hip --output artifacts/rocm/continuation_all/supplemental/hip_formatted.json
python -m leetcuda_rocm.cute_dsl_check run --backend hip --output artifacts/rocm/continuation_all/cute_dsl/hip_formatted.json
python -m leetcuda_rocm.swizzle_examples_check run --backend hip --output artifacts/rocm/continuation_all/swizzle/hip_formatted.json
for suite in supplemental cute_dsl swizzle; do
    python -m leetcuda_rocm.supplemental_isa --suite "$suite" --output "artifacts/rocm/continuation_all/$suite/formatted/resources.json"
done
