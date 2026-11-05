"""Check an exact NVFP4 decode route before porting its block-scaled GEMM."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np

from .environment import ROOT, verify


def main():
    identity = verify()
    output = ROOT / "artifacts/rocm/fp4"
    output.mkdir(parents=True, exist_ok=True)
    # Every finite NVFP4 E2M1 value times every nonnegative finite E4M3 scale.
    values4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
    values4 = np.concatenate((values4, -values4))
    codes = np.arange(127)
    values8 = np.where(
        codes < 8, codes * 2.0**-9, (8 + codes % 8) * np.exp2(codes // 8 - 10)
    )
    exact = values4[:, None] * values8[None, :]
    decoded = exact.astype(np.float16).astype(np.float64)
    equal = (decoded == exact) & (np.signbit(decoded) == np.signbit(exact))
    probe = output / "conversion_probe.cpp"
    probe.write_text(
        """#include <hip/hip_runtime.h>
using V2 = float __attribute__((ext_vector_type(2)));
__global__ void probe(V2 *out, unsigned int *x, float *scale) {
  out[threadIdx.x] = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(
      x[threadIdx.x], scale[threadIdx.x], 0);
}
"""
    )
    command = [
        "/opt/rocm/bin/hipcc",
        "--offload-arch=gfx1201",
        "-O3",
        "-c",
        str(probe),
        "-o",
        str(output / "conversion_probe.o"),
    ]
    proc = subprocess.run(command, text=True, capture_output=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": identity,
        "finite_code_scale_pairs": int(equal.size),
        "exact_fp16_decode_pairs": int(equal.sum()),
        "signed_zero_preserved": bool(
            np.array_equal(np.signbit(exact), np.signbit(decoded))
        ),
        "maximum_magnitude": float(np.abs(exact).max()),
        "minimum_nonzero_magnitude": float(np.abs(exact[exact != 0]).min()),
        "conversion_probe": {
            "command": command,
            "returncode": proc.returncode,
            "stderr": proc.stderr,
            "stdout": proc.stdout,
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "leetcuda_rocm/research_fp4.py", probe)
        },
        "qualified_public_api": False,
        "conclusion": "All finite E2M1 x nonnegative E4M3 products can be expanded exactly to FP16; this does not validate quantization, packed layout, GEMM or performance.",
    }
    (output / "research.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pairs": int(equal.size),
                "exact": int(equal.sum()),
                "conversion_compile_status": proc.returncode,
            }
        )
    )
    assert bool(equal.all())


if __name__ == "__main__":
    main()
