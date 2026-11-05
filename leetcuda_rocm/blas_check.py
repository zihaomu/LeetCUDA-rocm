"""Validate and benchmark the original cuBLAS APIs implemented with hipBLAS."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .blas_checks import blas_checks
from .environment import ROOT, verify
from .loader import load, source_paths
from .matrix_attention_check import timed_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.set_grad_enabled(False)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": verify(),
        "logical_device": args.device,
        "checks": [],
        "performance": [],
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                *source_paths("sgemm"),
                *source_paths("hgemm"),
                *sorted((ROOT / "leetcuda_rocm").glob("*.py")),
            )
        },
        "method": "HIP graphs; reused buffers; 5 alternating 40ms samples, median",
        "passed": False,
    }
    for module in ("sgemm", "hgemm"):
        lib = load(module)
        report["checks"].extend(blas_checks(module, lib))
        dtype = torch.float16 if module == "hgemm" else torch.float32
        tol = 0.01 if dtype == torch.float16 else 2e-4
        for n in (256, 1024, 2048):
            a, b = (
                torch.randn(n, n, device="cuda", dtype=dtype) * 0.25
                for _ in range(2)
            )
            out, ref = torch.empty_like(a), torch.empty_like(a)
            for name in sorted(
                n
                for n in dir(lib)
                if n.startswith(("sgemm_cublas", "hgemm_cublas"))
            ):
                packed = (
                    b.T.contiguous().view_as(b) if name.endswith("_tn") else b
                )
                ours = lambda: getattr(lib, name)(a, packed, out)
                baseline = lambda: torch.mm(a, b, out=ref)
                ours()
                baseline()
                torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
                timing = timed_pair(ours, baseline)
                record = {
                    "function": name,
                    "shape_mnk": [n, n, n],
                    "dtype": str(dtype),
                    "baseline": "torch.mm(out=), default ROCm BLAS backend",
                    "b_packing_timed": False,
                    "correctness_passed": True,
                    **timing,
                }
                report["performance"].append(record)
                print(json.dumps(record), flush=True)
    report["passed"] = all(c["passed"] for c in report["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
