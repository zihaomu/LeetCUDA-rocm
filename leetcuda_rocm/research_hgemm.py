"""Unqualified FP16-accumulation candidates: distinguish rounding from port bugs.

These eleven original variants are deliberately excluded from the public module.
The public hgemm module exposes only the separately qualified BLAS interfaces.
An FP64 accuracy failure is recorded and makes this command exit nonzero.
"""

import hashlib
import json
from datetime import datetime, timezone

import torch

from .environment import ROOT, cache_key, verify
from .extra_checks import gemm_checks
from .loader import _load, _prepare


def main():
    torch.set_grad_enabled(False)
    identity = verify()
    sources = [
        ROOT / "kernels" / path
        for path in (
            "hgemm/pybind/hgemm.cc",
            "hgemm/naive/hgemm.cu",
            "hgemm/naive/hgemm_async.cu",
        )
    ]
    lib = _load(
        "hgemm_research",
        tuple(_prepare(path.read_text()) for path in sources),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )
    checks = []
    for seed in (20261102, 20261103, 20261104):
        torch.manual_seed(seed)
        a, b = (
            torch.randn(256, 256, device="cuda", dtype=torch.float16) * 0.25
            for _ in range(2)
        )
        fp64 = a.double() @ b.double()
        fused = torch.zeros_like(a)
        separate = torch.zeros_like(a)
        # Explicit IEEE half rounding after each accumulator update. This is
        # independent of the source kernel and does not use a reduction/matmul.
        for k in range(256):
            product = a[:, k, None].double() * b[k, None, :].double()
            fused = (fused.double() + product).half()
            separate = (separate.double() + product.half().double()).half()
        for name in sorted(n for n in dir(lib) if n.startswith("hgemm_")):
            out = torch.full_like(a, float("nan"))
            getattr(lib, name)(a, b, out)
            error = out.double() - fp64
            accuracy_passed = True
            try:
                torch.testing.assert_close(
                    out, fp64.half(), atol=0.01, rtol=0.01
                )
            except AssertionError:
                accuracy_passed = False
            row = {
                "function": name,
                "seed": seed,
                "shape_mnk": [256, 256, 256],
                "fp64_accuracy_passed": accuracy_passed,
                "atol": 0.01,
                "rtol": 0.01,
                "max_abs_error_fp64": error.abs().max().item(),
                "relative_l2_error_fp64": (error.norm() / fp64.norm()).item(),
                "matches_fp16_fused_emulation": torch.equal(out, fused),
                "matches_fp16_separate_emulation": torch.equal(out, separate),
            }
            checks.append(row)
            print(json.dumps(row), flush=True)
    torch.manual_seed(20261103)
    boundary = gemm_checks("hgemm", lib)
    failed = sorted(
        {row["function"] for row in checks if not row["fp64_accuracy_passed"]}
        | {row["function"] for row in boundary if not row["passed"]}
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": identity,
        "qualified": False,
        "accuracy_passed": not failed,
        "failed_functions": failed,
        "checks": checks,
        "boundary_checks": boundary,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sources,
                ROOT / "leetcuda_rocm/research_hgemm.py",
                ROOT / "leetcuda_rocm/extra_checks.py",
                ROOT / "leetcuda_rocm/loader.py",
            ]
        },
        "binary_sha256": hashlib.sha256(
            open(lib.__file__, "rb").read()
        ).hexdigest(),
    }
    output = ROOT / "artifacts/rocm/continuation/hgemm_research.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Research only; {len(failed)} functions failed FP64 accuracy gates: {output}",
        flush=True,
    )
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
