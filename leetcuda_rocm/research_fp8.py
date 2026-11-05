"""Probe true FP8/BF8 WMMA before porting the scaled FP8 GEMM pipeline."""

import hashlib
import json
import re
from datetime import datetime, timezone

import torch
import triton
import triton.language as tl

from .environment import ROOT, verify


@triton.jit
def fp8_probe_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
):
    row = tl.program_id(0) * 64 + tl.arange(0, 64)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    red = tl.arange(0, 64)
    acc = tl.full((64, 64), 0, tl.float32)
    for base in range(tl.cdiv(K, 64)):
        kk = base * 64 + red
        # Load encoded bytes: masked zero needs no unsupported int->FP8 cast.
        av = tl.load(
            A + row[:, None] * K + kk[None, :],
            (row[:, None] < M) & (kk[None, :] < K),
            0,
        ).to(FORMAT, bitcast=True)
        bv = tl.load(
            B + kk[:, None] * N + col[None, :],
            (kk[:, None] < K) & (col[None, :] < N),
            0,
        ).to(FORMAT, bitcast=True)
        acc = tl.dot(av, bv, acc)
    tl.store(
        C + row[:, None] * N + col[None, :],
        acc,
        (row[:, None] < M) & (col[None, :] < N),
    )


def main():
    identity = verify()
    torch.manual_seed(20261103)
    output = ROOT / "artifacts/rocm/fp8"
    output.mkdir(parents=True, exist_ok=True)
    checks, isa = [], []
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        for m, n, k in ((17, 33, 19), (128, 256, 256), (512, 640, 320)):
            name = f"{str(dtype).split('.')[-1]}_{m}_{n}_{k}"
            row = {"dtype": str(dtype), "shape_mnk": [m, n, k]}
            try:
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    a = (torch.randn(m, k, device="cuda") * 0.5).to(dtype)
                    b = (torch.randn(k, n, device="cuda") * 0.5).to(dtype)
                    out = torch.full((m, n), float("nan"), device="cuda")
                    compiled = fp8_probe_kernel[
                        (triton.cdiv(m, 64), triton.cdiv(n, 64))
                    ](
                        a.view(torch.uint8),
                        b.view(torch.uint8),
                        out,
                        m,
                        n,
                        k,
                        FORMAT=(
                            tl.float8e4nv
                            if dtype == torch.float8_e4m3fn
                            else tl.float8e5
                        ),
                        num_warps=4,
                        num_stages=2,
                    )
                    expected = a.double() @ b.double()
                    torch.testing.assert_close(
                        out, expected.float(), atol=2e-4, rtol=2e-4
                    )
                    assembly = compiled.asm["amdgcn"]
                    instructions = sorted(
                        set(re.findall(r"v_wmma_[a-z0-9_]+", assembly))
                    )
                    suffix = (
                        "fp8_fp8" if dtype == torch.float8_e4m3fn else "bf8_bf8"
                    )
                    assert any(
                        suffix in inst for inst in instructions
                    ), instructions
                    path = output / f"{name}.amdgcn"
                    path.write_text(assembly)
                    isa.append(
                        {
                            "name": name,
                            "instructions": instructions,
                            "assembly": str(path.relative_to(ROOT)),
                            "assembly_sha256": hashlib.sha256(
                                assembly.encode()
                            ).hexdigest(),
                            "hsaco_sha256": hashlib.sha256(
                                compiled.asm["hsaco"]
                            ).hexdigest(),
                            "registers": compiled.n_regs,
                            "spills": compiled.n_spills,
                        }
                    )
                    row.update(
                        passed=True,
                        max_abs_error=(out.double() - expected)
                        .abs()
                        .max()
                        .item(),
                        atol=2e-4,
                        rtol=2e-4,
                        stream="non-default",
                    )
                stream.synchronize()
            except Exception as exc:
                row.update(passed=False, error=repr(exc))
            checks.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": identity,
        "checks": checks,
        "isa": isa,
        "capability_probe_passed": all(c["passed"] for c in checks),
        "qualified_public_api": False,
        "scope": "Already quantized FP8/BF8 operands, FP32 accumulation/output; scale/quantization API and C++ benchmarks remain to be ported.",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "leetcuda_rocm/research_fp8.py",)
        },
    }
    (output / "research.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["capability_probe_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
