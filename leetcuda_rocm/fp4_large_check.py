"""Validate automatic wide-tile dispatch at 4096 and with padded edge tiles."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .environment import ROOT, verify
from .fp4 import MODES, fp4_gemm, fp4_gemm_prepacked
from .fp4_check import gpu_quantize, quantize_reference
from .fp8_check import ATOL, RTOL, error_metrics


def check_shape(shape, checks):
    m, n, k = shape
    torch.manual_seed(20261103)
    a = (torch.randn(m, k, device="cuda") * 0.25).bfloat16()
    b = (torch.randn(k, n, device="cuda") * 0.25).bfloat16()
    for mode, idx in MODES.items():
        l1a, l1b = bool(idx & 1), bool(idx & 2)
        aq, af, sa, ad = quantize_reference(a, l1a)
        bq, bf, sb, bd = quantize_reference(b.T, l1b)
        ga, gf, gs, weight = gpu_quantize(a, b, l1a, l1b)
        for got, expected in (
            (ga, aq),
            (gf, af),
            (weight.data, bq),
            (weight.block_scales, bf),
        ):
            np.testing.assert_array_equal(got.cpu().numpy(), expected)
        if l1a:
            np.testing.assert_allclose(gs.cpu().numpy(), sa, atol=0, rtol=2e-7)
        if l1b:
            np.testing.assert_allclose(
                weight.scales.cpu().numpy(), sb, atol=0, rtol=2e-7
            )
        checks.append(
            {
                "kind": "quantization",
                "shape_mnk": list(shape),
                "mode": mode,
                "passed": True,
            }
        )
        expected = (
            (torch.from_numpy(ad).cuda() @ torch.from_numpy(bd).cuda().T)
            * torch.from_numpy(sa).cuda().double()[:, None]
            * torch.from_numpy(sb).cuda().double()[None, :]
        ).bfloat16()
        for ws in (False, True):
            for path in ("full", "offline_b"):
                out = (
                    fp4_gemm(a, b, mode=mode, use_ws=ws, tile="auto")
                    if path == "full"
                    else fp4_gemm_prepacked(
                        a, weight, level1_a=l1a, use_ws=ws, tile="auto"
                    )
                )
                torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
                checks.append(
                    {
                        "kind": "gemm",
                        "shape_mnk": list(shape),
                        "mode": mode,
                        "path": path,
                        "use_ws": ws,
                        "tile": "auto",
                        "selected_tile": "wide",
                        "reference": "independent CPU quantization, full FP64 GEMM, BF16 output",
                        **error_metrics(out, expected),
                        "passed": True,
                    }
                )
        print(
            json.dumps(
                {"shape_mnk": list(shape), "mode": mode, "passed": True}
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.set_grad_enabled(False)
    sources = (
        "kernels/rocm/fp4_gemm_hip.cuh",
        "leetcuda_rocm/native/fp4_bindings.cpp",
        "leetcuda_rocm/fp4.py",
        "leetcuda_rocm/fp4_check.py",
        "leetcuda_rocm/fp8_check.py",
        "leetcuda_rocm/fp4_large_check.py",
        "leetcuda_rocm/loader.py",
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": verify(),
        "logical_device": args.device,
        "source_sha256": {
            s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()
            for s in sources
        },
        "checks": [],
        "passed": False,
        "numeric_gate": {"atol": ATOL, "rtol": RTOL},
        "stream": "non-default",
    }
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            for shape in ((4096, 4096, 4096), (4097, 4104, 4160)):
                check_shape(shape, report["checks"])
        stream.synchronize()
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(report["checks"]),
                "error": report.get("error"),
            }
        )
    )
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
