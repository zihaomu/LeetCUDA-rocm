"""Independent correctness, scheduling and performance gates for AMD SGEMM prefetch."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from .benchmark import graph_pair
from .environment import ROOT, cache_key, verify
from .fp8_check import error_metrics
from .isa_objects import hip_isa_objects
from .loader import _load, _prepare, load, source_paths

VARIANTS = [(64, 64, 4), (128, 128, 8), (128, 256, 16)]
ATOL = RTOL = 2e-4


def name(tn):
    return f"sgemm_t_8x{tn}_sliced_k16_f32x4_bcf_dbuf_async"


def private_module(identity):
    bindings = "#include <torch/extension.h>\n"
    names = [
        n
        for _, _, tn in VARIANTS
        for n in (name(tn), name(tn).removesuffix("_async"))
    ]
    for n in names:
        bindings += f"void {n}(torch::Tensor,torch::Tensor,torch::Tensor);\n"
    bindings += "PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n"
    for n in names:
        bindings += f'm.def("{n}",&{n});\n'
    bindings += "}\n"
    return _load(
        "sgemm_async_research",
        (
            _prepare((ROOT / "kernels/sgemm/sgemm_async.cu").read_text()),
            bindings,
        ),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )


def check_output(fn, sync, a, b, checks, meta):
    expected = a.double() @ b.double()
    out = torch.full(
        (a.shape[0], b.shape[1]), float("nan"), device=a.device, dtype=a.dtype
    )
    prior = torch.empty_like(out)
    fn(a, b, out)
    sync(a, b, prior)
    torch.testing.assert_close(out, expected.float(), atol=ATOL, rtol=RTOL)
    assert torch.equal(
        out, prior
    ), "FP32 FMA order differs from the original synchronous variant"
    checks.append(
        {
            **meta,
            "shape_mnk": [a.shape[0], b.shape[1], a.shape[1]],
            "passed": True,
            "bitwise_matches_synchronous_variant": True,
            **error_metrics(out, expected),
        }
    )


def numerical(lib, checks):
    for bm, bn, tn in VARIANTS:
        fn, sync = getattr(lib, name(tn)), getattr(
            lib, name(tn).removesuffix("_async")
        )
        for seed in (20261103, 17, 93):
            torch.manual_seed(seed)
            for rows, cols, tiles in (
                (1, 1, 1),
                (1, 1, 2),
                (2, 2, 3),
                (3, 1, 5),
                (1, 3, 8),
                (2, 3, 17),
                (1, 1, 65),
                (1, 1, 256),
            ):
                m, n, k = bm * rows, bn * cols, tiles * 16
                a = torch.randn(m, k, device="cuda") * 0.25
                b = torch.randn(k, n, device="cuda") * 0.25
                check_output(
                    fn,
                    sync,
                    a,
                    b,
                    checks,
                    {
                        "function": name(tn),
                        "kind": "random",
                        "seed": seed,
                        "k_tiles": tiles,
                    },
                )
        for pattern in ("zero", "cancellation", "range"):
            a = torch.randn(bm * 2, 80, device="cuda") * 0.25
            b = torch.randn(80, bn * 2, device="cuda") * 0.25
            if pattern == "zero":
                a.zero_()
            elif pattern == "cancellation":
                a[:, 1::2] = a[:, ::2]
                b[1::2] = -b[::2]
            else:
                a[:, ::2] *= 64
                b[::2] /= 64
                a[:, 1::2] /= 64
                b[1::2] *= 64
            check_output(
                fn, sync, a, b, checks, {"function": name(tn), "kind": pattern}
            )


def contracts(lib, checks, guard_device):
    for bm, bn, tn in VARIANTS:
        fn = getattr(lib, name(tn))
        a = torch.ones(bm, 32, device="cuda")
        b = torch.ones(32, bn, device="cuda")
        out = torch.empty(bm, bn, device="cuda")

        def unaligned(t):
            return torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype)[
                1:
            ].view_as(t)

        bad = {
            "cpu": lambda: fn(a.cpu(), b, out),
            "input_dtype": lambda: fn(a.half(), b, out),
            "output_dtype": lambda: fn(a, b, out.double()),
            "rank": lambda: fn(a.flatten(), b, out),
            "strided": lambda: fn(a.T, b, out),
            "gradient": lambda: fn(a.clone().requires_grad_(), b, out),
            "K_mismatch": lambda: fn(a, b[:16], out),
            "output_shape": lambda: fn(a, b, out[:1]),
            "unaligned_a": lambda: fn(unaligned(a), b, out),
            "unaligned_b": lambda: fn(a, unaligned(b), out),
            "unaligned_out": lambda: fn(a, b, unaligned(out)),
            "M_tile": lambda: fn(a[: bm - 1], b, out[: bm - 1]),
            "N_tile": lambda: fn(
                a, b[:, : bn - 4].contiguous(), out[:, : bn - 4].contiguous()
            ),
            "K_tile": lambda: fn(a[:, :28].contiguous(), b[:28], out),
            "zero_M": lambda: fn(a[:0], b, out[:0]),
            "zero_K": lambda: fn(a[:, :0].contiguous(), b[:0], out),
            "zero_N": lambda: fn(
                a, b[:, :0].contiguous(), out[:, :0].contiguous()
            ),
        }
        overlap = torch.ones(bm, bn, device="cuda")
        bad["output_alias"] = lambda: fn(
            overlap, torch.eye(bn, device="cuda"), overlap
        )
        for label, call in bad.items():
            try:
                call()
            except (RuntimeError, ValueError) as exc:
                checks.append(
                    {
                        "function": name(tn),
                        "kind": "rejection",
                        "case": label,
                        "passed": True,
                        "rejection": str(exc),
                    }
                )
            else:
                raise AssertionError(f"{name(tn)} accepted {label}")
        if guard_device is not None:
            old = torch.cuda.current_device()
            torch.cuda.set_device(guard_device)
            try:
                fn(a, b, out)
                assert torch.cuda.current_device() == guard_device
                torch.testing.assert_close(
                    out, torch.full_like(out, 32), atol=0, rtol=0
                )
                checks.append(
                    {
                        "function": name(tn),
                        "kind": "device_guard",
                        "input_device": old,
                        "current_device": guard_device,
                        "passed": True,
                    }
                )
                try:
                    fn(a, b.to(f"cuda:{guard_device}"), out)
                except RuntimeError:
                    checks.append(
                        {
                            "function": name(tn),
                            "kind": "mixed_device_rejection",
                            "passed": True,
                        }
                    )
                else:
                    raise AssertionError("mixed devices accepted")
            finally:
                torch.cuda.set_device(old)


def concurrent(lib, checks):
    streams = [torch.cuda.Stream() for _ in range(2)]
    current = torch.cuda.current_stream()
    for s in streams:
        s.wait_stream(current)
    pending = []
    for bm, bn, tn in VARIANTS:
        for iteration in range(24):
            s = streams[iteration % 2]
            with torch.cuda.stream(s):
                k = 16 * (3 if iteration % 2 else 17)
                a = torch.randn(2 * bm, k, device="cuda") * 0.25
                b = torch.randn(k, 3 * bn, device="cuda") * 0.25
                expected = (a.double() @ b.double()).float()
                out = torch.full((2 * bm, 3 * bn), float("nan"), device="cuda")
                getattr(lib, name(tn))(a, b, out)
                pending.append((name(tn), iteration, out, expected))
    for s in streams:
        current.wait_stream(s)
    for n, iteration, out, expected in pending:
        torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
        checks.append(
            {
                "function": n,
                "kind": "concurrent_streams",
                "iteration": iteration,
                "passed": True,
            }
        )


def inspect_isa(lib, device):
    result = hip_isa_objects(
        lib.__file__, ROOT / f"artifacts/rocm/async_copy/isa_device{device}"
    )
    text = (ROOT / result["assembly"]).read_text()
    blocks = re.split(r"^[0-9a-f]+ <([^>]+)>:\n", text, flags=re.M)
    result["prefetch_kernels"] = []
    for symbol, body in zip(blocks[1::2], blocks[2::2]):
        if "sgemm_prefetch_kernel" not in symbol:
            continue
        pending, fmas, maximum = False, 0, 0
        for line in body.splitlines():
            if "global_load_b128" in line:
                pending, fmas = True, 0
            if pending:
                fmas += len(re.findall(r"\bv_(?:dual_)?fma(?:c)?_f32\b", line))
            if "s_wait_loadcnt" in line:
                maximum = max(maximum, fmas)
                pending, fmas = False, 0
        entry = {
            "symbol": symbol,
            "vector_load_instructions": body.count("global_load_b128"),
            "max_fmas_before_load_wait": maximum,
            "scratch_instructions": len(
                re.findall(r"\bscratch_(?:load|store)\w*", body)
            ),
            "device_call_instructions": body.count("s_swappc_b64"),
        }
        assert maximum > 0, f"No load/compute overlap found: {entry}"
        assert entry["scratch_instructions"] == 0, entry
        assert entry["device_call_instructions"] == 0, entry
        result["prefetch_kernels"].append(entry)
    assert len(result["prefetch_kernels"]) == 3
    return result


def performance(lib):
    records = []
    for n in (256, 1024, 2048):
        torch.manual_seed(n)
        a, b = [torch.randn(n, n, device="cuda") * 0.25 for _ in range(2)]
        ref = (a.double() @ b.double()).float()
        out, prior = torch.empty_like(ref), torch.empty_like(ref)
        for _, _, tn in VARIANTS:
            fn, sync = getattr(lib, name(tn)), getattr(
                lib, name(tn).removesuffix("_async")
            )
            fn(a, b, out)
            torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
            for baseline_name, baseline in (
                ("original synchronous HIP", lambda: sync(a, b, prior)),
                ("torch.mm FP32", lambda: torch.mm(a, b, out=prior)),
            ):
                baseline()
                torch.testing.assert_close(prior, ref, atol=ATOL, rtol=RTOL)
                row = {
                    "function": name(tn),
                    "shape_mnk": [n, n, n],
                    "baseline": baseline_name,
                    "passed": True,
                    **graph_pair(lambda: fn(a, b, out), baseline),
                }
                records.append(row)
                print(json.dumps(row), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--guard-device", type=int)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = verify()
    sources = [
        *source_paths("sgemm"),
        *(
            ROOT / p
            for p in (
                "leetcuda_rocm/async_gemm_check.py",
                "leetcuda_rocm/loader.py",
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/fp8_check.py",
                "leetcuda_rocm/isa_objects.py",
            )
        ),
    ]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": identity,
        "logical_device": args.device,
        "private_module": args.private,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "checks": [],
        "performance": [],
        "passed": False,
        "numeric_gate": {
            "atol": ATOL,
            "rtol": RTOL,
            "reference": "FP64 matmul rounded to FP32",
        },
        "mechanism": "A/B global_load_b128 to VGPRs, FP32 FMA on current LDS tile, explicit load-counter wait, next LDS tile; no direct-to-LDS DMA",
    }
    try:
        lib = private_module(identity) if args.private else load("sgemm")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            numerical(lib, report["checks"])
            contracts(lib, report["checks"], args.guard_device)
        stream.synchronize()
        concurrent(lib, report["checks"])
        report["isa"] = inspect_isa(lib, args.device)
        if args.performance:
            report["performance"] = performance(lib)
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
                "performance": len(report["performance"]),
                "error": report.get("error"),
            }
        )
    )
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
