"""Qualify the native FP8 pipeline against an independent CPU E4M3 oracle."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .environment import ROOT, verify
from .fp8 import (
    MODES,
    FP8Weight,
    _native,
    fp8_gemm,
    fp8_gemm_prepacked,
    quantize_fp8_weight,
)
from .isa import hip_isa
from .matrix_attention_check import timed_pair

# Positive finite E4M3FN codes, derived from the exponent/mantissa definition.
CODES = np.arange(127, dtype=np.int32)
VALUES = np.where(
    CODES < 8, CODES * 2.0**-9, (8 + CODES % 8) * np.exp2(CODES // 8 - 10)
).astype(np.float32)
MIDPOINTS = (VALUES[:-1] + VALUES[1:]) / np.float32(2)
ATOL, RTOL = (
    2**-10,
    2**-7,
)  # BF16 output: one relative ULP, plus near-zero floor.


def encode(x):
    magnitude = np.minimum(np.abs(x), np.float32(448))
    lower = np.searchsorted(MIDPOINTS, magnitude, side="left")
    ties = (lower < 126) & (magnitude == MIDPOINTS[np.minimum(lower, 125)])
    nearest = lower + (ties & ((lower & 1) != 0))
    return (nearest | (np.signbit(x).astype(np.int32) << 7)).astype(np.uint8)


def decode(x):
    return VALUES[x & 127] * np.where(x & 128, -1, 1)


def quantize_reference(x, individual):
    """No GPU FP8 conversion or production reduction code is used here."""
    x = x.float().cpu().numpy()
    codes = np.empty(x.shape, dtype=np.uint8)
    stride = 1 if individual else 128
    scales = []
    for start in range(0, len(x), stride):
        part = x[start : start + stride]
        amax = np.abs(part).max()
        inv = np.float32(448) / amax if amax else np.float32(0)
        scales.append(amax / np.float32(448))
        codes[start : start + stride] = encode(part * inv)
    return codes, np.asarray(scales, dtype=np.float32)


def expanded(scale, length, individual, device):
    v = torch.from_numpy(scale.copy()).to(device=device, dtype=torch.float64)
    return v if individual else v.repeat_interleave(128)[:length]


def oracle(a8, bt8, sa, sb, row, col, device):
    a = torch.from_numpy(decode(a8)).to(device)
    bt = torch.from_numpy(decode(bt8)).to(device)
    return (
        (a @ bt.T)
        * expanded(sa, len(a), row, device)[:, None]
        * expanded(sb, len(bt), col, device)[None, :]
    ).to(torch.bfloat16)


def error_metrics(actual, expected):
    delta = actual.double() - expected.double()
    return {
        "max_abs_error": delta.abs().max().item(),
        "relative_l2": (
            delta.norm() / expected.double().norm().clamp_min(1e-300)
        ).item(),
    }


def check_case(shape, seed, pattern, checks):
    m, n, k = shape
    torch.manual_seed(seed)
    a = (torch.randn(m, k, device="cuda") * 0.25).bfloat16()
    b = (torch.randn(k, n, device="cuda") * 0.25).bfloat16()
    if pattern == "zero_groups":
        a[: min(m, 128)].zero_()
        b[:, : min(n, 128)].zero_()
    elif pattern == "dynamic_range":
        a[::2] *= 256
        a[1::2] /= 256
        b[:, ::2] /= 256
        b[:, 1::2] *= 256
    elif pattern in ("small_a_large_b", "large_a_small_b"):
        exponent = -100 if pattern == "small_a_large_b" else 100
        a *= 2.0**exponent
        b *= 2.0**-exponent
    elif pattern == "rounding":
        # Exact ties and both signed zeros, with amax=448 => scale=1.
        values = (
            torch.from_numpy(
                np.concatenate(
                    (MIDPOINTS, -MIDPOINTS, [448.0, -448.0, 0.0, -0.0])
                )
            )
            .bfloat16()
            .cuda()
        )
        a.copy_(
            values[
                torch.arange(a.numel(), device="cuda") % len(values)
            ].reshape_as(a)
        )
        b.copy_(
            values[
                torch.arange(b.numel(), device="cuda") % len(values)
            ].reshape_as(b)
        )
    lib = _native(a.device.index)
    full_precision = a.double() @ b.double()
    for mode in MODES:
        row, col = mode.startswith("row"), mode.endswith("col")
        meta = {
            "shape_mnk": list(shape),
            "seed": seed,
            "pattern": pattern,
            "mode": mode,
        }
        try:
            aq, sa = quantize_reference(a, row)
            bq, sb = quantize_reference(b.T, col)
            a8 = torch.empty_like(a, dtype=torch.uint8)
            ascale = torch.empty(len(sa), device=a.device, dtype=torch.float32)
            lib.quantize_a(a, a8, ascale, row)
            weight = quantize_fp8_weight(b, per_col=col)
            np.testing.assert_array_equal(a8.cpu().numpy(), aq)
            np.testing.assert_array_equal(weight.data.cpu().numpy(), bq)
            np.testing.assert_allclose(
                ascale.cpu().numpy(), sa, rtol=2e-7, atol=0
            )
            np.testing.assert_allclose(
                weight.scales.cpu().numpy(), sb, rtol=2e-7, atol=0
            )
            checks.append(
                {
                    **meta,
                    "kind": "quantization",
                    "passed": True,
                    "codes": "CPU E4M3 RNE oracle, bitwise equal",
                    "scale_rtol": 2e-7,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    **meta,
                    "kind": "quantization",
                    "passed": False,
                    "error": str(exc),
                }
            )
            continue
        expected = oracle(aq, bq, sa, sb, row, col, a.device)
        for ws in (False, True):
            for path in ("quantized", "full", "offline_b"):
                record = {**meta, "kind": "gemm", "use_ws": ws, "path": path}
                try:
                    if path == "full":
                        out = fp8_gemm(a, b, mode=mode, use_ws=ws)
                    elif path == "offline_b":
                        out = fp8_gemm_prepacked(
                            a, weight, per_row=row, use_ws=ws
                        )
                    else:
                        out = torch.full(
                            (m, n),
                            float("nan"),
                            device=a.device,
                            dtype=torch.bfloat16,
                        )
                        lib.quantized(
                            a8,
                            weight.data,
                            ascale,
                            weight.scales,
                            out,
                            row,
                            col,
                            ws,
                        )
                    torch.testing.assert_close(
                        out, expected, atol=ATOL, rtol=RTOL
                    )
                    if pattern == "zero_groups":
                        assert torch.count_nonzero(out[: min(m, 128)]) == 0
                        assert torch.count_nonzero(out[:, : min(n, 128)]) == 0
                    record.update(
                        passed=True,
                        atol=ATOL,
                        rtol=RTOL,
                        quantized_oracle_error=error_metrics(out, expected),
                        unquantized_error=error_metrics(out, full_precision),
                    )
                except Exception as exc:
                    record.update(passed=False, error=str(exc))
                checks.append(record)
    failures = sum(not r["passed"] for r in checks)
    print(
        json.dumps(
            {
                "shape": shape,
                "seed": seed,
                "pattern": pattern,
                "checks_so_far": len(checks),
                "failures_so_far": failures,
            }
        ),
        flush=True,
    )


def contracts(checks, guard_device):
    from . import fp8_gemm as public_gemm
    from . import fp8_gemm_prepacked as public_prepacked
    from . import quantize_fp8_weight as public_quantize

    assert public_gemm is fp8_gemm
    assert public_prepacked is fp8_gemm_prepacked
    assert public_quantize is quantize_fp8_weight
    checks.append({"kind": "public_exports", "passed": True})
    a = torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)
    b = a.clone()
    w = quantize_fp8_weight(b)
    lib = _native(a.device.index)
    size = lib.workspace_size(16, 16, 16)
    storage = torch.empty(size + 256, dtype=torch.uint8, device=a.device)
    bad_a = torch.ones(a.numel() + 1, device=a.device, dtype=a.dtype)[
        1:
    ].view_as(a)
    cases = {
        "cpu": lambda: fp8_gemm(a.cpu(), b),
        "dtype": lambda: fp8_gemm(a.float(), b),
        "rank": lambda: fp8_gemm(a.flatten(), b),
        "noncontiguous": lambda: fp8_gemm(a.T, b),
        "gradient": lambda: fp8_gemm(a.clone().requires_grad_(), b),
        "unaligned_input": lambda: fp8_gemm(bad_a, b),
        "k_mismatch": lambda: fp8_gemm(a, b[:8]),
        "k_alignment": lambda: fp8_gemm(a[:, :15].contiguous(), b[:15]),
        "n_alignment": lambda: fp8_gemm(a, b[:, :15].contiguous()),
        "empty": lambda: fp8_gemm(a[:0], b),
        "mode": lambda: fp8_gemm(a, b, mode="tensor"),
        "workspace_size": lambda: fp8_gemm(a, b, workspace=storage[:16]),
        "workspace_alignment": lambda: fp8_gemm(
            a, b, workspace=storage[16 : 16 + size]
        ),
        "workspace_dtype": lambda: fp8_gemm(a, b, workspace=storage.float()),
        "output_alias": lambda: fp8_gemm(a, b, out=a),
        "output_shape": lambda: fp8_gemm(a, b, out=torch.empty_like(a[:8])),
        "output_dtype": lambda: fp8_gemm(a, b, out=a.float()),
        "packed_k": lambda: fp8_gemm_prepacked(a[:, :8].contiguous(), w),
        "packed_dtype": lambda: fp8_gemm_prepacked(
            a, FP8Weight(w.data.float(), w.scales, True)
        ),
        "packed_metadata": lambda: fp8_gemm_prepacked(
            a, FP8Weight(None, None, True)
        ),
        "packed_scales": lambda: fp8_gemm_prepacked(
            a, FP8Weight(w.data, w.scales[:1], True)
        ),
        "packed_scale_dtype": lambda: fp8_gemm_prepacked(
            a, FP8Weight(w.data, w.scales.double(), True)
        ),
        "packed_workspace_alias": lambda: fp8_gemm_prepacked(
            a, w, workspace=w.data.flatten()
        ),
    }
    for name, fn in cases.items():
        record = {"kind": "contract", "name": name}
        try:
            fn()
            record.update(passed=False, error="invalid input accepted")
        except (ValueError, RuntimeError) as exc:
            record.update(passed=True, rejection=str(exc))
        checks.append(record)
    # Valid offset; optionally select another device after checking occupancy.
    storage_a = torch.ones(a.numel() + 8, device=a.device, dtype=a.dtype)
    offset_a = storage_a[8:].view_as(a)
    old = torch.cuda.current_device()
    alternate = old if guard_device is None else guard_device
    torch.cuda.set_device(alternate)
    try:
        out = fp8_gemm(offset_a, b)
        torch.testing.assert_close(
            out, torch.full_like(a, 16), atol=ATOL, rtol=RTOL
        )
        assert torch.cuda.current_device() == alternate
        checks.append(
            {
                "kind": "device_guard",
                "input_device": old,
                "active_device": alternate,
                "aligned_storage_offset": 8,
                "passed": True,
            }
        )
    finally:
        torch.cuda.set_device(old)


def performance():
    records = []
    for n in (512, 2048):
        torch.manual_seed(n)
        a, b = [
            (torch.randn(n, n, device="cuda") * 0.25).bfloat16()
            for _ in range(2)
        ]
        out, ref = torch.empty_like(a), torch.empty_like(a)
        full_precision = a.double() @ b.double()
        lib = _native(a.device.index)
        full_ws = torch.empty(
            lib.workspace_size(n, n, n), dtype=torch.uint8, device=a.device
        )
        act_ws = torch.empty(
            lib.activation_size(n, n), dtype=torch.uint8, device=a.device
        )
        baseline = lambda: torch.mm(a, b, out=ref)
        for mode in MODES:
            row, col = mode.startswith("row"), mode.endswith("col")
            w = quantize_fp8_weight(b, per_col=col)
            a8 = torch.empty_like(a, dtype=torch.uint8)
            sa = torch.empty(
                n if row else (n + 127) // 128,
                device=a.device,
                dtype=torch.float32,
            )
            lib.quantize_a(a, a8, sa, row)
            # Decode already checked native bytes, then use independent FP64 GEMM.
            expected = (
                (
                    a8.view(torch.float8_e4m3fn).double()
                    @ w.data.view(torch.float8_e4m3fn).double().T
                )
                * (sa if row else sa.repeat_interleave(128))[:, None].double()
                * (w.scales if col else w.scales.repeat_interleave(128))[
                    None, :
                ].double()
            ).bfloat16()
            for ws in (False, True):
                paths = {
                    "quantized": lambda: lib.quantized(
                        a8, w.data, sa, w.scales, out, row, col, ws
                    ),
                    "full": lambda: fp8_gemm(
                        a, b, mode=mode, use_ws=ws, workspace=full_ws, out=out
                    ),
                    "offline_b": lambda: fp8_gemm_prepacked(
                        a, w, per_row=row, use_ws=ws, workspace=act_ws, out=out
                    ),
                }
                for path, fn in paths.items():
                    fn()
                    torch.testing.assert_close(
                        out, expected, atol=ATOL, rtol=RTOL
                    )
                    timing = timed_pair(fn, baseline)
                    record = {
                        "shape_mnk": [n, n, n],
                        "mode": mode,
                        "use_ws": ws,
                        "path": path,
                        "baseline": "torch.mm BF16, no FP8 quantization",
                        "passed": True,
                        "unquantized_error": error_metrics(out, full_precision),
                        "quantization_timed": {
                            "quantized": "none",
                            "full": "A and B",
                            "offline_b": "A only",
                        }[path],
                        **timing,
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--guard-device", type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.set_grad_enabled(False)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": verify(),
        "logical_device": args.device,
        "checks": [],
        "performance": [],
        "passed": False,
        "oracle": "CPU E4M3 RNE bitwise quantization, FP64 quantized GEMM, BF16 output",
        "numeric_gate": {"atol": ATOL, "rtol": RTOL},
        "input_scope": "finite BF16; quantization uses FP32 amax/448 and 448/amax; no guarantee for FP32 scale underflow/overflow",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / s
                for s in (
                    "kernels/rocm/fp8_gemm_hip.cuh",
                    "kernels/interview/fp8_gemm.cuh",
                    "leetcuda_rocm/native/fp8_bindings.cpp",
                    "leetcuda_rocm/fp8.py",
                    "leetcuda_rocm/fp8_check.py",
                    "leetcuda_rocm/loader.py",
                    "leetcuda_rocm/isa.py",
                    "leetcuda_rocm/__init__.py",
                )
            ]
        },
    }
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for seed in (20261103, 17, 93):
            for shape in (
                (1, 8, 16),
                (7, 24, 32),
                (65, 136, 80),
                (129, 264, 176),
                (33, 72, 576),
                (256, 256, 256),
                (17, 40, 1024),
            ):
                check_case(shape, seed, "normal", report["checks"])
        for pattern in (
            "zero_groups",
            "dynamic_range",
            "rounding",
            "small_a_large_b",
            "large_a_small_b",
        ):
            check_case((129, 136, 272), 20261103, pattern, report["checks"])
        contracts(report["checks"], args.guard_device)
    stream.synchronize()
    report["stream"] = "non-default"
    lib = _native(args.device)
    report["isa"] = hip_isa(
        lib.__file__,
        ROOT / f"artifacts/rocm/fp8_native/isa_device{args.device}",
    )
    assert "v_wmma_f32_16x16x16_fp8_fp8" in report["isa"]["instructions"]
    if args.performance and all(x["passed"] for x in report["checks"]):
        try:
            report["performance"] = performance()
        except Exception as exc:
            report["performance_error"] = str(exc)
    report["passed"] = (
        all(x["passed"] for x in report["checks"])
        and "performance_error" not in report
    )
    report["benchmark_method"] = (
        "reused buffers, HIP graphs, 5 alternating 40 ms samples, median; allocations and offline B preparation excluded"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(report["checks"]),
                "performance": len(report["performance"]),
            }
        ),
        flush=True,
    )
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
