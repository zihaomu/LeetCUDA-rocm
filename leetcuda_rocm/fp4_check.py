"""Independent NVFP4 code/scale/layout and matrix qualification."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .benchmark import graph_pair
from .environment import ROOT, verify
from .fp4 import (
    MODES,
    FP4Weight,
    _native,
    fp4_gemm,
    fp4_gemm_prepacked,
    quantize_fp4_weight,
)
from .fp8_check import ATOL, RTOL
from .fp8_check import decode as decode8
from .fp8_check import encode as encode8
from .fp8_check import error_metrics
from .isa import hip_isa

VALUES4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
MIDPOINTS4 = (VALUES4[:-1] + VALUES4[1:]) / np.float32(2)


def encode4(x):
    magnitude = np.minimum(np.abs(x), np.float32(6))
    lower = np.searchsorted(MIDPOINTS4, magnitude, side="left")
    tie = (lower < 7) & (magnitude == MIDPOINTS4[np.minimum(lower, 6)])
    return (
        (lower + (tie & ((lower & 1) != 0)))
        | (np.signbit(x).astype(np.int32) << 3)
    ).astype(np.uint8)


def canonical(sf):
    rows, groups = sf.shape
    padded = np.zeros(((rows + 127) // 128 * 128, groups), dtype=np.uint8)
    padded[:rows] = sf
    # Reshape and transpose the five layout axes, independently of sf_byte_offset.
    return (
        padded.reshape(-1, 4, 32, groups // 4, 4)
        .transpose(0, 3, 2, 1, 4)
        .reshape(-1)
    )


def quantize_reference(x, level1):
    x = x.float().cpu().numpy()
    rows, k = x.shape
    scale = (
        np.abs(x).max(axis=1) / np.float32(2688)
        if level1
        else np.ones(rows, dtype=np.float32)
    )
    inv1 = np.divide(
        np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0
    )
    grouped = (x * inv1[:, None]).reshape(rows, k // 16, 16)
    sf = encode8(np.abs(grouped).max(axis=2) / np.float32(6))
    sfval = decode8(sf).astype(np.float32)
    inv2 = np.divide(
        np.float32(1), sfval, out=np.zeros_like(sfval), where=sfval != 0
    )
    codes = encode4(grouped * inv2[:, :, None]).reshape(rows, k)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    decoded = VALUES4[codes & 7] * np.where(codes & 8, -1, 1)
    decoded = decoded * sfval.repeat(16, axis=1)
    return packed, canonical(sf), scale, decoded


def gpu_quantize(a, b, l1a, l1b):
    lib = _native(a.device.index)
    m, k = a.shape
    aq = torch.full((m, k // 2), 0x55, device=a.device, dtype=torch.uint8)
    sf = torch.full(
        (lib.sf_size(m, k),), 0xFF, device=a.device, dtype=torch.uint8
    )
    sa = torch.full((m,), float("nan"), device=a.device, dtype=torch.float32)
    lib.quantize_a(a, aq, sf, sa, l1a)
    weight = quantize_fp4_weight(b, level1=l1b)
    if not l1b:
        weight.scales.fill_(float("nan"))
    return aq, sf, sa, weight


def case(shape, seed, pattern, checks):
    m, n, k = shape
    torch.manual_seed(seed)
    a = (torch.randn(m, k, device="cuda") * 0.25).bfloat16()
    b = (torch.randn(k, n, device="cuda") * 0.25).bfloat16()
    if pattern == "zero":
        a.zero_()
        b.zero_()
    elif pattern == "zero_groups":
        a[: min(128, m), :64] = 0
        b[:64, : min(128, n)] = 0
    elif pattern == "range":
        a[::2] *= 256
        a[1::2] /= 256
        b[:, ::2] /= 256
        b[:, 1::2] *= 256
    elif pattern == "sf_underflow":
        a.fill_(6 * 2**-10)
        a[::2] *= 0.99
        b.fill_(6 * 2**-10)
        b[:, ::2] *= 1.01
    elif pattern == "saturation":
        a *= 2**16
        b *= 2**16
    elif pattern == "rounding":
        vals = torch.tensor(
            [
                6,
                -6,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5,
            ],
            device=a.device,
            dtype=a.dtype,
        )
        a.copy_(vals.repeat(m * k // 16).reshape_as(a))
        b.copy_(vals.repeat(k * n // 16).reshape_as(b))
        a[0, :8] = torch.tensor([0.0, -0.0] * 4, device=a.device, dtype=a.dtype)
    elif pattern in ("small_a_large_b", "large_a_small_b"):
        exponent = -100 if pattern == "small_a_large_b" else 100
        a *= 2.0**exponent
        b *= 2.0**-exponent
    lib = _native(a.device.index)
    unquantized = a.double() @ b.double()
    for mode, mode_id in MODES.items():
        l1a, l1b = bool(mode_id & 1), bool(mode_id & 2)
        meta = {
            "shape_mnk": list(shape),
            "seed": seed,
            "pattern": pattern,
            "mode": mode,
        }
        try:
            aq, af, sa, ad = quantize_reference(a, l1a)
            bq, bf, sb, bd = quantize_reference(b.T, l1b)
            ga, gf, gs, w = gpu_quantize(a, b, l1a, l1b)
            np.testing.assert_array_equal(ga.cpu().numpy(), aq)
            np.testing.assert_array_equal(gf.cpu().numpy(), af)
            np.testing.assert_array_equal(w.data.cpu().numpy(), bq)
            np.testing.assert_array_equal(w.block_scales.cpu().numpy(), bf)
            if l1a:
                np.testing.assert_allclose(
                    gs.cpu().numpy(), sa, rtol=2e-7, atol=0
                )
            if l1b:
                np.testing.assert_allclose(
                    w.scales.cpu().numpy(), sb, rtol=2e-7, atol=0
                )
            checks.append(
                {
                    **meta,
                    "kind": "quantization",
                    "passed": True,
                    "codes_scales_layout": "CPU reference, bitwise equal including padded SF rows",
                    "level1_scale_rtol": 2e-7,
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
        ad = torch.from_numpy(ad).to(a.device)
        bd = torch.from_numpy(bd).to(a.device)
        expected = (
            (ad @ bd.T)
            * torch.from_numpy(sa).to(a.device).double()[:, None]
            * torch.from_numpy(sb).to(a.device).double()[None, :]
        ).bfloat16()
        for tile in ("default", "wide"):
            for ws in (False, True):
                for path in ("quantized", "full", "offline_b"):
                    record = {
                        **meta,
                        "tile": tile,
                        "use_ws": ws,
                        "path": path,
                        "kind": "gemm",
                    }
                    try:
                        if path == "full":
                            out = fp4_gemm(
                                a, b, mode=mode, tile=tile, use_ws=ws
                            )
                        elif path == "offline_b":
                            out = fp4_gemm_prepacked(
                                a, w, level1_a=l1a, tile=tile, use_ws=ws
                            )
                        else:
                            out = torch.full(
                                (m, n),
                                float("nan"),
                                device=a.device,
                                dtype=a.dtype,
                            )
                            lib.quantized(
                                ga,
                                w.data,
                                gf,
                                w.block_scales,
                                gs,
                                w.scales,
                                out,
                                l1a,
                                l1b,
                                ws,
                                tile == "wide",
                            )
                        torch.testing.assert_close(
                            out, expected, atol=ATOL, rtol=RTOL
                        )
                        if pattern == "zero":
                            assert torch.count_nonzero(out) == 0
                        record.update(
                            passed=True,
                            atol=ATOL,
                            rtol=RTOL,
                            quantized_error=error_metrics(out, expected),
                            unquantized_error=error_metrics(out, unquantized),
                        )
                    except Exception as exc:
                        record.update(passed=False, error=str(exc))
                    checks.append(record)
    print(
        json.dumps(
            {
                "shape": shape,
                "seed": seed,
                "pattern": pattern,
                "checks": len(checks),
                "failures": sum(not c["passed"] for c in checks),
            }
        ),
        flush=True,
    )


def contracts(checks, guard_device):
    from . import fp4_gemm as public_gemm
    from . import fp4_gemm_prepacked as public_prepacked
    from . import quantize_fp4_weight as public_quantize

    assert public_gemm is fp4_gemm and public_prepacked is fp4_gemm_prepacked
    assert public_quantize is quantize_fp4_weight
    checks.append({"kind": "public_exports", "passed": True})
    a = torch.ones(16, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(64, 16, device=a.device, dtype=a.dtype)
    w = quantize_fp4_weight(b)
    lib = _native(a.device.index)
    size = lib.workspace_size(16, 16, 64)
    workspace = torch.empty(size + 256, device=a.device, dtype=torch.uint8)
    alias_storage = torch.ones(size // 2 + 512, device=a.device, dtype=a.dtype)
    alias_a = alias_storage[: a.numel()].view_as(a)
    bad_a = torch.ones(a.numel() + 1, device=a.device, dtype=a.dtype)[
        1:
    ].view_as(a)
    cases = {
        "cpu": lambda: fp4_gemm(a.cpu(), b),
        "dtype": lambda: fp4_gemm(a.float(), b),
        "rank": lambda: fp4_gemm(a.flatten(), b),
        "strides": lambda: fp4_gemm(a.T, b),
        "gradient": lambda: fp4_gemm(a.clone().requires_grad_(), b),
        "unaligned_input": lambda: fp4_gemm(bad_a, b),
        "empty": lambda: fp4_gemm(a[:0], b),
        "k_mismatch": lambda: fp4_gemm(a, b[:32]),
        "k_alignment": lambda: fp4_gemm(a[:, :32].contiguous(), b[:32]),
        "n_alignment": lambda: fp4_gemm(a, b[:, :15].contiguous()),
        "tile": lambda: fp4_gemm(a, b, tile="ignored"),
        "mode": lambda: fp4_gemm(a, b, mode="row"),
        "workspace_size": lambda: fp4_gemm(a, b, workspace=workspace[:16]),
        "workspace_alignment": lambda: fp4_gemm(
            a, b, workspace=workspace[16 : 16 + size]
        ),
        "workspace_dtype": lambda: fp4_gemm(a, b, workspace=workspace.float()),
        "output_alias": lambda: fp4_gemm(
            a, b, out=a.flatten()[:256].view(16, 16)
        ),
        "workspace_alias": lambda: fp4_gemm(
            alias_a, b, workspace=alias_storage.view(torch.uint8)[:size]
        ),
        "output_shape": lambda: fp4_gemm(a, b, out=torch.empty_like(a)),
        "output_dtype": lambda: fp4_gemm(
            a, b, out=torch.empty(16, 16, device=a.device)
        ),
        "weight_type": lambda: fp4_gemm_prepacked(
            a, FP4Weight(None, None, None, True)
        ),
        "packed_dtype": lambda: fp4_gemm_prepacked(
            a, FP4Weight(w.data.float(), w.block_scales, w.scales, True)
        ),
        "sf_length": lambda: fp4_gemm_prepacked(
            a, FP4Weight(w.data, w.block_scales[:-1], w.scales, True)
        ),
        "sf_dtype": lambda: fp4_gemm_prepacked(
            a, FP4Weight(w.data, w.block_scales.float(), w.scales, True)
        ),
        "scale_length": lambda: fp4_gemm_prepacked(
            a, FP4Weight(w.data, w.block_scales, w.scales[:1], True)
        ),
        "scale_dtype": lambda: fp4_gemm_prepacked(
            a, FP4Weight(w.data, w.block_scales, w.scales.double(), True)
        ),
        "packed_k": lambda: fp4_gemm_prepacked(a[:, :32].contiguous(), w),
    }
    for name, fn in cases.items():
        try:
            fn()
            checks.append(
                {
                    "kind": "contract",
                    "name": name,
                    "passed": False,
                    "error": "invalid input accepted",
                }
            )
        except (ValueError, RuntimeError) as exc:
            checks.append(
                {
                    "kind": "contract",
                    "name": name,
                    "passed": True,
                    "rejection": str(exc),
                }
            )
    if guard_device is not None:
        current = torch.cuda.current_device()
        torch.cuda.set_device(guard_device)
        try:
            out = fp4_gemm(a, b)
            torch.testing.assert_close(
                out, torch.full_like(out, 64), atol=ATOL, rtol=RTOL
            )
            assert torch.cuda.current_device() == guard_device
            checks.append(
                {
                    "kind": "device_guard",
                    "input_device": current,
                    "active_device": guard_device,
                    "passed": True,
                }
            )
        finally:
            torch.cuda.set_device(current)


def exhaustive_decode(checks):
    # Test the production expansion before BF16 output rounding can hide errors.
    packed = np.repeat(np.arange(256, dtype=np.uint8), 127)
    scales = np.tile(np.arange(127, dtype=np.uint8), 256)
    unpacked = np.stack((packed & 15, packed >> 4), axis=1)
    signed = np.copysign(VALUES4[unpacked & 7], np.where(unpacked & 8, -1, 1))
    expected_raw = (signed * decode8(scales)[:, None]).astype(np.float32)
    codes_gpu, scales_gpu = [
        torch.from_numpy(x).cuda() for x in (packed, scales)
    ]
    raw = torch.empty(packed.size * 2, dtype=torch.float32, device="cuda")
    lib = _native(raw.device.index)
    lib.decode_pairs(codes_gpu, scales_gpu, raw)
    np.testing.assert_array_equal(
        raw.cpu().numpy().view(np.uint32),
        expected_raw.reshape(-1).view(np.uint32),
    )
    checks.append(
        {
            "kind": "exhaustive_expansion",
            "packed_pairs": int(packed.size),
            "scalar_values": int(raw.numel()),
            "bitwise_fp32_including_signed_zero": True,
            "passed": True,
        }
    )
    # All 16 E2M1 codes x all 127 finite nonnegative E4M3 scales.
    codes = np.repeat(np.arange(16, dtype=np.uint8), 127)
    sfcode = np.tile(np.arange(127, dtype=np.uint8), 16)
    m, n, k = len(codes), 8, 64
    ap = np.zeros((m, 32), dtype=np.uint8)
    ap[:, 0] = codes
    bp = np.zeros((n, 32), dtype=np.uint8)
    bp[:, 0] = 2  # E2M1 1
    af = np.zeros((m, 4), dtype=np.uint8)
    af[:, 0] = sfcode
    bf = np.zeros((n, 4), dtype=np.uint8)
    bf[:, 0] = 56  # E4M3 1
    a, b, sfa, sfb = [
        torch.from_numpy(x).cuda()
        for x in (ap, bp, canonical(af), canonical(bf))
    ]
    sa, sb = [
        torch.full((length,), float("nan"), device="cuda", dtype=torch.float32)
        for length in (m, n)
    ]
    expected = (
        torch.from_numpy(
            VALUES4[codes & 7] * np.where(codes & 8, -1, 1) * decode8(sfcode)
        )
        .cuda()
        .bfloat16()[:, None]
        .expand(m, n)
    )
    out = torch.empty_like(expected, memory_format=torch.contiguous_format)
    lib = _native(a.device.index)
    for wide in (False, True):
        for ws in (False, True):
            lib.quantized(a, b, sfa, sfb, sa, sb, out, False, False, ws, wide)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            checks.append(
                {
                    "kind": "exhaustive_decode",
                    "pairs": m,
                    "wide": wide,
                    "use_ws": ws,
                    "passed": True,
                    "atol": 0,
                    "rtol": 0,
                }
            )


def performance():
    records = []
    for n in (512, 2048):
        torch.manual_seed(n)
        a, b = [
            (torch.randn(n, n, device="cuda") * 0.25).bfloat16()
            for _ in range(2)
        ]
        lib = _native(a.device.index)
        out, ref = torch.empty_like(a), torch.empty_like(a)
        full_ws = torch.empty(
            lib.workspace_size(n, n, n), device=a.device, dtype=torch.uint8
        )
        act_ws = torch.empty(
            lib.activation_size(n, n), device=a.device, dtype=torch.uint8
        )
        for mode, idx in MODES.items():
            l1a, l1b = bool(idx & 1), bool(idx & 2)
            qa, qf, qs, w = gpu_quantize(a, b, l1a, l1b)
            _, _, sa, ad = quantize_reference(a, l1a)
            _, _, sb, bd = quantize_reference(b.T, l1b)
            expected = (
                (
                    torch.from_numpy(ad).to(a.device)
                    @ torch.from_numpy(bd).to(a.device).T
                )
                * torch.from_numpy(sa).to(a.device).double()[:, None]
                * torch.from_numpy(sb).to(a.device).double()[None, :]
            ).bfloat16()
            unquantized = a.double() @ b.double()
            torch.mm(a, b, out=ref)
            torch.testing.assert_close(
                ref, unquantized.bfloat16(), atol=ATOL, rtol=RTOL
            )
            for tile in ("default", "wide"):
                for ws in (False, True):
                    functions = {
                        "quantized": lambda: lib.quantized(
                            qa,
                            w.data,
                            qf,
                            w.block_scales,
                            qs,
                            w.scales,
                            out,
                            l1a,
                            l1b,
                            ws,
                            tile == "wide",
                        ),
                        "full": lambda: fp4_gemm(
                            a,
                            b,
                            mode=mode,
                            tile=tile,
                            use_ws=ws,
                            workspace=full_ws,
                            out=out,
                        ),
                        "offline_b": lambda: fp4_gemm_prepacked(
                            a,
                            w,
                            level1_a=l1a,
                            tile=tile,
                            use_ws=ws,
                            workspace=act_ws,
                            out=out,
                        ),
                    }
                    for path, fn in functions.items():
                        fn()
                        torch.testing.assert_close(
                            out, expected, atol=ATOL, rtol=RTOL
                        )
                        timing = graph_pair(fn, lambda: torch.mm(a, b, out=ref))
                        row = {
                            "shape_mnk": [n, n, n],
                            "mode": mode,
                            "tile": tile,
                            "use_ws": ws,
                            "path": path,
                            "passed": True,
                            "baseline": "torch.mm BF16, unquantized",
                            **timing,
                            "unquantized_error": error_metrics(
                                out, unquantized
                            ),
                        }
                        records.append(row)
                        print(json.dumps(row), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--guard-device", type=int)
    parser.add_argument("--performance", action="store_true")
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
        "passed": False,
        "arithmetic": "NVFP4 E2M1/E4M3 storage, exact FP16 expansion, FP32 WMMA accumulation, BF16 output",
        "native_fp4_matrix_instruction": False,
        "numeric_gate": {"atol": ATOL, "rtol": RTOL},
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / s
                for s in (
                    "kernels/interview/fp4_gemm.cuh",
                    "kernels/rocm/fp4_gemm_hip.cuh",
                    "leetcuda_rocm/native/fp4_bindings.cpp",
                    "leetcuda_rocm/fp4.py",
                    "leetcuda_rocm/fp4_check.py",
                    "leetcuda_rocm/fp8.py",
                    "leetcuda_rocm/fp8_check.py",
                    "leetcuda_rocm/benchmark.py",
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
                (1, 8, 64),
                (7, 24, 128),
                (65, 136, 192),
                (129, 264, 320),
                (33, 72, 576),
                (256, 256, 256),
                (17, 40, 1024),
            ):
                case(shape, seed, "normal", report["checks"])
        for pattern in (
            "zero",
            "zero_groups",
            "range",
            "sf_underflow",
            "saturation",
            "rounding",
            "small_a_large_b",
            "large_a_small_b",
        ):
            case((129, 136, 320), 20261103, pattern, report["checks"])
        contracts(report["checks"], args.guard_device)
        exhaustive_decode(report["checks"])
    stream.synchronize()
    report["stream"] = "non-default"
    lib = _native(args.device)
    report["configurations"] = {
        name: lib.configuration(wide)
        for name, wide in (("default", False), ("wide", True))
    }
    report["configuration_fields"] = [
        "bm",
        "bn",
        "bk",
        "stages",
        "consumer_threads",
        "lds_bytes",
    ]
    report["isa"] = hip_isa(
        lib.__file__, ROOT / f"artifacts/rocm/fp4/isa_device{args.device}"
    )
    assert "v_wmma_f32_16x16x16_f16" in report["isa"]["instructions"]
    assembly = (ROOT / report["isa"]["assembly"]).read_text()
    report["isa"]["scratch_instructions"] = len(
        re.findall(r"\bscratch_(?:load|store)\w*", assembly)
    )
    report["isa"]["device_call_instructions"] = len(
        re.findall(r"\bs_swappc_b64\b", assembly)
    )
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
        "32 calls per HIP graph, adaptive replays targeting 40ms, 5 alternating samples; reused buffers; offline B preparation excluded"
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
        )
    )
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
