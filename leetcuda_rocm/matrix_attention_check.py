"""Qualify AMD WMMA GEMM and FlashAttention, then measure GPU graph latency."""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from . import flash_attention, gemm
from .environment import LOCK, ROOT, verify
from .matrix_attention import _launch_attention, _launch_gemm

DTYPES = (torch.float16, torch.bfloat16)


def random(shape, dtype):
    return torch.randn(shape, device="cuda", dtype=dtype) * 0.5


def compare(actual, expected, dtype, attention=False):
    atol, rtol = (
        ((0.002, 0.005) if dtype == torch.float16 else (0.01, 0.02))
        if attention
        else ((0.001, 0.001) if dtype == torch.float16 else (0.01, 0.01))
    )
    torch.testing.assert_close(actual, expected.to(dtype), atol=atol, rtol=rtol)
    error = actual.double() - expected.double()
    return {
        "max_abs_error": error.abs().max().item(),
        "relative_l2_error": (
            error.norm() / expected.double().norm().clamp_min(1e-30)
        ).item(),
        "atol": atol,
        "rtol": rtol,
        "passed": True,
    }


def attention_reference(q, k, v, causal, scale):
    # Independent, explicitly materialized FP64 oracle; never used in the kernel.
    scores = (q.double() @ k.double().transpose(-1, -2)) * scale
    if causal:
        mask = torch.ones(
            q.shape[2], k.shape[2], device=q.device, dtype=torch.bool
        ).tril()
        scores.masked_fill_(~mask, -float("inf"))
    return scores.softmax(-1) @ v.double()


def isa_evidence(kernel, name, directory, dtype):
    assembly = kernel.asm["amdgcn"]
    suffix = "f16" if dtype == torch.float16 else "bf16"
    instructions = sorted(set(re.findall(r"v_wmma_[a-z0-9_]+", assembly)))
    if not any(
        f"v_wmma_f32_16x16x16_{suffix}" in inst for inst in instructions
    ):
        raise AssertionError(
            f"{name}: expected FP32-accumulating {suffix} WMMA, found {instructions}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.amdgcn"
    path.write_text(assembly)
    try:
        recorded_path = str(path.relative_to(ROOT))
    except ValueError:
        recorded_path = str(path)
    return {
        "instructions": instructions,
        "assembly": recorded_path,
        "assembly_sha256": hashlib.sha256(assembly.encode()).hexdigest(),
        "hsaco_sha256": hashlib.sha256(kernel.asm["hsaco"]).hexdigest(),
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "shared_bytes": kernel.metadata.shared,
    }


def correctness(isa_dir):
    results, isa = [], {}
    for dtype in DTYPES:
        tag = str(dtype).split(".")[-1]
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for m, n, k in (
                (1, 1, 1),
                (17, 33, 19),
                (63, 65, 31),
                (128, 128, 128),
                (129, 257, 65),
                (256, 128, 256),
                (512, 640, 320),
                (513, 769, 321),
            ):
                for layout in ("contiguous", "transposed"):
                    if layout == "contiguous":
                        a, b = random((m, k), dtype), random((k, n), dtype)
                    else:
                        a, b = random((k, m), dtype).T, random((n, k), dtype).T
                    out = torch.full(
                        (m, n), float("nan"), device="cuda", dtype=dtype
                    )
                    actual = gemm(a, b, out=out)
                    assert actual is out
                    result = compare(actual, a.double() @ b.double(), dtype)
                    results.append(
                        {
                            "operator": "gemm",
                            "dtype": tag,
                            "shape_mnk": [m, n, k],
                            "layout": layout,
                            "stream": "non-default",
                            **result,
                        }
                    )
            compiled = _launch_gemm(a, b, out)
            isa[f"gemm_{tag}"] = isa_evidence(
                compiled, f"gemm_{tag}", isa_dir, dtype
            )
            for bsz, heads, nq, nk, d in (
                (1, 1, 1, 1, 16),
                (1, 2, 17, 17, 32),
                (2, 3, 65, 65, 64),
                (1, 2, 129, 129, 128),
                (1, 2, 31, 77, 64),
                (1, 2, 257, 257, 64),
            ):
                for causal in (False, True):
                    if causal and nq != nk:
                        continue
                    q = random((bsz, nq, heads, d), dtype).transpose(1, 2)
                    k = random((bsz, nk, heads, d), dtype).transpose(1, 2)
                    v = random((bsz, nk, heads, d), dtype).transpose(1, 2)
                    scale = d**-0.5
                    actual = flash_attention(q, k, v, causal=causal)
                    expected = attention_reference(q, k, v, causal, scale)
                    result = compare(actual, expected, dtype, attention=True)
                    results.append(
                        {
                            "operator": "attention",
                            "dtype": tag,
                            "shape_bhnq_nk_d": [bsz, heads, nq, nk, d],
                            "causal": causal,
                            "layout": "BSHD-transposed",
                            "stream": "non-default",
                            **result,
                        }
                    )
            for mode, scale in (("uniform", 0.0), ("large-logits", 1.0)):
                q, k, v = (random((1, 2, 65, 64), dtype) for _ in range(3))
                if mode == "large-logits":
                    q *= 12
                    k *= 12
                out = torch.empty_like(q)
                actual = flash_attention(q, k, v, scale=scale, out=out)
                assert actual is out
                result = compare(
                    actual,
                    attention_reference(q, k, v, False, scale),
                    dtype,
                    attention=True,
                )
                results.append(
                    {
                        "operator": "attention",
                        "dtype": tag,
                        "case": mode,
                        "stream": "non-default",
                        **result,
                    }
                )
            compiled = _launch_attention(q, k, v, out, False, scale)
            isa[f"attention_{tag}"] = isa_evidence(
                compiled, f"attention_{tag}", isa_dir, dtype
            )
        stream.synchronize()
    # Reject unsupported contracts before entering a GPU kernel.
    a = random((32, 32), torch.float16)
    q = random((1, 1, 32, 32), torch.float16)
    bad = {
        "cpu-input": lambda: gemm(a.cpu(), a.cpu()),
        "dtype-mismatch": lambda: gemm(a, a.bfloat16()),
        "invalid-gemm-shape": lambda: gemm(a, a[:16]),
        "output-alias": lambda: gemm(a, a, out=a),
        "autograd": lambda: gemm(a.clone().requires_grad_(), a),
        "empty": lambda: gemm(a[:0], a),
        "attention-head-dimension": lambda: flash_attention(
            q[..., :15], q[..., :15], q[..., :15]
        ),
        "causal-cross-attention": lambda: flash_attention(
            q, q[:, :, :16], q[:, :, :16], causal=True
        ),
        "nonfinite-scale": lambda: flash_attention(q, q, q, scale=float("inf")),
        "attention-output-alias": lambda: flash_attention(q, q, q, out=q),
    }
    for name, fn in bad.items():
        try:
            fn()
        except ValueError:
            results.append(
                {"operator": "contract", "case": name, "passed": True}
            )
        else:
            raise AssertionError(f"{name} did not reject invalid inputs")
    return results, isa


def timed_pair(ours, baseline):
    # Compile and initialize libraries before graph capture and all measurements.
    for _ in range(5):
        ours()
        baseline()
    torch.cuda.synchronize()
    samples = {"ours": [], "baseline": []}
    for trial in range(5):
        order = (("ours", ours), ("baseline", baseline))
        if trial % 2:
            order = order[::-1]
        for name, fn in order:
            samples[name].append(
                triton.testing.do_bench_cudagraph(
                    fn, rep=40, return_mode="median"
                )
            )
    ours_ms, baseline_ms = (
        statistics.median(samples[name]) for name in ("ours", "baseline")
    )
    return {
        "ours_ms": ours_ms,
        "baseline_ms": baseline_ms,
        "baseline_over_ours": baseline_ms / ours_ms,
        "samples_ms": samples,
    }


def sdpa_dispatch(fn):
    # Inspect dispatch separately; profiler overhead is outside timed regions.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        fn()
    torch.cuda.synchronize()
    return sorted(
        event.key
        for event in profile.key_averages()
        if "scaled_dot_product" in event.key
    )


def performance(isa_dir):
    results, isa = [], {}
    for dtype in DTYPES:
        tag = str(dtype).split(".")[-1]
        for m, n, k in (
            (256, 256, 256),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (1024, 2048, 512),
        ):
            a, b = random((m, k), dtype), random((k, n), dtype)
            out, ref = torch.empty(
                (m, n), device="cuda", dtype=dtype
            ), torch.empty((m, n), device="cuda", dtype=dtype)
            ours = lambda: gemm(a, b, out=out)
            baseline = lambda: torch.mm(a, b, out=ref)
            ours()
            baseline()
            comparison = compare(out, ref, dtype)
            timing = timed_pair(ours, baseline)
            result = {
                "operator": "gemm",
                "dtype": tag,
                "shape_mnk": [m, n, k],
                "baseline": "torch.mm(out=), default ROCm BLAS backend",
                **timing,
                "correctness_vs_baseline": comparison,
            }
            result["ours_tflops"] = 2 * m * n * k / (timing["ours_ms"] * 1e9)
            results.append(result)
            print(json.dumps(result), flush=True)
            compiled = _launch_gemm(a, b, out)
            name = f"gemm_bench_{tag}_{m}_{n}_{k}"
            isa[name] = isa_evidence(compiled, name, isa_dir, dtype)
        for bsz, heads, n, d in (
            (1, 8, 256, 64),
            (1, 8, 1024, 64),
            (1, 8, 2048, 64),
            (1, 8, 1024, 128),
        ):
            for causal in (False, True):
                q, k, v = (random((bsz, heads, n, d), dtype) for _ in range(3))
                out = torch.empty_like(q)
                ours = lambda: flash_attention(q, k, v, causal=causal, out=out)
                baseline = lambda: F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal, dropout_p=0.0
                )
                ours()
                comparison = compare(out, baseline(), dtype, attention=True)
                dispatch = sdpa_dispatch(baseline)
                timing = timed_pair(ours, baseline)
                # Causal FLOPs count only the useful lower-triangular scores.
                pairs = n * (n + 1) / 2 if causal else n * n
                result = {
                    "operator": "attention",
                    "dtype": tag,
                    "shape_bhnd": [bsz, heads, n, d],
                    "causal": causal,
                    "baseline": "torch SDPA default dispatch",
                    "baseline_dispatch": dispatch,
                    **timing,
                    "correctness_vs_baseline": comparison,
                }
                result["ours_tflops_useful"] = (
                    4 * bsz * heads * pairs * d / (timing["ours_ms"] * 1e9)
                )
                results.append(result)
                print(json.dumps(result), flush=True)
                compiled = _launch_attention(q, k, v, out, causal, d**-0.5)
                name = f"attention_bench_{tag}_{n}_{d}_{causal}"
                isa[name] = isa_evidence(compiled, name, isa_dir, dtype)
    return results, isa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/rocm/matrix_attention/report.json",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261102)
    torch.set_grad_enabled(False)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": verify(),
        "logical_device": args.device,
        "seed": 20261102,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "leetcuda_rocm/matrix_attention.py",
                ROOT / "leetcuda_rocm/__init__.py",
                ROOT / "leetcuda_rocm/environment.py",
                LOCK,
            )
        },
        "benchmark_method": "HIP graph replay, JIT/library warmup excluded, 5 alternating-order trials, 40ms sampling per trial; medians; warm-cache; clocks not locked",
        "gemm_config": {
            "small_tile": [64, 64, 32],
            "small_warps": 4,
            "large_tile": [128, 128, 64],
            "large_warps": 8,
            "large_when": "min(M,N)>=512",
            "stages": 2,
        },
        "attention_config": {"tile_qk": [64, 64], "warps": 4, "stages": 1},
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    isa_dir = args.output.parent / "isa"
    started = time.monotonic()
    try:
        report["correctness"], report["isa"] = correctness(isa_dir)
        print(
            f"Correctness: {len(report['correctness'])} cases passed",
            flush=True,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not args.correctness_only:
            report["performance"], perf_isa = performance(isa_dir)
            report["isa"].update(perf_isa)
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {args.output}", flush=True)


if __name__ == "__main__":
    main()
