"""Frozen-input comparison of original CUDA E4M3 / explicit E5M2 and HIP."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import fp8_check as reference
from .cross_validate import CUTLASS, UPSTREAM, sha, tensor_sha
from .environment import ROOT


def fixtures(args):
    torch.set_num_threads(8)
    cases = []
    for seed in (17, 93):
        generator = torch.Generator().manual_seed(seed)
        for m, n, k in ((1, 8, 16), (129, 136, 80), (256, 256, 1024)):
            a = (torch.randn(m, k, generator=generator) * 0.25).bfloat16()
            b = (torch.randn(k, n, generator=generator) * 0.25).bfloat16()
            cases.append(dict(id=f"{seed}/{m},{n},{k}", a=a, b=b))
    for pattern in ("zero_groups", "dynamic_range", "rounding", "reciprocal"):
        generator = torch.Generator().manual_seed(20261106)
        a = (torch.randn(129, 272, generator=generator) * 0.25).bfloat16()
        b = (torch.randn(272, 136, generator=generator) * 0.25).bfloat16()
        if pattern == "zero_groups":
            a[:128].zero_()
            b[:, :128].zero_()
        elif pattern == "dynamic_range":
            a[0].mul_(256)
            b[:, 0].mul_(256)
        elif pattern == "reciprocal":
            a.mul_(2**-30)
            b.mul_(2**30)
        else:
            a = (torch.arange(a.numel()).reshape_as(a) % 257 - 128).bfloat16()
            b = (
                (torch.arange(b.numel()).reshape_as(b) % 251 - 125) / 64
            ).bfloat16()
        cases.append(dict(id=pattern, a=a, b=b))
    for case in cases:
        case["input_sha256"] = {
            key: tensor_sha(case[key]) for key in ("a", "b")
        }
        case["formats"] = {}
        for fmt in ("e4m3", "e5m2"):
            reference.configure(fmt)
            modes = {}
            for mode in reference.MODES:
                row, col = mode.startswith("row"), mode.endswith("col")
                a8, sa = reference.quantize_reference(case["a"], row)
                bt8, sb = reference.quantize_reference(case["b"].T, col)
                expected = reference.oracle(a8, bt8, sa, sb, row, col, "cpu")
                modes[mode] = dict(
                    a8=torch.from_numpy(a8),
                    bt8=torch.from_numpy(bt8),
                    sa=torch.from_numpy(sa),
                    sb=torch.from_numpy(sb),
                    expected=expected,
                )
            case["formats"][fmt] = modes
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            cases=cases,
            generator_sha256=sha(__file__),
            reference_sha256=sha(reference.__file__),
        ),
        args.fixtures,
    )
    print(json.dumps(dict(cases=len(cases), sha256=sha(args.fixtures))))


def cuda_build(fmt, original, cutlass):
    from torch.utils.cpp_extension import load

    upstream = original / "kernels/interview/fp8_gemm.cuh"
    header = upstream.read_text()
    if fmt == "e5m2":
        header = (
            header.replace("e4m3", "e5m2")
            .replace("E4M3", "E5M2")
            .replace("448", "57344")
        )
    binding = (ROOT / "leetcuda_rocm/native/fp8_cuda_reference.cpp").read_text()
    include_dir = original / "kernels/interview"
    headers = {
        str(p.relative_to(original)): sha(p)
        for p in sorted(include_dir.glob("*.cuh"))
    }
    flags = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-DNOTES_V2_ENABLE_CUTE=1",
        "-DNOTES_V2_ENABLE_TMA_MMA_WS=1",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    digest = hashlib.sha256(
        (
            header
            + binding
            + json.dumps(flags + [headers, CUTLASS], sort_keys=True)
        ).encode()
    ).hexdigest()[:16]
    build = ROOT / ".cache/cuda_comparison" / f"fp8_{fmt}_{digest}"
    build.mkdir(parents=True, exist_ok=True)
    (build / "fp8_reference.cuh").write_text(header)
    source = build / "binding.cu"
    source.write_text(binding)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    os.environ.setdefault("MAX_JOBS", "2")
    lib = load(
        name=f"leetcuda_fp8_cuda_{fmt}_{digest}",
        sources=[str(source)],
        build_directory=str(build),
        extra_cuda_cflags=flags,
        extra_cflags=["-O3", "-std=c++20"],
        extra_include_paths=[str(cutlass / "include"), str(include_dir)],
        extra_ldflags=["-lcuda"],
    )
    return lib, dict(
        source_kind=(
            "upstream_original"
            if fmt == "e4m3"
            else "explicit_e5m2_format_reference"
        ),
        upstream_commit=UPSTREAM,
        cutlass_commit=CUTLASS,
        upstream_header_sha256=sha(upstream),
        reference_header_sha256=sha(build / "fp8_reference.cuh"),
        binding_sha256=sha(source),
        included_header_sha256=headers,
        binary_sha256=sha(lib.__file__),
        compiler_flags=flags,
    )


def run(args):
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    fixture = torch.load(args.fixtures, map_location="cpu", weights_only=True)
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        checks=[],
        builds={},
        passed=False,
        gate=dict(atol=reference.ATOL, rtol=reference.RTOL),
    )
    if args.backend == "hip":
        from .environment import verify

        report["environment"] = verify()
    else:
        report["environment"] = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )
    outputs = {}
    stream = torch.cuda.Stream()
    for fmt in ("e4m3", "e5m2"):
        if args.backend == "cuda":
            lib, build = cuda_build(
                fmt, args.original.resolve(), args.cutlass.resolve()
            )
        else:
            from .fp8 import _native

            lib = _native(args.device, fmt)
            build = dict(
                source_kind="rocm_port",
                binary_sha256=sha(lib.__file__),
                source_sha256={
                    name: sha(ROOT / name)
                    for name in (
                        "kernels/rocm/fp8_gemm_hip.cuh",
                        "leetcuda_rocm/native/fp8_bindings.cpp",
                        "leetcuda_rocm/fp8.py",
                    )
                },
            )
        report["builds"][fmt] = build
        with torch.cuda.stream(stream):
            for case in fixture["cases"]:
                a, b = (case[key].cuda() for key in ("a", "b"))
                m, k = a.shape
                n = b.shape[1]
                out = torch.empty((m, n), device=a.device, dtype=a.dtype)
                full_ws = torch.empty(
                    lib.workspace_size(m, n, k),
                    device=a.device,
                    dtype=torch.uint8,
                )
                act_ws = torch.empty(
                    lib.activation_size(m, k),
                    device=a.device,
                    dtype=torch.uint8,
                )
                for mode, expected in case["formats"][fmt].items():
                    row, col = mode.startswith("row"), mode.endswith("col")
                    a8 = torch.empty_like(a, dtype=torch.uint8)
                    bt8 = torch.empty(
                        (n, k), device=a.device, dtype=torch.uint8
                    )
                    sa = torch.empty_like(expected["sa"], device=a.device)
                    sb = torch.empty_like(expected["sb"], device=a.device)
                    lib.quantize_a(a, a8, sa, row)
                    lib.quantize_b(b, bt8, sb, col)
                    for key, value in (
                        ("a8", a8),
                        ("bt8", bt8),
                        ("sa", sa),
                        ("sb", sb),
                    ):
                        actual = value.cpu()
                        passed = (
                            torch.equal(actual, expected[key])
                            if key in ("a8", "bt8")
                            else torch.allclose(
                                actual, expected[key], atol=0, rtol=2e-7
                            )
                        )
                        report["checks"].append(
                            dict(
                                id=f"{fmt}/{case['id']}/{mode}/{key}",
                                kind=(
                                    "exact_quantization"
                                    if key in ("a8", "bt8")
                                    else "fp32_scale_rtol_2e-7"
                                ),
                                passed=passed,
                            )
                        )
                    for ws in (False, True):
                        paths = dict(
                            quantized=lambda: lib.quantized(
                                a8, bt8, sa, sb, out, row, col, ws
                            ),
                            full=lambda: lib.forward(
                                a, b, out, full_ws, reference.MODES[mode], ws
                            ),
                            offline_b=lambda: lib.offline(
                                a, bt8, sb, out, act_ws, row, col, ws
                            ),
                        )
                        for path, fn in paths.items():
                            out.fill_(float("nan"))
                            fn()
                            actual = out.cpu()
                            key = f"{fmt}/{case['id']}/{mode}/{int(ws)}/{path}"
                            passed = torch.allclose(
                                actual,
                                expected["expected"],
                                atol=reference.ATOL,
                                rtol=reference.RTOL,
                            )
                            report["checks"].append(
                                dict(
                                    id=key,
                                    kind="gemm",
                                    passed=passed,
                                    **reference.error_metrics(
                                        actual, expected["expected"]
                                    ),
                                )
                            )
                            outputs[key] = actual
                print(f"{args.backend} {fmt} {case['id']} checked", flush=True)
        stream.synchronize()
    report["passed"] = all(c["passed"] for c in report["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.output.with_suffix(".pt"))
    report["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                checks=len(report["checks"]),
                outputs=len(outputs),
            )
        )
    )
    return int(not report["passed"])


def compare(args):
    reports = [json.loads(p.read_text()) for p in (args.hip, args.cuda)]
    assert reports[0]["fixture_sha256"] == reports[1]["fixture_sha256"]
    results = [
        torch.load(p.with_suffix(".pt"), weights_only=True)
        for p in (args.hip, args.cuda)
    ]
    assert results[0].keys() == results[1].keys()
    checks = []
    for key in results[0]:
        a, b = (o[key] for o in results)
        checks.append(
            dict(
                id=key,
                passed=torch.allclose(
                    a, b, atol=reference.ATOL, rtol=reference.RTOL
                ),
                exact=torch.equal(a, b),
                **reference.error_metrics(a, b),
            )
        )
    report = dict(
        passed=all(r["passed"] for r in reports)
        and all(c["passed"] for c in checks),
        checks=checks,
        fixture_sha256=reports[0]["fixture_sha256"],
        report_sha256={str(p): sha(p) for p in (args.hip, args.cuda)},
        scope="E4M3 upstream CUDA; E5M2 explicit format extension of same CUDA pipeline; both use independent CPU oracle",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], checks=len(checks))))
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("fixtures")
    gen.add_argument("--fixtures", type=Path, required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--backend", choices=("hip", "cuda"), required=True)
    runner.add_argument("--device", type=int, default=0)
    runner.add_argument("--fixtures", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--original", type=Path, default=ROOT / "original")
    runner.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    cmp = sub.add_parser("compare")
    for name in ("hip", "cuda", "output"):
        cmp.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
