"""Frozen-input NVFP4 quantization, full pipeline and offline-B comparison."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import fp4_check as reference
from .cross_validate import CUTLASS, UPSTREAM, sha, tensor_sha
from .environment import ROOT


def fixtures(args):
    torch.set_num_threads(8)
    cases = []
    for seed in (17, 93):
        generator = torch.Generator().manual_seed(seed)
        for m, n, k in ((1, 8, 64), (129, 136, 320), (256, 256, 1024)):
            a = (torch.randn(m, k, generator=generator) * 0.25).bfloat16()
            b = (torch.randn(k, n, generator=generator) * 0.25).bfloat16()
            cases.append(dict(id=f"{seed}/{m},{n},{k}", a=a, b=b))
    for pattern in (
        "zero",
        "zero_groups",
        "range",
        "rounding",
        "reciprocal",
        "sf_underflow",
        "saturation",
    ):
        generator = torch.Generator().manual_seed(20261106)
        a = (torch.randn(129, 320, generator=generator) * 0.25).bfloat16()
        b = (torch.randn(320, 136, generator=generator) * 0.25).bfloat16()
        if pattern == "zero":
            a.zero_()
            b.zero_()
        elif pattern == "zero_groups":
            a[:128, :64].zero_()
            b[:64, :128].zero_()
        elif pattern == "range":
            a[::2] *= 256
            a[1::2] /= 256
            b[:, ::2] /= 256
            b[:, 1::2] *= 256
        elif pattern == "reciprocal":
            a *= 2**-30
            b *= 2**30
        elif pattern == "sf_underflow":
            a.fill_(6 * 2**-10)
            a[::2] *= 0.99
            b.fill_(6 * 2**-10)
            b[:, ::2] *= 1.01
        elif pattern == "saturation":
            a *= 2**16
            b *= 2**16
        else:
            values = torch.tensor(
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
                dtype=torch.bfloat16,
            )
            a.copy_(values.repeat(a.numel() // 16).reshape_as(a))
            b.copy_(values.repeat(b.numel() // 16).reshape_as(b))
            a[0, :8] = torch.tensor([0.0, -0.0] * 4, dtype=a.dtype)
        cases.append(dict(id=pattern, a=a, b=b))
    for case in cases:
        case["input_sha256"] = {
            key: tensor_sha(case[key]) for key in ("a", "b")
        }
        modes = {}
        for mode, code in reference.MODES.items():
            aq, af, sa, ad = reference.quantize_reference(
                case["a"], bool(code & 1)
            )
            bq, bf, sb, bd = reference.quantize_reference(
                case["b"].T, bool(code & 2)
            )
            expected = (
                (
                    torch.from_numpy(ad).double()
                    @ torch.from_numpy(bd).double().T
                )
                * torch.from_numpy(sa).double()[:, None]
                * torch.from_numpy(sb).double()[None, :]
            ).bfloat16()
            modes[mode] = dict(
                aq=torch.from_numpy(aq),
                af=torch.from_numpy(af),
                sa=torch.from_numpy(sa),
                bq=torch.from_numpy(bq),
                bf=torch.from_numpy(bf),
                sb=torch.from_numpy(sb),
                expected=expected,
            )
        case["modes"] = modes
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


def cuda_build(original, cutlass, compiler_fix):
    from torch.utils.cpp_extension import load

    upstream = original / "kernels/interview/fp4_gemm.cuh"
    header = upstream.read_text()
    if compiler_fix:
        old = (
            "static_assert(size<0>(typename TiledMma::AtomShape_MNK{}) == 16 &&\n"
            "                    size<2>(typename TiledMma::AtomShape_MNK{}) == kBK,"
        )
        new = (
            "using AtomShapeMNK = typename TiledMma::AtomShape_MNK;\n"
            "  static_assert(decltype(size<0>(AtomShapeMNK{}))::value == 16 &&\n"
            "                    decltype(size<2>(AtomShapeMNK{}))::value == kBK,"
        )
        assert header.count(old) == 1
        header = header.replace(old, new)
    binding = (ROOT / "leetcuda_rocm/native/fp4_cuda_reference.cpp").read_text()
    include = original / "kernels/interview"
    headers = {
        str(p.relative_to(original)): sha(p)
        for p in sorted(include.glob("*.cuh"))
    }
    flags = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-DNOTES_V2_ENABLE_CUTE=1",
        "-DNOTES_V2_ENABLE_TMA_MMA_WS=1",
        "-DNOTES_V2_ENABLE_SM120_FP4=1",
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
    build = ROOT / ".cache/cuda_comparison" / f"fp4_{digest}"
    build.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("fp4_reference.cuh", header),
        ("binding.cu", binding),
    ):
        path = build / name
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    os.environ.setdefault("MAX_JOBS", "2")
    lib = load(
        name=f"leetcuda_fp4_cuda_{digest}",
        sources=[str(build / "binding.cu")],
        build_directory=str(build),
        extra_cuda_cflags=flags,
        extra_cflags=["-O3", "-std=c++20"],
        extra_include_paths=[str(cutlass / "include"), str(include)],
        extra_ldflags=["-lcuda"],
    )
    return lib, dict(
        source_kind=(
            "explicit_constexpr_compiler_compatibility"
            if compiler_fix
            else "upstream_original"
        ),
        upstream_commit=UPSTREAM,
        cutlass_commit=CUTLASS,
        upstream_header_sha256=sha(upstream),
        reference_header_sha256=sha(build / "fp4_reference.cuh"),
        binding_sha256=sha(build / "binding.cu"),
        included_header_sha256=headers,
        binary_sha256=sha(lib.__file__),
        binary=lib.__file__,
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
        passed=False,
        gate=dict(atol=reference.ATOL, rtol=reference.RTOL),
    )
    if args.backend == "hip":
        from .environment import verify
        from .fp4 import _native

        report["environment"] = verify()
        lib = _native(args.device)
        report["build"] = dict(
            source_kind="rocm_port",
            binary_sha256=sha(lib.__file__),
            binary=lib.__file__,
            source_sha256={
                name: sha(ROOT / name)
                for name in (
                    "kernels/rocm/fp4_gemm_hip.cuh",
                    "leetcuda_rocm/native/fp4_bindings.cpp",
                    "leetcuda_rocm/fp4.py",
                )
            },
        )
    else:
        report["environment"] = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )
        lib, report["build"] = cuda_build(
            args.original.resolve(), args.cutlass.resolve(), args.compiler_fix
        )
    outputs = {}
    stream = torch.cuda.Stream()

    def record(key, actual, expected, kind):
        exact = kind == "exact_quantization"
        atol = 0 if kind != "gemm" else reference.ATOL
        rtol = 0 if exact else 2e-7 if kind == "scale" else reference.RTOL
        passed = (
            torch.equal(actual, expected)
            if exact
            else torch.allclose(actual, expected, atol=atol, rtol=rtol)
        )
        report["checks"].append(
            dict(
                id=key,
                kind=kind,
                passed=passed,
                atol=atol,
                rtol=rtol,
                max_abs=float(
                    (actual.double() - expected.double()).abs().max()
                ),
            )
        )
        outputs[key] = actual

    with torch.cuda.stream(stream):
        for case in fixture["cases"]:
            a, b = (case[key].cuda() for key in ("a", "b"))
            m, k = a.shape
            n = b.shape[1]
            # Storage margins and poisoned outputs check writes independently of allocation reuse.
            backing = torch.full(
                (m * n + 32,), 91.0, device=a.device, dtype=a.dtype
            )
            out = backing[16:-16].view(m, n)
            full_ws = torch.empty(
                lib.workspace_size(m, n, k), device=a.device, dtype=torch.uint8
            )
            act_ws = torch.empty(
                lib.activation_size(m, k), device=a.device, dtype=torch.uint8
            )
            for mode, expected in case["modes"].items():
                code = reference.MODES[mode]
                l1a, l1b = bool(code & 1), bool(code & 2)
                aq = torch.full_like(expected["aq"], 0x55, device=a.device)
                bq = torch.full_like(expected["bq"], 0x55, device=a.device)
                af = torch.full_like(expected["af"], 0xFF, device=a.device)
                bf = torch.full_like(expected["bf"], 0xFF, device=a.device)
                sa = torch.full_like(
                    expected["sa"], float("nan") if l1a else 1, device=a.device
                )
                sb = torch.full_like(
                    expected["sb"], float("nan") if l1b else 1, device=a.device
                )
                lib.quantize_a(a, aq, af, sa, l1a)
                lib.quantize_b(b, bq, bf, sb, l1b)
                for key, value in (
                    ("aq", aq),
                    ("af", af),
                    ("sa", sa),
                    ("bq", bq),
                    ("bf", bf),
                    ("sb", sb),
                ):
                    record(
                        f"{case['id']}/{mode}/{key}",
                        value.cpu(),
                        expected[key],
                        (
                            "scale"
                            if key in ("sa", "sb")
                            else "exact_quantization"
                        ),
                    )
                for wide in (False, True):
                    for ws in (False, True):
                        paths = dict(
                            quantized=lambda: lib.quantized(
                                aq, bq, af, bf, sa, sb, out, l1a, l1b, ws, wide
                            ),
                            full=lambda: lib.forward(
                                a, b, out, full_ws, code, ws, wide
                            ),
                            offline_b=lambda: lib.offline(
                                a, bq, bf, sb, out, act_ws, l1a, l1b, ws, wide
                            ),
                        )
                        for path, fn in paths.items():
                            out.fill_(float("nan"))
                            fn()
                            key = f"{case['id']}/{mode}/{int(wide)}/{int(ws)}/{path}"
                            record(key, out.cpu(), expected["expected"], "gemm")
                            assert bool((backing[:16] == 91).all()) and bool(
                                (backing[-16:] == 91).all()
                            ), key
                            # Warm call is complete before capture; reset AFTER capture.
                            stream.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                fn()
                            out.fill_(float("nan"))
                            graph.replay()
                            record(
                                key + "/graph",
                                out.cpu(),
                                expected["expected"],
                                "gemm",
                            )
                            assert bool((backing[:16] == 91).all()) and bool(
                                (backing[-16:] == 91).all()
                            ), key
            print(f"{args.backend} {case['id']} checked", flush=True)
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
                failed=sum(not c["passed"] for c in report["checks"]),
            )
        )
    )
    return int(not report["passed"])


def compare(args):
    reports = [json.loads(p.read_text()) for p in (args.hip, args.cuda)]
    assert reports[0]["fixture_sha256"] == reports[1]["fixture_sha256"]
    outputs = [
        torch.load(p.with_suffix(".pt"), map_location="cpu", weights_only=True)
        for p in (args.hip, args.cuda)
    ]
    assert outputs[0].keys() == outputs[1].keys()
    checks = []
    for entry in reports[0]["checks"]:
        key = entry["id"]
        a, b = (o[key] for o in outputs)
        exact = torch.equal(a, b)
        checks.append(
            dict(
                id=key,
                passed=(
                    exact
                    if entry["kind"] == "exact_quantization"
                    else torch.allclose(
                        a, b, atol=entry["atol"], rtol=entry["rtol"]
                    )
                ),
                exact=exact,
                max_abs=float((a.double() - b.double()).abs().max()),
            )
        )
    report = dict(
        passed=all(r["passed"] for r in reports)
        and all(c["passed"] for c in checks),
        checks=checks,
        fixture_sha256=reports[0]["fixture_sha256"],
        report_sha256={str(p): sha(p) for p in (args.hip, args.cuda)},
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                checks=len(checks),
                exact=sum(c["exact"] for c in checks),
            )
        )
    )
    return int(not report["passed"])


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("fixtures")
    gen.add_argument("--fixtures", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=("hip", "cuda"), required=True)
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--fixtures", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--original", type=Path, default=ROOT / "original")
    r.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    r.add_argument("--compiler-fix", action="store_true")
    c = sub.add_parser("compare")
    for name in ("hip", "cuda", "output"):
        c.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
