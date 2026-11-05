"""Cross-device regression of every currently qualified original interface.

Uses a frozen fixture on both backends, with the existing numerical gates.
CUDA original source and ROCm adapted source have separate recorded hashes.
This finite-shape comparison supplements, not replaces, stream/contract tests.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .cross_validate import CUTLASS, ROOT, UPSTREAM, sha, tensor_sha, timed
from .loader import KERNELS


def create_fixtures(args):
    from .extra_checks import merge_reference
    from .smoke import case

    qualification = json.loads(args.qualification.read_text())
    assert qualification["passed"]
    torch.manual_seed(20261105)
    cases = []
    modules = {}
    for module, record in qualification["modules"].items():
        names = sorted({c["function"] for c in record["checks"]})
        modules[module] = names
        for name in names:
            for rows in (8, 16):
                if module == "merge_attn_states":
                    for dtype in (torch.float32, torch.float16, torch.bfloat16):
                        p, s = (
                            torch.randn(
                                rows, 3, 128, device="cuda", dtype=dtype
                            )
                            for _ in range(2)
                        )
                        pl, sl = (
                            torch.randn(3, rows, device="cuda")
                            for _ in range(2)
                        )
                        out = torch.empty_like(p)
                        expected, _ = merge_reference(p, pl, s, sl)
                        tol = {
                            torch.float32: 2e-5,
                            torch.float16: 0.001,
                            torch.bfloat16: 0.008,
                        }[dtype]
                        cases.append(
                            dict(
                                id=f"{module}/{name}/{rows}/{dtype}",
                                module=module,
                                function=name,
                                args=(out, None, p, pl, s, sl),
                                output_index=0,
                                expected=expected.to(dtype),
                                atol=tol,
                                rtol=tol,
                            )
                        )
                    continue
                inputs, expected, out, atol, rtol = case(module, name, rows)
                # These newly qualified interfaces have a TF32 CUDA oracle.
                # Keep its historical 0.005 gate from cuda_pending_check;
                # AMD smoke and tf32_check separately enforce 0.0002 for the
                # explicitly residual-corrected implementation. Existing
                # FP32 SGEMM fixtures keep their unchanged 0.0002 threshold.
                tf32_oracle = module == "sgemm" and (
                    name == "sgemm_cublas_tf32"
                    or name.startswith("sgemm_wmma_")
                )
                if tf32_oracle:
                    atol = rtol = 0.005
                index = (
                    None
                    if out is None
                    else next(i for i, a in enumerate(inputs) if a is out)
                )
                cases.append(
                    dict(
                        id=f"{module}/{name}/{rows}",
                        module=module,
                        function=name,
                        args=inputs,
                        output_index=index,
                        expected=expected,
                        atol=atol,
                        rtol=rtol,
                        **(
                            {
                                "precision_contract": "Original TF32 cross-platform gate 0.005; AMD independently qualified at 0.0002 with TF32x3"
                            }
                            if tf32_oracle
                            else {}
                        ),
                    )
                )
    for c in cases:
        c["args"] = tuple(
            t.cpu() if isinstance(t, torch.Tensor) else t for t in c["args"]
        )
        c["expected"] = c["expected"].cpu()
        c["tensor_sha256"] = [
            tensor_sha(t) for t in c["args"] if isinstance(t, torch.Tensor)
        ]
    payload = {
        "modules": modules,
        "cases": cases,
        "qualification_sha256": sha(args.qualification),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"Saved {len(cases)} cases / {sum(map(len, modules.values()))} interfaces; {sha(args.output)}",
        flush=True,
    )


def cuda_build(module, names, original, cutlass):
    from torch.utils.cpp_extension import load

    extras = {
        "flash_attn": (
            "flash-attn/cutlass/flash_attn_cute.cu",
            "flash-attn/mma/basic/flash_attn_mma_share_kv.cu",
            "flash-attn/mma/basic/flash_attn_mma_share_qkv.cu",
            "flash-attn/mma/basic/flash_attn_mma_split_kv.cu",
            "flash-attn/mma/basic/flash_attn_mma_split_q.cu",
            "flash-attn/mma/basic/flash_attn_mma_tiling_qk.cu",
            "flash-attn/mma/basic/flash_attn_mma_tiling_qkv.cu",
            "flash-attn/mma/others/flash_attn_mma_share_qkv_Os2g.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_share_kv_swizzle_q.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_share_kv_swizzle_qk.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_share_kv_swizzle_qkv.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_share_qkv_swizzle_q.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_share_qkv_swizzle_qk.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_share_qkv_swizzle_qkv.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qk_swizzle_q.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qk_swizzle_qk.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qk_swizzle_qkv.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_q.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qk.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qkv.cu",
            "flash-attn/mma/basic/flash_attn_mma_share_kv_F32F16F16F32.cu",
            "flash-attn/mma/basic/flash_attn_mma_share_qkv_F32F16F16F32.cu",
            "flash-attn/mma/basic/flash_attn_mma_tiling_qk_F32F16F16F32.cu",
            "flash-attn/mma/basic/flash_attn_mma_tiling_qkv_F32F16F16F32.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_q_F32F16F16F32.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qk_F32F16F16F32.cu",
            "flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qkv_F32F16F16F32.cu",
            "flash-attn/mma/others/flash_attn_mma_share_kv_F32F16F16F32_rr.cu",
            "flash-attn/mma/others/flash_attn_mma_share_qkv_F32F16F16F32_rr.cu",
        ),
        "hgemv": (
            ("hgemv/hgemv_cute.cu",)
            if "hgemv_tensor_core_cute" in names
            else ()
        ),
        "sgemm": ("sgemm/sgemm_async.cu", "sgemm/sgemm_cublas.cu")
        + (
            ("sgemm/sgemm_wmma_tf32_stage.cu",)
            if any(n.startswith("sgemm_wmma_") for n in names)
            else ()
        ),
        "mat_transpose": ("mat-transpose/mat_transpose_cute.cu",),
        "hgemm": ("hgemm/cublas/hgemm_cublas.cu",)
        + (
            ("hgemm/naive/hgemm_async.cu",)
            if any(
                n.startswith("hgemm_t_") and n.endswith("_async") for n in names
            )
            else ()
        )
        + (
            ("hgemm/wmma/hgemm_wmma.cu", "hgemm/wmma/hgemm_wmma_stage.cu")
            if any(n.startswith("hgemm_wmma_") for n in names)
            else ()
        )
        + (
            (
                "hgemm/mma/basic/hgemm_mma.cu",
                "hgemm/mma/basic/hgemm_mma_stage.cu",
                "hgemm/mma/basic/hgemm_mma_stage_tn.cu",
                "hgemm/mma/swizzle/hgemm_mma_stage_swizzle.cu",
                "hgemm/mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu",
                "hgemm/cutlass/hgemm_mma_stage_tn_cute.cu",
            )
            if any(n.startswith("hgemm_mma_") for n in names)
            else ()
        ),
    }
    paths = [KERNELS[module], *extras.get(module, ())]
    sources = [
        (original / "kernels" / p).read_text(encoding="utf-8-sig")
        for p in paths
    ]
    # Restrict exports to the intended acceptance set. Kernel bodies unchanged.
    source = sources[0]
    start = source.index("PYBIND11_MODULE(")
    opening = source.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    source = (
        source[: opening + 1]
        + "\n"
        + "\n".join(f'm.def("{n}", &{n});' for n in names)
        + "\n}"
        + source[end:]
    )
    sources[0] = source
    flags = [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DNO_CUBLAS_HGEMM_BIN",
        *(
            ["-DNO_MMA_HGEMM_BIN", "-DNO_CUTE_HGEMM_BIN"]
            if module == "hgemm"
            else []
        ),
        *(
            ["-DBUILD_FLASH_ATTN_MMA_OTHERS=1"]
            if module == "flash_attn"
            else []
        ),
    ]
    headers = (
        sorted(
            p
            for p in (original / "kernels/flash-attn").rglob("*")
            if p.suffix in (".h", ".hpp")
        )
        if module == "flash_attn"
        else (
            sorted((original / "kernels/hgemm").rglob("*.h"))
            if module == "hgemm"
            else []
        )
    )
    header_hashes = {str(p.relative_to(original)): sha(p) for p in headers}
    digest = hashlib.sha256(
        (
            "\0".join(sources + flags)
            + CUTLASS
            + (json.dumps(header_hashes, sort_keys=True) if headers else "")
        ).encode()
    ).hexdigest()[:16]
    build = ROOT / ".cache/cuda_comparison" / f"qualified_{module}_{digest}"
    build.mkdir(parents=True, exist_ok=True)
    staged = []
    for i, text in enumerate(sources):
        path = build / f"source_{i}.cu"
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        staged.append(str(path))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ.setdefault("MAX_JOBS", "2")
    lib = load(
        name=f"leetcuda_cuda_qualified_{module}_{digest}",
        sources=staged,
        build_directory=str(build),
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=flags,
        extra_include_paths=[
            str(cutlass / "include"),
            *sorted({str(p.parent) for p in headers}),
        ],
        extra_ldflags=["-lcublas"] if module in ("sgemm", "hgemm") else [],
    )
    return lib, {
        "source_sha256": {
            **header_hashes,
            **{"kernels/" + p: sha(original / "kernels" / p) for p in paths},
        },
        "staged_sha256": {Path(p).name: sha(p) for p in staged},
        "flags": flags,
        "binary_sha256": sha(lib.__file__),
        "binary": lib.__file__,
    }


def run(args):
    from .environment import verify
    from .loader import load, source_paths

    fixture = torch.load(args.fixtures, weights_only=True)
    if args.backend == "hip":
        identity = verify()
    else:
        assert (
            torch.version.hip is None
            and torch.cuda.get_device_capability()
            == (
                12,
                0,
            )
        )
        identity = {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "selected_gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "image": os.environ["LEETCUDA_CUDA_IMAGE"],
            "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
            "nvidia_smi": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            "cutlass_commit": CUTLASS,
        }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "environment": identity,
        "fixture_sha256": sha(args.fixtures),
        "harness_sha256": sha(__file__),
        "upstream_commit": UPSTREAM,
        "source_kind": (
            args.source_kind if args.backend == "cuda" else "rocm_port"
        ),
        "expected_interfaces": fixture["modules"],
        "modules": {},
        "checks": [],
        "errors": [],
        "scope": "Shared finite fixtures only; does not replace full AMD qualification or cover remaining unported interfaces.",
    }
    outputs = {}
    for module, names in fixture["modules"].items():
        if args.modules and module not in args.modules:
            continue
        try:
            if args.backend == "cuda":
                lib, metadata = cuda_build(
                    module, names, args.original, args.cutlass
                )
            else:
                lib = load(module)
                metadata = {
                    "source_sha256": {
                        str(p.relative_to(ROOT)): sha(p)
                        for p in source_paths(module)
                    },
                    "binary_sha256": sha(lib.__file__),
                    "binary": lib.__file__,
                }
            report["modules"][module] = metadata
            for c in fixture["cases"]:
                if c["module"] != module:
                    continue
                assert c["tensor_sha256"] == [
                    tensor_sha(t)
                    for t in c["args"]
                    if isinstance(t, torch.Tensor)
                ]
                inputs = tuple(
                    t.cuda() if isinstance(t, torch.Tensor) else t
                    for t in c["args"]
                )
                fn = getattr(lib, c["function"])

                def call():
                    result = fn(*inputs)
                    return (
                        result
                        if c["output_index"] is None
                        else inputs[c["output_index"]]
                    )

                actual = call().cpu()
                expected = c["expected"]
                passed = actual.shape == expected.shape and torch.allclose(
                    actual, expected, atol=c["atol"], rtol=c["rtol"]
                )
                outputs[c["id"]] = actual
                row = {
                    "id": c["id"],
                    "module": module,
                    "function": c["function"],
                    "passed": passed,
                    "max_abs_error": (
                        (actual.double() - expected.double()).abs().max().item()
                        if actual.shape == expected.shape
                        else None
                    ),
                    "atol": c["atol"],
                    "rtol": c["rtol"],
                    "output_sha256": tensor_sha(actual),
                }
                if row["max_abs_error"] is not None and not math.isfinite(
                    row["max_abs_error"]
                ):
                    row["max_abs_error"] = None
                row["finite_output"] = bool(torch.isfinite(actual).all())
                if args.performance:
                    row["performance"] = timed(call)
                report["checks"].append(row)
            print(
                f"{args.backend}: {module}: {len(names)} interfaces checked",
                flush=True,
            )
        except Exception as exc:
            report["errors"].append({"module": module, "error": str(exc)})
            print(f"{args.backend}: {module}: ERROR {exc}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save(outputs, args.output.with_suffix(".pt"))
    report["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    report["passed"] = not report["errors"] and all(
        c["passed"] for c in report["checks"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["passed"])


def compare(args):
    reports = [json.loads(p.read_text()) for p in (args.cuda, args.hip)]
    assert reports[0]["fixture_sha256"] == reports[1]["fixture_sha256"]
    assert reports[0]["harness_sha256"] == reports[1]["harness_sha256"]
    outputs = []
    for p, r in zip((args.cuda, args.hip), reports):
        assert sha(p.with_suffix(".pt")) == r["outputs_sha256"]
        outputs.append(torch.load(p.with_suffix(".pt"), weights_only=True))
    rows = [{c["id"]: c for c in r["checks"]} for r in reports]
    checks = []
    for key in sorted(set(outputs[0]) & set(outputs[1])):
        a, b = (o[key] for o in outputs)
        cr, hr = (r[key] for r in rows)
        same_shape = a.shape == b.shape
        row = {
            "id": key,
            "module": cr["module"],
            "function": cr["function"],
            "cuda_passed": cr["passed"],
            "hip_passed": hr["passed"],
            "bitwise_equal": same_shape and tensor_sha(a) == tensor_sha(b),
            "parity_within_gate": same_shape
            and torch.allclose(a, b, atol=cr["atol"], rtol=cr["rtol"]),
            "max_abs_cuda_hip": (
                (a.double() - b.double()).abs().max().item()
                if same_shape
                else None
            ),
        }
        if "performance" in cr and "performance" in hr:
            row["cuda_ms"] = cr["performance"]["median_ms"]
            row["hip_ms"] = hr["performance"]["median_ms"]
            row["cuda_time_div_hip_time"] = row["cuda_ms"] / row["hip_ms"]
        checks.append(row)
    summary = {}
    for module in reports[0]["expected_interfaces"]:
        selected = [c for c in checks if c["module"] == module]
        summary[module] = {
            "interfaces": len({c["function"] for c in selected}),
            "cases": len(selected),
            **{
                f: sum(c[f] for c in selected)
                for f in (
                    "cuda_passed",
                    "hip_passed",
                    "bitwise_equal",
                    "parity_within_gate",
                )
            },
        }
    result = {
        "fixture_sha256": reports[0]["fixture_sha256"],
        "report_sha256": {p.name: sha(p) for p in (args.cuda, args.hip)},
        "missing_cuda": sorted(set(outputs[1]) - set(outputs[0])),
        "missing_hip": sorted(set(outputs[0]) - set(outputs[1])),
        "errors": [r["errors"] for r in reports],
        "summary": summary,
        "checks": checks,
        "timing_note": "GPU events, 5 warmups, 5 samples of 20 calls, includes binding/launch overhead. Distinct GPUs and framework stacks; not hardware-normalized.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("fixtures")
    gen.add_argument(
        "--qualification",
        type=Path,
        default=ROOT / "docs/rocm/qualification_20261103_async.json",
    )
    gen.add_argument("--output", type=Path, required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--backend", choices=("cuda", "hip"), required=True)
    runner.add_argument("--fixtures", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--modules", nargs="+", choices=KERNELS)
    runner.add_argument("--original", type=Path, default=ROOT / "original")
    runner.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    runner.add_argument("--performance", action="store_true")
    runner.add_argument(
        "--source-kind",
        choices=("upstream_snapshot", "ported_worktree"),
        default="upstream_snapshot",
    )
    cmp = sub.add_parser("compare")
    cmp.add_argument("--cuda", type=Path, required=True)
    cmp.add_argument("--hip", type=Path, required=True)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixtures":
        create_fixtures(args)
    elif args.command == "run":
        return run(args)
    else:
        compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
