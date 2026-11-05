"""Identical-input CUDA/HIP experiment; parity and accuracy are separate gates.

CUDA builds original source snapshots, with a small generated binding for GEMM.
HIP builds private research candidates. This does not publish rejected kernels.
Fixtures contain CPU tensors and independently evaluated CPU references only.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "3dbd618e703943e6761db05f29dd66cb7de94d13"
CUTLASS = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
HGEMM = (
    "hgemm_naive_f16",
    "hgemm_sliced_k_f16",
    "hgemm_t_8x8_sliced_k_f16x4",
    "hgemm_t_8x8_sliced_k_f16x4_pack",
    "hgemm_t_8x8_sliced_k_f16x4_bcf",
    "hgemm_t_8x8_sliced_k_f16x4_pack_bcf",
    "hgemm_t_8x8_sliced_k_f16x8_pack_bcf",
    "hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf",
    "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf",
    "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf",
    "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf",
)
HGEMV = (
    "hgemv_f16_cute",
    "hgemv_f16x8_cute",
    "hgemv_tensor_core_cute",
)
REDUCE = tuple(
    f"block_all_reduce_sum_{pack}_bf16"
    for pack in ("bf16", "bf16x2", "bf16x8_pack")
)
NAMES = {"hgemm": HGEMM, "hgemv": HGEMV, "reduce": REDUCE}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def binding(names):
    return (
        "#include <torch/extension.h>\n"
        + "\n".join(
            f"void {n}(torch::Tensor, torch::Tensor, torch::Tensor);"
            for n in names
        )
        + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        + "\n".join(f'm.def("{n}", &{n});' for n in names)
        + "\n}\n"
    )


def reduce_model(x, pack):
    # CPU BF16 rounding at each addition, followed by the original XOR tree.
    parts = x.reshape(x.shape[0], -1, pack)
    lanes = torch.zeros(parts.shape[:2], dtype=torch.bfloat16)
    for i in range(pack):
        lanes = (lanes.double() + parts[:, :, i].double()).bfloat16()
    warps = lanes.reshape(x.shape[0], -1, 32)
    ids = torch.arange(32)
    for shift in (16, 8, 4, 2, 1):
        warps = (warps.double() + warps[:, :, ids ^ shift].double()).bfloat16()
    leaders = warps[:, :, 0]
    ids = torch.arange(leaders.shape[1])
    shift = leaders.shape[1] // 2
    while shift:
        leaders = (
            leaders.double() + leaders[:, ids ^ shift].double()
        ).bfloat16()
        shift //= 2
    # These small BF16 block sums add exactly in FP32 for the fixture range.
    return leaders[:, 0].float().sum().reshape(1)


def half_round(value):
    # NumPy converts FP64 directly to FP16. Torch's CPU conversion goes via
    # FP32, which double-rounds some halfway cases and is not an F16 FMA oracle.
    return torch.from_numpy(np.asarray(value).astype(np.float16))


def gemv_models(a, b):
    m, k = a.shape
    padded = (k + 255) // 256 * 256
    products = torch.zeros(m, padded, dtype=torch.float64)
    products[:, :k] = a.double() * b.T.double()
    refs = {}
    for name, block in ((HGEMV[0], 32), (HGEMV[1], 256)):
        lanes = torch.zeros(m, 32, dtype=torch.float16)
        for base in range(0, padded, block):
            tile = products[:, base : base + block]
            if block == 32:
                lanes = half_round(lanes.double() + tile)
            else:
                tile = tile.reshape(m, 32, 8)
                pair = torch.zeros(m, 32, 2, dtype=torch.float16)
                for index in range(0, 8, 2):
                    pair = half_round(
                        pair.double() + tile[:, :, index : index + 2]
                    )
                local = half_round(
                    pair[:, :, 0].double() + pair[:, :, 1].double()
                )
                lanes = half_round(lanes.double() + local.double())
        index = torch.arange(32)
        for shift in (16, 8, 4, 2, 1):
            lanes = half_round(
                lanes.double() + lanes[:, index ^ shift].double()
            )
        refs[name] = lanes[:, :1]
    accumulator = torch.zeros(m, 1, dtype=torch.float16)
    for base in range(0, padded, 16):
        accumulator = half_round(
            accumulator.double()
            + products[:, base : base + 16].sum(1, keepdim=True)
        )
    refs[HGEMV[2]] = accumulator
    return refs


def fixtures(path):

    cases = []
    for seed in (20261105, 20261106, 20261107):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def rand(shape, dtype=torch.float16):
            return (torch.randn(shape, generator=generator) * 0.25).to(dtype)

        for m, n, k in ((128, 256, 32), (256, 256, 256), (256, 256, 1024)):
            a, b = rand((m, k)), rand((k, n))
            reference = a.double() @ b.double()
            model = torch.zeros((m, n), dtype=torch.float16)
            for column in range(k):
                model = half_round(
                    model.double()
                    + a[:, column, None].double() * b[column, None, :].double()
                )
            cases.append(
                dict(
                    id=f"hgemm_{seed}_{m}_{n}_{k}",
                    module="hgemm",
                    names=HGEMM,
                    inputs=(a, b),
                    reference=reference,
                    models={name: model for name in HGEMM},
                    atol=0.01,
                    rtol=0.01,
                )
            )
        for m, k in ((17, 19), (64, 128), (65, 256), (64, 1024), (64, 8192)):
            a, b = rand((m, k)), rand((k, 1))
            names = tuple(n for n in HGEMV if "x8" not in n or k % 8 == 0)
            models = gemv_models(a, b)
            cases.append(
                dict(
                    id=f"hgemv_{seed}_{m}_{k}",
                    module="hgemv",
                    names=names,
                    inputs=(a, b),
                    reference=a.double() @ b.double(),
                    models={n: models[n] for n in names},
                    atol=0.003,
                    rtol=0.003,
                )
            )
        for rows in (8, 16, 128):
            x = rand((rows, 1024), torch.bfloat16)
            cases.append(
                dict(
                    id=f"reduce_{seed}_{rows}",
                    module="reduce",
                    names=REDUCE,
                    inputs=(x,),
                    reference=x.double().sum().reshape(1),
                    models={
                        n: reduce_model(x, p) for n, p in zip(REDUCE, (1, 2, 8))
                    },
                    atol=0.15,
                    rtol=0.01,
                )
            )
    for case in cases:
        case["input_sha256"] = [tensor_sha(t) for t in case["inputs"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": 1, "cases": cases}, path)
    print(f"Saved {len(cases)} CPU fixtures, SHA256={sha(path)}", flush=True)


def build(module, backend, original, cutlass):
    if backend == "hip":
        from .environment import cache_key, verify
        from .loader import _load, _prepare

        identity = verify()
        paths = {
            "hgemm": ["hgemm/naive/hgemm.cu", "hgemm/naive/hgemm_async.cu"],
            "hgemv": ["rocm/hgemv_layouts.cu"],
            "reduce": ["reduce/block_all_reduce.cu"],
        }[module]
        sources = [(ROOT / "kernels" / p).read_text() for p in paths]
        if module == "reduce":
            token = "#ifndef LEETCUDA_ROCM\n  // These teaching variants"
            assert sources[0].count(token) == 1
            sources[0] = sources[0].replace(
                token, "#if 1\n  // These teaching variants"
            )
        else:
            sources.insert(0, binding(NAMES[module]))
        lib = _load(
            "cross_" + module,
            tuple(_prepare(s) for s in sources),
            cache_key(identity),
            identity["gpu_architecture"],
            False,
        )
        source_hashes = {
            "kernels/" + p: sha(ROOT / "kernels" / p) for p in paths
        }
        flags = ["-O3", "-std=c++20", "-DLEETCUDA_ROCM=1"]
    else:
        from torch.utils.cpp_extension import load

        paths = {
            "hgemm": ["hgemm/naive/hgemm.cu", "hgemm/naive/hgemm_async.cu"],
            "hgemv": ["hgemv/hgemv_cute.cu"],
            "reduce": ["reduce/block_all_reduce.cu"],
        }[module]
        sources = [
            (original / "kernels" / p).read_text(encoding="utf-8-sig")
            for p in paths
        ]
        if module != "reduce":
            sources.insert(0, binding(NAMES[module]))
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
        ]
        digest = hashlib.sha256(
            ("\0".join(sources + flags) + CUTLASS).encode()
        ).hexdigest()[:16]
        directory = ROOT / ".cache/cuda_comparison" / f"{module}_{digest}"
        directory.mkdir(parents=True, exist_ok=True)
        staged = []
        for index, source in enumerate(sources):
            file = directory / f"source_{index}.cu"
            file.write_text(source)
            staged.append(str(file))
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
        os.environ.setdefault("MAX_JOBS", "2")
        lib = load(
            name=f"leetcuda_cuda_{module}_{digest}",
            sources=staged,
            build_directory=str(directory),
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=flags,
            extra_include_paths=[str(cutlass / "include")],
        )
        source_hashes = {
            "kernels/" + p: sha(original / "kernels" / p) for p in paths
        }
    return lib, {
        "source_sha256": source_hashes,
        "adapted_source_sha256": [
            hashlib.sha256(s.encode()).hexdigest() for s in sources
        ],
        "binary_sha256": sha(lib.__file__),
        "binary": lib.__file__,
        "flags": flags,
    }


def timed(fn):
    for _ in range(5):
        fn()
    samples = []
    for _ in range(5):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 20)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def run(args):
    backend = args.backend
    if backend == "hip":
        from .environment import verify

        environment = verify()
    else:
        assert (
            torch.version.hip is None
            and torch.cuda.get_device_capability()
            == (
                12,
                0,
            )
        )
        environment = {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(),
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
    fixture = torch.load(args.fixtures, map_location="cpu", weights_only=True)
    report = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "environment": environment,
        "upstream_commit": UPSTREAM,
        "fixture_sha256": sha(args.fixtures),
        "harness_sha256": sha(__file__),
        "qualification": "research; parity and FP64 accuracy reported separately",
        "modules": {},
        "checks": [],
        "errors": [],
        "timing_method": "GPU events; 5 warmups, 5 samples x 20 calls; includes launch and binding overhead",
    }
    outputs = {}
    for module in args.modules:
        try:
            lib, metadata = build(module, backend, args.original, args.cutlass)
            report["modules"][module] = metadata
            for case in fixture["cases"]:
                if case["module"] != module:
                    continue
                assert [tensor_sha(t) for t in case["inputs"]] == case[
                    "input_sha256"
                ]
                inputs = tuple(t.cuda() for t in case["inputs"])
                reference = case["reference"]
                for name in case["names"]:
                    fn = getattr(lib, name)
                    out = torch.full(
                        reference.shape,
                        float("nan"),
                        dtype=torch.float16,
                        device="cuda",
                    )

                    def call():
                        if module == "reduce":
                            return fn(*inputs)
                        fn(*inputs, out)
                        return out

                    torch.cuda.synchronize()
                    actual = call().cpu()
                    repeated = [call().cpu() for _ in range(2)]
                    key = case["id"] + "/" + name
                    outputs[key] = actual
                    error = (actual.double() - reference).abs()
                    target = reference.to(actual.dtype)
                    row = {
                        "id": key,
                        "module": module,
                        "function": name,
                        "input_sha256": case["input_sha256"],
                        "shape": [list(t.shape) for t in case["inputs"]],
                        "finite": bool(actual.isfinite().all()),
                        "repeat_equal": all(
                            torch.equal(actual, t) for t in repeated
                        ),
                        "fp64_accuracy_passed": torch.allclose(
                            actual, target, atol=case["atol"], rtol=case["rtol"]
                        ),
                        "atol": case["atol"],
                        "rtol": case["rtol"],
                        "max_abs_error_fp64": error.max().item(),
                        "relative_l2_error_fp64": (
                            error.norm() / reference.norm().clamp_min(1e-30)
                        ).item(),
                        "matches_arithmetic_model": torch.equal(
                            actual, case["models"][name]
                        ),
                        "model_is_hypothesis": name == "hgemv_tensor_core_cute",
                        "output_sha256": tensor_sha(actual),
                    }
                    if "20261105" in case["id"] and args.performance:
                        row["performance"] = timed(call)
                    report["checks"].append(row)
                print(f"{backend}: {case['id']} finished", flush=True)
        except Exception as exc:
            report["errors"].append({"module": module, "error": str(exc)})
            print(f"{backend}: {module} ERROR {exc}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save(outputs, args.output.with_suffix(".pt"))
    report["execution_passed"] = not report["errors"]
    report["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(bool(report["errors"]))


def compare(args):
    reports = [json.loads(p.read_text()) for p in (args.cuda, args.hip)]
    assert reports[0]["fixture_sha256"] == reports[1]["fixture_sha256"]
    assert reports[0]["harness_sha256"] == reports[1]["harness_sha256"]
    outputs = []
    for path, report in zip((args.cuda, args.hip), reports):
        assert sha(path.with_suffix(".pt")) == report["outputs_sha256"]
        outputs.append(torch.load(path.with_suffix(".pt"), weights_only=True))
    rows = [{r["id"]: r for r in report["checks"]} for report in reports]
    checked = []
    for key in sorted(set(outputs[0]) & set(outputs[1])):
        cuda, hip = [o[key] for o in outputs]
        cr, hr = [r[key] for r in rows]
        assert cr["input_sha256"] == hr["input_sha256"]
        row = {
            "id": key,
            "function": cr["function"],
            "module": cr["module"],
            "bitwise_equal": tensor_sha(cuda) == tensor_sha(hip),
            "max_abs_cuda_hip": (cuda.double() - hip.double())
            .abs()
            .max()
            .item(),
            "cuda_accuracy": cr["fp64_accuracy_passed"],
            "hip_accuracy": hr["fp64_accuracy_passed"],
            "cuda_model_equal": cr["matches_arithmetic_model"],
            "hip_model_equal": hr["matches_arithmetic_model"],
            "cuda_repeat_equal": cr["repeat_equal"],
            "hip_repeat_equal": hr["repeat_equal"],
        }
        if "performance" in cr and "performance" in hr:
            row["cuda_ms"] = cr["performance"]["median_ms"]
            row["hip_ms"] = hr["performance"]["median_ms"]
            row["cuda_time_div_hip_time"] = row["cuda_ms"] / row["hip_ms"]
        checked.append(row)
    summary = {}
    for name in sorted({r["function"] for r in checked}):
        selected = [r for r in checked if r["function"] == name]
        summary[name] = {
            "cases": len(selected),
            **{
                field: sum(r[field] for r in selected)
                for field in (
                    "bitwise_equal",
                    "cuda_accuracy",
                    "hip_accuracy",
                    "cuda_model_equal",
                    "hip_model_equal",
                )
            },
        }
    result = {
        "fixture_sha256": reports[0]["fixture_sha256"],
        "report_sha256": {p.name: sha(p) for p in (args.cuda, args.hip)},
        "missing_cuda": sorted(set(outputs[1]) - set(outputs[0])),
        "missing_hip": sorted(set(outputs[0]) - set(outputs[1])),
        "execution_errors": [r["errors"] for r in reports],
        "summary": summary,
        "checks": checked,
        "note": "Research evidence, not promotion to the default qualified interface. CUDA and HIP use different hardware/framework stacks; timing is measured, not a hardware-normalized ranking.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixtures")
    fixture.add_argument("--output", type=Path, required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--backend", choices=("cuda", "hip"), required=True)
    runner.add_argument("--fixtures", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument(
        "--modules", nargs="+", choices=NAMES, default=list(NAMES)
    )
    runner.add_argument("--original", type=Path, default=ROOT / "original")
    runner.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    runner.add_argument("--performance", action="store_true")
    cmp = sub.add_parser("compare")
    cmp.add_argument("--cuda", type=Path, required=True)
    cmp.add_argument("--hip", type=Path, required=True)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixtures":
        fixtures(args.output)
    elif args.command == "run":
        return run(args)
    else:
        compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
