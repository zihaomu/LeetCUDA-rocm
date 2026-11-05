"""Numerical qualification of original LeetCUDA kernels on the locked profile."""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .blas_checks import blas_checks
from .environment import ROOT, verify
from .extra_checks import gemm_checks, merge_checks
from .loader import KERNELS, load, source_paths
from .transpose_checks import transpose_checks

ACTIVATIONS = {
    "relu": F.relu,
    "gelu": lambda x: F.gelu(x, approximate="tanh"),
    "sigmoid": torch.sigmoid,
    "swish": F.silu,
    "hardswish": F.hardswish,
    "hardshrink": F.hardshrink,
    "elu": F.elu,
}


def dtype_for(name):
    if "bf16" in name:
        return torch.bfloat16
    if "fp8_e4m3" in name:
        return torch.float8_e4m3fn
    if "fp8_e5m2" in name:
        return torch.float8_e5m2
    if "i8" in name:
        return torch.int8
    return torch.float16 if "f16" in name else torch.float32


def random(shape, dtype=torch.float32):
    return (torch.randn(shape, device="cuda") * 0.25).to(dtype)


def nms_reference(boxes, scores, threshold):
    boxes, scores = boxes.cpu(), scores.cpu()
    order = scores.argsort(descending=True, stable=True).tolist()
    keep = []
    while order:
        i, *order = order
        keep.append(i)
        remaining = []
        for j in order:
            wh = (
                torch.minimum(boxes[i, 2:], boxes[j, 2:])
                - torch.maximum(boxes[i, :2], boxes[j, :2])
            ).clamp_min(0)
            inter = wh.prod()
            ai = (boxes[i, 2:] - boxes[i, :2]).prod()
            aj = (boxes[j, 2:] - boxes[j, :2]).prod()
            if inter / (ai + aj - inter) <= threshold:
                remaining.append(j)
        order = remaining
    return torch.tensor(keep, device="cuda", dtype=torch.int64)


def case(module, name, rows):
    if module == "flash_attn":
        n, d, stage = (128, 64, 1) if rows == 8 else (256, 128, 2)
        q, k, v = [random((1, 2, n, d), torch.float16) for _ in range(3)]
        out = torch.full_like(q, float("nan"))
        reference = (
            q.double() @ k.double().transpose(-1, -2) / (d**0.5)
        ).softmax(-1) @ v.double()
        return (q, k, v, out, stage), reference.half(), out, 0.003, 0.003
    """Return arguments, independently computed expected output and tolerance."""
    dtype = torch.float16 if module in ("hgemm", "hgemv") else dtype_for(name)
    x = random((rows, 256), dtype)
    out = (
        torch.full_like(x, float("nan"))
        if dtype.is_floating_point
        else torch.empty_like(x)
    )
    atol, rtol = (2e-5, 2e-5) if dtype == torch.float32 else (3e-3, 3e-3)
    if module in ACTIVATIONS:
        x = torch.randn((rows, 256), device="cuda", dtype=dtype)
        x.view(-1)[:9] = torch.tensor(
            [-8, -3, -0.5, -0.0, 0, 0.5, 1, 3, 8], device="cuda", dtype=dtype
        )
        return (
            (x, out),
            ACTIVATIONS[module](x.float()).to(dtype),
            out,
            atol,
            rtol,
        )
    if module == "elementwise":
        b = random(x.shape, dtype)
        return (x, b, out), x + b, out, 0, 0
    if module == "softmax":
        return (x, out), x.float().softmax(-1).to(dtype), out, atol, rtol
    if module == "layer_norm":
        expected = F.layer_norm(x.float(), (256,), eps=1e-5) * 1.25 - 0.125
        return (x, out, 1.25, -0.125), expected.to(dtype), out, atol, rtol
    if module == "rms_norm":
        expected = (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
            * 1.25
        )
        return (x, out, 1.25), expected.to(dtype), out, atol, rtol
    if module == "rope":
        theta = torch.arange(rows, device="cuda")[:, None] / (
            10000 ** (torch.arange(0, 256, 2, device="cuda") / 256)
        )
        expected = torch.empty_like(x)
        expected[:, ::2] = x[:, ::2] * theta.cos() - x[:, 1::2] * theta.sin()
        expected[:, 1::2] = x[:, ::2] * theta.sin() + x[:, 1::2] * theta.cos()
        return (x, out), expected, out, 2e-5, 2e-5
    if module == "embedding":
        weight = random((37, 256), dtype)
        indices = torch.randint(
            0, 37, (rows,), device="cuda", dtype=torch.int32
        )
        return (indices, weight, out), weight[indices.long()], out, 0, 0
    if module == "histogram":
        x = torch.randint(
            0, 37, (rows * 256,), device="cuda", dtype=torch.int32
        )
        return (x,), torch.bincount(x.long()).int(), None, 0, 0
    if module in ("reduce", "dot_product"):
        # Width 1024 accommodates every packed dispatcher, including int8x16.
        x = random((rows, 1024), dtype)
        if dtype == torch.int8:
            x = torch.randint(-4, 5, x.shape, device="cuda", dtype=dtype)
        if module == "dot_product":
            b = random(x.shape, dtype)
            return (
                (x, b),
                (x.double() * b.double()).sum().float().reshape(1),
                None,
                0.005,
                0.003,
            )
        # Low-precision accumulation variants are deliberately tested separately
        # from FP32 accumulation, with their arithmetic precision reported.
        atol = 0.15 if "bf16" in name and name.endswith("bf16") else 0.025
        if dtype in (torch.float32, torch.int8) or name.endswith("_f32"):
            atol = 0 if dtype == torch.int8 else 2e-4
        return (
            (x,),
            x.double()
            .sum()
            .to(torch.int32 if dtype == torch.int8 else torch.float32)
            .reshape(1),
            None,
            atol,
            (
                0
                if dtype == torch.int8
                else (2e-5 if name.endswith("_f32") else 0.01)
            ),
        )
    if module in ("sgemv", "hgemv"):
        k = 16 if "_k16_" in name else 128
        a, b = random((rows * 8, k), dtype), random((k, 1), dtype)
        out = torch.full(
            (rows * 8, 1), float("nan"), device="cuda", dtype=dtype
        )
        return (a, b, out), (a.double() @ b.double()).to(dtype), out, atol, rtol
    if module in ("sgemm", "hgemm"):
        m, k, n = (128, 128, 256) if rows == 8 else (256, 256, 128)
        if name.startswith("hgemm_wmma_"):
            # Shared CUDA/HIP fixtures must respect the original 256-row tile.
            # AMD's extended masked-tail contract is exercised separately.
            m = 256 if rows == 8 else 512
        if module == "hgemm" or "_sliced_k16_" in name:
            n = 256
        elif rows == 16 and "_dbuf" in name:
            k = 24  # Three BK=8 tiles: exercise odd final-buffer parity.
        a, b = random((m, k), dtype), random((k, n), dtype)
        out = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
        tol = 0.01 if module == "hgemm" else 2e-4
        reference_b = b.view(n, k).T if "_tn" in name else b
        return (
            (
                (a, b, out, 2 if rows == 8 else 3, rows == 16, 256)
                if "_stages" in name
                else (a, b, out)
            ),
            (a.double() @ reference_b.double()).to(dtype),
            out,
            tol,
            tol,
        )
    if module == "mat_transpose":
        x = random((256, 256 if rows == 8 else 512))
        out = torch.full((x.shape[1], x.shape[0]), float("nan"), device="cuda")
        return (x, out), x.T.contiguous(), out, 0, 0
    if module == "nms":
        xy = torch.rand((rows * 4 + 1, 2), device="cuda")
        boxes = torch.cat((xy, xy + torch.rand_like(xy) * 0.7 + 0.1), dim=1)
        scores = torch.rand(boxes.shape[0], device="cuda")
        return (
            (boxes, scores, 0.3),
            nms_reference(boxes, scores, 0.3),
            None,
            0,
            0,
        )
    raise AssertionError(module)


def run_module(module, bench=False):
    lib = load(module)
    if module == "merge_attn_states":
        return merge_checks(lib)
    results = []
    for name in sorted(n for n in dir(lib) if not n.startswith("_")):
        if name in ("init_cublas_handle", "destroy_cublas_handle"):
            continue
        fn = getattr(lib, name)
        for rows in (8, 16):
            # Every case runs on a non-default stream. Constructing inputs,
            # reference and outputs here also exercises allocator/stream order.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                args, expected, out, atol, rtol = case(module, name, rows)
                result = fn(*args)
                actual = result if out is None else out
                failure = None
                try:
                    torch.testing.assert_close(
                        actual, expected, atol=atol, rtol=rtol
                    )
                except AssertionError as exc:
                    failure = str(exc)
                error = (actual.double() - expected.double()).abs().max().item()
                record = {
                    "function": name,
                    "rows": rows,
                    "tensor_arguments": [
                        {"shape": list(arg.shape), "dtype": str(arg.dtype)}
                        for arg in args
                        if isinstance(arg, torch.Tensor)
                    ],
                    "max_abs_error": error,
                    "atol": atol,
                    "rtol": rtol,
                    "stream": "non-default",
                    "passed": failure is None,
                }
                if failure:
                    record["error"] = failure
                    print(
                        f"[{module}.{name}, rows={rows}] FAIL: {failure}",
                        flush=True,
                    )
                if bench:
                    for _ in range(3):
                        fn(*args)
                    start, end = torch.cuda.Event(
                        enable_timing=True
                    ), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        fn(*args)
                    end.record()
                    end.synchronize()
                    record["milliseconds"] = start.elapsed_time(end) / 20
                results.append(record)
            stream.synchronize()
    # Boundary cases that are supported by the original scalar/tail paths.
    with torch.cuda.stream(torch.cuda.Stream()):
        if module == "elementwise":
            for name in sorted(
                n for n in dir(lib) if n.startswith("elementwise_add_")
            ):
                dtype = dtype_for(name)
                for length in (1, 7, 255, 257, 1025):
                    a, b = random((length,), dtype), random((length,), dtype)
                    out = torch.full_like(a, float("nan"))
                    getattr(lib, name)(a, b, out)
                    torch.testing.assert_close(out, a + b, atol=0, rtol=0)
                    results.append(
                        {
                            "function": name,
                            "shape": [length],
                            "case": "tail",
                            "passed": True,
                        }
                    )
        if module in ACTIVATIONS:
            for dtype, suffix in (
                (torch.float32, "f32"),
                (torch.float16, "f16"),
            ):
                name = f"{module}_{suffix}"
                x = random((257,), dtype)
                out = torch.full_like(x, float("nan"))
                getattr(lib, name)(x, out)
                torch.testing.assert_close(
                    out,
                    ACTIVATIONS[module](x.float()).to(dtype),
                    atol=0.003,
                    rtol=0.003,
                )
                results.append(
                    {
                        "function": name,
                        "shape": [257],
                        "case": "scalar-tail",
                        "passed": True,
                    }
                )
        if module == "nms":
            boxes = torch.empty((0, 4), device="cuda")
            scores = torch.empty((0,), device="cuda")
            torch.testing.assert_close(
                lib.nms(boxes, scores, 0.5),
                torch.empty(0, device="cuda", dtype=torch.int64),
            )
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 65, device="cuda")
            scores = torch.ones(65, device="cuda")
            torch.testing.assert_close(
                lib.nms(boxes, scores, 0.5), torch.tensor([0], device="cuda")
            )
            results.extend(
                [
                    {"function": "nms", "case": key, "passed": True}
                    for key in ("empty", "ties-65-boxes")
                ]
            )
        torch.cuda.current_stream().synchronize()
    if module in ("sgemm", "hgemm"):
        results.extend(gemm_checks(module, lib))
        results.extend(blas_checks(module, lib))
    if module == "mat_transpose":
        results.extend(transpose_checks(lib))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="logical GPU index within HIP_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--modules", nargs="+", choices=list(KERNELS), default=list(KERNELS)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/rocm/qualification.json",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="also record HIP event timings; no speedup claim",
    )
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261102)
    torch.set_grad_enabled(False)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": verify(),
        "logical_device": args.device,
        "excluded_functions": {
            "reduce": {
                name: "BF16 accumulation exceeds the numerical threshold; use the named FP32 accumulator"
                for name in (
                    "block_all_reduce_sum_bf16_bf16",
                    "block_all_reduce_sum_bf16x2_bf16",
                    "block_all_reduce_sum_bf16x8_pack_bf16",
                )
            }
        },
        "seed": 20261102,
        "modules": {},
        "passed": False,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256(
                (ROOT / "kernels" / KERNELS[name]).read_bytes()
            ).hexdigest()
            for name in args.modules
        },
        "source_files_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for name in args.modules
            for path in source_paths(name)
        },
    }
    report["adapter_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((ROOT / "leetcuda_rocm").glob("*.py"))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for module in args.modules:
        start = time.monotonic()
        print(f"[{module}] compile + numerical checks", flush=True)
        try:
            checks = run_module(module, args.bench)
            report["modules"][module] = {
                "passed": all(check["passed"] for check in checks),
                "checks": checks,
                "binary_sha256": hashlib.sha256(
                    Path(load(module).__file__).read_bytes()
                ).hexdigest(),
            }
            status = "PASS" if report["modules"][module]["passed"] else "FAIL"
            print(f"[{module}] {status} ({len(checks)} cases)", flush=True)
        except Exception as exc:
            report["modules"][module] = {"passed": False, "error": str(exc)}
            traceback.print_exc()
        report["modules"][module]["elapsed_seconds"] = time.monotonic() - start
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["passed"] = all(
        item["passed"] for item in report["modules"].values()
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Qualification passed={report['passed']}; report={args.output}",
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
