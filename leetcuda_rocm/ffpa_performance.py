"""FFPA same-input local SDPA comparisons, graph replay and linear storage."""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmark import graph_pair
from .cross_validate import sha
from .environment import ROOT
from .ffpa import HEAD_DIMS, VARIANTS
from .ffpa_check import STAGES, accurate, cuda_build


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--backend", choices=("hip", "cuda"), required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--cuda-tail-fix", action="store_true")
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--original", type=Path, default=ROOT / "original")
    p.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    fixture = torch.load(args.fixtures, map_location="cpu", weights_only=True)
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        reference_harness_sha256=sha(ROOT / "leetcuda_rocm/ffpa_check.py"),
        method="8 calls per graph, five alternating samples targeting 40ms each; same-GPU Torch SDPA; allocation excluded from GPU replay",
        builds={},
        checks=[],
        performance=[],
        memory=[],
        passed=False,
    )
    if args.backend == "hip":
        from .environment import verify
        from .ffpa import _native

        lib = _native(args.device)
        report["environment"] = verify()
        report["sources_sha256"] = {
            s: sha(ROOT / s)
            for s in (
                "kernels/rocm/ffpa_attention_hip.cuh",
                "leetcuda_rocm/native/ffpa_bindings.cpp",
            )
        }
    else:
        report["environment"] = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for d in HEAD_DIMS:
        if args.backend == "cuda":
            lib, report["builds"][d] = cuda_build(
                d, args.original, args.cutlass, args.cuda_tail_fix
            )
        c = next(c for c in fixture["cases"] if c["id"] == f"random/{d}/17/384")
        xs = [c[k].to(args.device) for k in ("q", "k", "v")]
        o = torch.empty_like(xs[0])
        l = torch.empty(xs[0].shape[:-1], device=args.device)
        baseline = lambda: F.scaled_dot_product_attention(*xs)
        b = baseline()
        a = accurate(b, c["expected"])
        assert a["passed"], a
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            baseline()
        backend_ops = [
            e.key for e in prof.key_averages() if "scaled_dot_product" in e.key
        ]
        report["checks"].append(
            dict(id=f"baseline/{d}", **a, operators=backend_ops)
        )
        for variant in VARIANTS:
            for sq, sv in STAGES:

                def fn():
                    if args.backend == "hip":
                        lib.forward(
                            *xs, o, l, None, VARIANTS[variant], sq, sv, 0
                        )
                    else:
                        lib.forward(*xs, o, l, VARIANTS[variant], sq, sv)

                fn()
                torch.cuda.synchronize()
                a = accurate(o, c["expected"])
                le = (
                    accurate(l, c["lse"], True)
                    if args.backend == "hip" or variant == "non_ws"
                    else None
                )
                assert a["passed"] and (le is None or le["passed"]), (
                    d,
                    variant,
                    sq,
                    sv,
                    a,
                    le,
                )
                before = o.clone()
                logs = l.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn()
                with torch.cuda.stream(stream):
                    o.fill_(float("nan"))
                    l.fill_(float("nan"))
                    graph.replay()
                stream.synchronize()
                replay = torch.equal(before, o) and (
                    le is None or torch.equal(logs, l)
                )
                assert replay
                row = dict(
                    d=d,
                    n=384,
                    b=1,
                    h=2,
                    variant=variant,
                    sq=sq,
                    sv=sv,
                    output=a,
                    lse=le,
                    graph_replay=replay,
                    passed=True,
                    sdpa_operators=backend_ops,
                    **graph_pair(fn, baseline, calls=8, sample_ms=40, trials=5),
                )
                report["performance"].append(row)
                print(
                    "PERF",
                    d,
                    variant,
                    sq,
                    sv,
                    row["baseline_over_ours"],
                    flush=True,
                )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.backend == "hip":
        from .ffpa import ffpa_attention

        for variant in VARIANTS:
            q = torch.zeros(
                (1, 2, 4096, 128), device=args.device, dtype=torch.float16
            )
            k = torch.zeros_like(q)
            v = torch.ones_like(q)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            result, lse = ffpa_attention(
                q, k, v, variant=variant, return_lse=True
            )
            torch.cuda.synchronize()
            extra = torch.cuda.max_memory_allocated() - base
            linear = q.numel() * 2 + q.shape[0] * q.shape[1] * q.shape[2] * 4
            passed = (
                torch.equal(result, torch.ones_like(result))
                and torch.allclose(
                    lse,
                    torch.full_like(lse, math.log(4096)),
                    atol=2e-5,
                    rtol=2e-5,
                )
                and extra <= linear + 4096
            )
            report["memory"].append(
                dict(
                    variant=variant,
                    shape=list(q.shape),
                    peak_extra_bytes=extra,
                    linear_output_lse_bytes=linear,
                    dense_probability_fp16_bytes=1 * 2 * 4096 * 4096 * 2,
                    passed=passed,
                )
            )
            del q, k, v, result, lse
    report["passed"] = (
        all(
            c["passed"]
            for key in ("checks", "performance", "memory")
            for c in report[key]
        )
        and len(report["performance"]) == 135
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", report["passed"], len(report["performance"]), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
