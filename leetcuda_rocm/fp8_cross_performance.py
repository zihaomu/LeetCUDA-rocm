"""Measure identical FP8 inputs on each GPU against its local BF16 baseline."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import fp8_check as reference
from .cross_validate import sha
from .environment import ROOT
from .fp8_cross_check import cuda_build
from .matrix_attention_check import timed_pair


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=ROOT / "original")
    parser.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    fixture = torch.load(args.fixtures, weights_only=True)
    case = next(c for c in fixture["cases"] if c["id"] == "17/256,256,1024")
    a, b = (case[key].cuda() for key in ("a", "b"))
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    ref = torch.empty_like(out)
    baseline = lambda: torch.mm(a, b, out=ref)
    baseline()
    expected_bf16 = (case["a"].double() @ case["b"].double()).bfloat16()
    torch.testing.assert_close(
        ref.cpu(), expected_bf16, atol=reference.ATOL, rtol=reference.RTOL
    )
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        gpu=torch.cuda.get_device_name(),
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        imported_harness_sha256=sha(ROOT / "leetcuda_rocm/fp8_cross_check.py"),
        builds={},
        performance=[],
        passed=False,
        method="Triton do_bench_cudagraph adaptive graph repetitions, rep=40ms; five alternating samples; local torch.mm BF16 baseline; offline B preparation reported separately",
    )
    for fmt in ("e4m3", "e5m2"):
        if args.backend == "cuda":
            lib, build = cuda_build(
                fmt, args.original.resolve(), args.cutlass.resolve()
            )
        else:
            from .fp8 import _native
            from .environment import verify

            lib = _native(args.device, fmt)
            build = dict(binary_sha256=sha(lib.__file__), environment=verify())
        report["builds"][fmt] = build
        full_ws = torch.empty(
            lib.workspace_size(m, n, k), device=a.device, dtype=torch.uint8
        )
        act_ws = torch.empty(
            lib.activation_size(m, k), device=a.device, dtype=torch.uint8
        )
        for mode, expected in case["formats"][fmt].items():
            row, col = mode.startswith("row"), mode.endswith("col")
            a8, bt8, sa, sb = (
                expected[key].cuda() for key in ("a8", "bt8", "sa", "sb")
            )
            prepare = timed_pair(
                lambda: lib.quantize_b(b, bt8, sb, col), baseline
            )
            assert torch.equal(bt8.cpu(), expected["bt8"])
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
                    fn()
                    torch.testing.assert_close(
                        out.cpu(),
                        expected["expected"],
                        atol=reference.ATOL,
                        rtol=reference.RTOL,
                    )
                    prior = out.clone()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        fn()
                    with torch.cuda.stream(stream):
                        out.fill_(float("nan"))
                        graph.replay()
                    stream.synchronize()
                    assert torch.equal(out, prior)
                    result = dict(
                        fp8_format=fmt,
                        shape_mnk=[m, n, k],
                        mode=mode,
                        use_ws=ws,
                        path=path,
                        passed=True,
                        graph_replay_verified=True,
                        offline_b_preparation=prepare,
                        **timed_pair(fn, baseline)
                    )
                    report["performance"].append(result)
                    print(json.dumps(result), flush=True)
    report["passed"] = len(report["performance"]) == 48
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"], performance=len(report["performance"])
            )
        )
    )


if __name__ == "__main__":
    main()
