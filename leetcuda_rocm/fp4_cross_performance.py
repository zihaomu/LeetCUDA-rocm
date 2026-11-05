"""Measure complete NVFP4 paths against each GPU's local BF16 GEMM."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import fp4_check as reference
from .cross_validate import sha
from .environment import ROOT
from .fp4_cross_check import cuda_build
from .matrix_attention_check import timed_pair


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--backend", choices=("hip", "cuda"), required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--original", type=Path, default=ROOT / "original")
    p.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    p.add_argument("--compiler-fix", action="store_true")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    fixture = torch.load(args.fixtures, map_location="cpu", weights_only=True)
    case = next(c for c in fixture["cases"] if c["id"] == "17/256,256,1024")
    a, b = (case[key].cuda() for key in ("a", "b"))
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    ref = torch.empty_like(out)
    baseline = lambda: torch.mm(a, b, out=ref)
    baseline()
    torch.testing.assert_close(
        ref.cpu(),
        (case["a"].double() @ case["b"].double()).bfloat16(),
        atol=reference.ATOL,
        rtol=reference.RTOL,
    )
    if args.backend == "cuda":
        lib, build = cuda_build(
            args.original.resolve(), args.cutlass.resolve(), args.compiler_fix
        )
    else:
        from .fp4 import _native
        from .environment import verify

        lib = _native(args.device)
        build = dict(
            binary=lib.__file__,
            binary_sha256=sha(lib.__file__),
            environment=verify(),
        )
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        gpu=torch.cuda.get_device_name(),
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        imported_harness_sha256=sha(ROOT / "leetcuda_rocm/fp4_cross_check.py"),
        build=build,
        performance=[],
        passed=False,
        method="do_bench_cudagraph adaptive repetitions, rep=40ms, five alternating samples; local torch.mm BF16; B preparation separate",
    )
    full_ws = torch.empty(
        lib.workspace_size(m, n, k), device=a.device, dtype=torch.uint8
    )
    act_ws = torch.empty(
        lib.activation_size(m, k), device=a.device, dtype=torch.uint8
    )
    for mode, expected in case["modes"].items():
        code = reference.MODES[mode]
        l1a, l1b = bool(code & 1), bool(code & 2)
        aq, bq, af, bf, sa, sb = (
            expected[key].cuda() for key in ("aq", "bq", "af", "bf", "sa", "sb")
        )
        prepare = timed_pair(
            lambda: lib.quantize_b(b, bq, bf, sb, l1b), baseline
        )
        assert torch.equal(bq.cpu(), expected["bq"]) and torch.equal(
            bf.cpu(), expected["bf"]
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
                        shape_mnk=[m, n, k],
                        mode=mode,
                        tile="wide" if wide else "default",
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
