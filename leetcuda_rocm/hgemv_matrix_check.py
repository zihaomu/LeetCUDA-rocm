"""Qualify FP16-I/O HGEMV with explicitly strengthened FP32 WMMA state."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .cross_validate import sha, timed
from .environment import ROOT, verify
from .isa_objects import hip_isa_objects
from .loader import load, source_paths

NAME = "hgemv_tensor_core_cute"


def check(fn, a, b, checks, kind):
    out = torch.full(
        (a.shape[0], 1), float("nan"), dtype=a.dtype, device=a.device
    )
    fn(a, b, out)
    expected = a.double() @ b.double()
    torch.testing.assert_close(out, expected.half(), atol=0.003, rtol=0.003)
    checks.append(
        dict(
            function=NAME,
            case=kind,
            shape_mk=list(a.shape),
            passed=True,
            max_abs_error_fp64=(
                (out.double() - expected).abs().max().item()
                if out.numel()
                else 0.0
            ),
            atol=0.003,
            rtol=0.003,
        )
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.set_grad_enabled(False)
    identity = verify()
    lib = load("hgemv")
    fn = getattr(lib, NAME)
    checks, performance = [], []
    for seed in (20261105, 17, 93):
        torch.manual_seed(seed)
        for m, k in (
            (0, 8),
            (3, 0),
            (1, 1),
            (3, 8),
            (17, 19),
            (65, 256),
            (129, 257),
            (64, 1024),
            (64, 8192),
            (17, 65536),
            (4097, 128),
        ):
            a = (torch.randn(m, k, device="cuda") * 0.25).half()
            b = (torch.randn(k, 1, device="cuda") * 0.25).half()
            check(fn, a, b, checks, f"random-seed-{seed}")
        for pattern in ("zero", "cancellation", "reciprocal-range"):
            a = (torch.randn(65, 4096, device="cuda") * 0.25).half()
            b = (torch.randn(4096, 1, device="cuda") * 0.25).half()
            if pattern == "zero":
                a.zero_()
            elif pattern == "cancellation":
                a[:, 1::2] = a[:, ::2]
                b[1::2] = -b[::2]
            else:
                a[:, ::2] *= 256
                b[::2] /= 256
                a[:, 1::2] /= 256
                b[1::2] *= 256
            check(fn, a, b, checks, pattern)
    a = torch.ones(64, 128, device="cuda", dtype=torch.float16)
    b = torch.ones(128, 1, device="cuda", dtype=torch.float16)
    out = torch.empty(64, 1, device="cuda", dtype=torch.float16)
    invalid = {
        "cpu": lambda: fn(a.cpu(), b, out),
        "input-type": lambda: fn(a.float(), b, out),
        "output-type": lambda: fn(a, b, out.float()),
        "rank": lambda: fn(a.flatten(), b, out),
        "K-mismatch": lambda: fn(a, b[:32], out),
        "output-shape": lambda: fn(a, b, out[:32]),
        "strides": lambda: fn(a[:, ::2], b[:64], out),
        "alias-a": lambda: fn(a, b, a.flatten()[:64].view(64, 1)),
        "alias-b": lambda: fn(a, b, b[:64]),
        "input-grad": lambda: fn(a.clone().requires_grad_(), b, out),
        "output-grad": lambda: fn(a, b, out.clone().requires_grad_()),
    }
    if torch.cuda.device_count() > 1:
        other = (args.device + 1) % torch.cuda.device_count()
        invalid["mixed-devices"] = lambda: fn(a, b.to(f"cuda:{other}"), out)
        with torch.cuda.device(other):
            fn(a, b, out)
            assert torch.cuda.current_device() == other
        torch.cuda.synchronize(args.device)
        assert torch.equal(out, torch.full_like(out, 128))
        checks.append(
            dict(
                function=NAME, case="device-guard-restores-caller", passed=True
            )
        )
    for label, call in invalid.items():
        try:
            call()
        except RuntimeError:
            checks.append(
                dict(function=NAME, case="reject-" + label, passed=True)
            )
        else:
            raise AssertionError("accepted " + label)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    pending = []
    for iteration in range(4):
        for stream in streams:
            with torch.cuda.stream(stream):
                a = (torch.randn(129, 1025, device="cuda") * 0.25).half()
                b = (torch.randn(1025, 1, device="cuda") * 0.25).half()
                out = torch.full(
                    (129, 1), float("nan"), device="cuda", dtype=torch.float16
                )
                fn(a, b, out)
                expected = (a.double() @ b.double()).half()
                pending.append((out, expected))
    for stream in streams:
        stream.synchronize()
    for out, expected in pending:
        torch.testing.assert_close(out, expected, atol=0.003, rtol=0.003)
        checks.append(
            dict(function=NAME, case="concurrent-current-stream", passed=True)
        )
    if args.performance:
        for m, k in ((64, 128), (1024, 1024), (4096, 8192)):
            a = (torch.randn(m, k, device="cuda") * 0.25).half()
            b = (torch.randn(k, 1, device="cuda") * 0.25).half()
            out = torch.empty(m, 1, device="cuda", dtype=torch.float16)
            check(fn, a, b, checks, "performance-input")
            candidate = timed(lambda: fn(a, b, out))
            baseline = timed(lambda: torch.mm(a, b, out=out))
            performance.append(
                dict(
                    shape_mk=[m, k],
                    candidate=candidate,
                    torch_mm=baseline,
                    speedup=baseline["median_ms"] / candidate["median_ms"],
                )
            )
    isa = hip_isa_objects(
        lib.__file__,
        ROOT / f"artifacts/rocm/hgemv_matrix/isa_device{args.device}",
    )
    assert "v_wmma_f32_16x16x16_f16" in isa["instructions"]
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        environment=identity,
        passed=True,
        checks=checks,
        performance=performance,
        isa=isa,
        source_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                *source_paths("hgemv"),
                Path(__file__),
                ROOT / "leetcuda_rocm/loader.py",
            )
        },
        binary_sha256=sha(lib.__file__),
        original_accumulator="FP16",
        rocm_accumulator="FP32",
        semantics="FP16 I/O and real matrix-unit GEMV; explicitly strengthened accumulation, not bitwise FP16 MMA emulation",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(checks)} checks passed; {args.output}", flush=True)


if __name__ == "__main__":
    main()
