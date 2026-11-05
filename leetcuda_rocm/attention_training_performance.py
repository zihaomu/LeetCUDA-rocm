"""Correctness-gated graph timings and workspace checks for training attention."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from . import attention_training as implementation
from .attention_training_check import Capture, check, reference
from .benchmark import graph_pair
from .cross_validate import sha
from .environment import ROOT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", choices=("hip", "cuda"), default="hip")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    if args.backend == "hip":
        from .environment import verify

        environment = verify()
    else:
        assert torch.version.cuda and torch.cuda.get_device_capability() == (
            12,
            0,
        )
        implementation._check_environment = lambda device: None
        import os

        environment = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        environment=environment,
        harness_sha256=sha(__file__),
        implementation_sha256=sha(implementation.__file__),
        checks=[],
        performance=[],
        memory=[],
        isa={},
        passed=False,
        method="8 calls per graph; adaptive 40ms samples; five alternating samples; same-device torch SDPA baseline, forward and forward+backward separately",
    )
    for name in ("_forward", "_delta", "_backward_q", "_backward_kv"):
        setattr(
            implementation,
            name,
            Capture(
                getattr(implementation, name),
                name,
                report["isa"],
                args.output.parent / (args.output.stem + "_isa"),
                args.backend,
            ),
        )
    device = torch.device("cuda", args.device)
    for dtype in (torch.float16, torch.bfloat16):
        for n, d, hk, causal in (
            (256, 64, 1, False),
            (512, 128, 2, True),
            (1024, 256, 4, True),
        ):
            generator = torch.Generator().manual_seed(20261106 + n + d)
            cpu = [
                (torch.randn(shape, generator=generator) * 0.5).to(dtype)
                for shape in ((1, 4, n, d), (1, hk, n, d), (1, hk, n, d))
            ]
            dy = (torch.randn(cpu[0].shape, generator=generator) * 0.25).to(
                dtype
            )
            q, k, v = [x.to(device).requires_grad_() for x in cpu]
            dout = dy.to(device)
            qr, kr, vr = [x.double().requires_grad_() for x in cpu]
            expected_o = reference(qr, kr, vr, None, causal, 0, 0, 0)
            expected = (
                expected_o,
                *torch.autograd.grad(expected_o, (qr, kr, vr), dy.double()),
            )
            base = lambda: F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, enable_gqa=True
            )
            ours = lambda: implementation.attention(q, k, v, causal=causal)
            for label, fn in (("native", ours), ("torch_sdpa", base)):
                out = fn()
                grads = torch.autograd.grad(out, (q, k, v), dout)
                for name, value, ref in zip(
                    ("O", "dQ", "dK", "dV"), (out, *grads), expected
                ):
                    check(
                        value,
                        ref,
                        dtype,
                        f"{dtype}/{n}/{d}/{label}/{name}",
                        report["checks"],
                    )
            if not all(c["passed"] for c in report["checks"]):
                raise AssertionError(
                    "performance inputs failed independent FP64 gate"
                )

            def operation(native, backward):
                def call():
                    # Fresh leaf views give each graph its own AccumulateGrad
                    # nodes on the capture stream, with shared input storage.
                    leaves = tuple(
                        x.detach().requires_grad_() for x in (q, k, v)
                    )
                    if native:
                        out = implementation.attention(*leaves, causal=causal)
                    else:
                        out = F.scaled_dot_product_attention(
                            *leaves, is_causal=causal, enable_gqa=True
                        )
                    if backward:
                        return (out, *torch.autograd.grad(out, leaves, dout))
                    return out

                return call

            # Warm JIT first, then measure PyTorch-allocated live workspace.
            train = operation(True, True)
            warm = train()
            del warm
            torch.cuda.synchronize()
            start = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            result = train()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device) - start
            # Output + dQ/dK/dV + LSE + delta, with allocator-block slack.
            linear = (
                2 * q.numel() + k.numel() + v.numel()
            ) * q.element_size() + 2 * q.shape[0] * q.shape[1] * n * 4
            limit = 2 * linear + 65536
            report["memory"].append(
                dict(
                    dtype=str(dtype),
                    n=n,
                    d=d,
                    hk=hk,
                    peak_extra_bytes=peak,
                    linear_buffer_bytes=linear,
                    limit=limit,
                    passed=peak <= limit,
                )
            )
            assert peak <= limit, "workspace exceeds linear-memory bound"
            del result
            for name, ofn, bfn in (
                ("forward", operation(True, False), operation(False, False)),
                ("forward_backward", train, operation(False, True)),
            ):
                expected_native = ofn()
                prior = (
                    (expected_native,)
                    if isinstance(expected_native, torch.Tensor)
                    else expected_native
                )
                prior = tuple(t.detach().clone() for t in prior)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = ofn()
                captured = (
                    (captured,)
                    if isinstance(captured, torch.Tensor)
                    else captured
                )
                with torch.cuda.stream(stream), torch.no_grad():
                    for t in captured:
                        t.fill_(float("nan"))
                    graph.replay()
                stream.synchronize()
                assert all(torch.equal(a, b) for a, b in zip(captured, prior))
                result = dict(
                    dtype=str(dtype),
                    shape_b_hq_hk_n_d=[1, 4, hk, n, d],
                    causal=causal,
                    path=name,
                    passed=True,
                    graph_replay_verified=True,
                    **graph_pair(ofn, bfn, calls=8, sample_ms=40, trials=5),
                )
                report["performance"].append(result)
                print(json.dumps(result), flush=True)
            del expected, expected_o, qr, kr, vr
    # N >> D makes a dense probability buffer much larger than the allowed
    # linear workspace. Uniform attention has an exact analytical gradient.
    n, d = 4096, 64
    q = torch.zeros(
        (1, 4, n, d), device=device, dtype=torch.float16, requires_grad=True
    )
    k = torch.zeros(
        (1, 1, n, d), device=device, dtype=torch.float16, requires_grad=True
    )
    v = torch.ones_like(k, requires_grad=True)
    dout = torch.ones_like(q)

    def uniform():
        y = implementation.attention(q, k, v)
        return (y, *torch.autograd.grad(y, (q, k, v), dout))

    warm = uniform()
    del warm
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = uniform()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) - start
    linear = (
        2 * q.numel() + k.numel() + v.numel()
    ) * q.element_size() + 2 * 4 * n * 4
    report["memory"].append(
        dict(
            kind="long_uniform_analytic",
            n=n,
            d=d,
            peak_extra_bytes=peak,
            linear_buffer_bytes=linear,
            limit=2 * linear + 65536,
            dense_fp16_probability_bytes=4 * n * n * 2,
            passed=peak <= 2 * linear + 65536,
        )
    )
    for label, value, constant in zip(
        ("O", "dQ", "dK", "dV"), result, (1, 0, 0, 4)
    ):
        report["checks"].append(
            dict(
                id="long_uniform/" + label,
                passed=torch.equal(value, torch.full_like(value, constant)),
            )
        )
    report["passed"] = all(
        c["passed"] for c in report["checks"] + report["memory"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                checks=len(report["checks"]),
                performance=len(report["performance"]),
                memory=len(report["memory"]),
            )
        )
    )


if __name__ == "__main__":
    main()
