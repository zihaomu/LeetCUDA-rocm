"""Independent FP64/Philox qualification of GQA attention and its gradients."""

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
from . import attention_training as implementation
from .cross_validate import sha, tensor_sha
from .environment import ROOT
from .matrix_attention_check import timed_pair

GATES = {torch.float16: (0.003, 0.01), torch.bfloat16: (0.015, 0.03)}


def random_cpu(seed, count, offset=0):
    c0 = np.arange(offset, offset + count, dtype=np.uint64)
    c1 = np.zeros_like(c0)
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    k0 = np.uint64(seed & 0xFFFFFFFF)
    k1 = np.uint64(seed >> 32)
    mask = np.uint64(0xFFFFFFFF)
    for _ in range(10):
        p0 = np.uint64(0xD2511F53) * c0
        p2 = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (
            (p2 >> np.uint64(32)) ^ c1 ^ k0,
            p2 & mask,
            (p0 >> np.uint64(32)) ^ c3 ^ k1,
            p0 & mask,
        )
        k0 = (k0 + np.uint64(0x9E3779B9)) & mask
        k1 = (k1 + np.uint64(0xBB67AE85)) & mask
    norm = np.where(c0 >= 2**31, c0 ^ mask, c0).astype(np.float32)
    return norm * np.float32(4.6566127342e-10)


def reference(q, k, v, mask, causal, p, seed, offset, scale=None):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    scores = (q @ k.transpose(-1, -2)) * (
        1 / math.sqrt(q.shape[-1]) if scale is None else scale
    )
    if mask is not None:
        scores = (
            scores.masked_fill(~mask, float("-inf"))
            if mask.dtype == torch.bool
            else scores + mask
        )
    if causal:
        valid = (
            torch.arange(k.shape[2])[None, :]
            <= torch.arange(q.shape[2])[:, None]
        )
        scores = scores.masked_fill(~valid, float("-inf"))
    valid = (scores != -float("inf")).any(-1, keepdim=True)
    probs = torch.where(valid, scores, 0).softmax(-1) * valid
    if p:
        keep = (
            torch.from_numpy(
                random_cpu(seed, probs.numel(), offset).reshape(probs.shape)
            )
            >= p
        )
        probs = probs * keep / (1 - p)
    return probs @ v


def fixture_case(
    identifier, dtype, d, shape, kind, causal, p, seed, pattern="normal"
):
    b, h, hk, nq, nk = shape
    gen = torch.Generator().manual_seed(seed)
    inputs = [
        torch.randn(*s, generator=gen).to(dtype)
        for s in ((b, h, nq, d), (b, hk, nk, d), (b, hk, nk, d))
    ]
    mask = None
    if kind == "bool":
        mask = torch.rand(nq, nk, generator=gen) > 0.2
        if nq:
            mask[0] = False
    if kind == "add":
        mask = torch.randn(1, 1, nq, nk, generator=gen) * 0.25
        if nq:
            mask[:, :, 0, :] = -float("inf")
    if pattern == "fully_masked":
        mask = torch.zeros(nq, nk, dtype=torch.bool)
    if pattern == "zero_qk":
        inputs[0].zero_()
        inputs[1].zero_()
    if pattern == "large_logits":
        inputs[0].mul_(4)
        inputs[1].mul_(4)
    dy = (torch.randn(inputs[0].shape, generator=gen) * 0.25).to(dtype)
    seed_rng = seed + (1 << 40)
    offset = 37
    q, k, v = [x.double().requires_grad_() for x in inputs]
    y = reference(q, k, v, mask, causal, p, seed_rng, offset)
    grads = torch.autograd.grad(y, (q, k, v), dy.double())
    return dict(
        id=identifier,
        inputs=inputs,
        dy=dy,
        mask=mask,
        causal=causal,
        p=p,
        seed=seed_rng,
        offset=offset,
        expected=[x.detach() for x in (y, *grads)],
        input_sha256=[tensor_sha(x) for x in (*inputs, dy)]
        + ([] if mask is None else [tensor_sha(mask)]),
    )


def fixtures(args):
    torch.set_num_threads(8)
    cases = []
    for dtype in GATES:
        for d in (16, 32, 64, 128, 256):
            for seed in (17, 93):
                for i, (shape, kind, causal, p) in enumerate(
                    [
                        ((1, 2, 2, 17, 31), "none", False, 0.0),
                        ((1, 4, 2, 65, 79), "bool", True, 0.25),
                        ((1, 4, 1, 129, 257), "add", False, 0.125),
                    ]
                ):
                    cases.append(
                        fixture_case(
                            f"{dtype}/{d}/{seed}/{i}",
                            dtype,
                            d,
                            shape,
                            kind,
                            causal,
                            p,
                            seed,
                        )
                    )
        for nq, nk in (
            (1, 1),
            (0, 17),
            (17, 0),
            (0, 0),
            (1, 79),
            (65, 1),
            (257, 129),
        ):
            cases.append(
                fixture_case(
                    f"{dtype}/boundary/{nq},{nk}",
                    dtype,
                    64,
                    (1, 2, 1, nq, nk),
                    "none",
                    True,
                    0.25,
                    17,
                )
            )
        for pattern in ("fully_masked", "zero_qk", "large_logits"):
            cases.append(
                fixture_case(
                    f"{dtype}/{pattern}",
                    dtype,
                    64,
                    (1, 4, 2, 65, 129),
                    "none",
                    False,
                    0.0,
                    93,
                    pattern,
                )
            )
        cases.append(
            fixture_case(
                f"{dtype}/long",
                dtype,
                256,
                (1, 4, 1, 33, 1025),
                "bool",
                False,
                0.25,
                17,
            )
        )
        cases.append(
            fixture_case(
                f"{dtype}/empty_batch",
                dtype,
                32,
                (0, 4, 2, 17, 31),
                "none",
                False,
                0.0,
                17,
            )
        )
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cases=cases, harness_sha256=sha(__file__)), args.fixtures)
    print(json.dumps(dict(cases=len(cases), fixture_sha256=sha(args.fixtures))))


def check(actual, expected, dtype, identifier, checks):
    a = actual.detach().cpu().double()
    e = expected.detach().cpu().double()
    atol, rtol = GATES[dtype]
    delta = a - e
    checks.append(
        dict(
            id=identifier,
            passed=torch.allclose(a, e, atol=atol, rtol=rtol),
            max_abs=delta.abs().max().item() if delta.numel() else 0,
            relative_l2=(delta.norm() / e.norm().clamp_min(1e-300)).item(),
            atol=atol,
            rtol=rtol,
        )
    )


def evaluate(case, device, checks, outputs=None, prefix=""):
    q, k, v = [x.to(device).requires_grad_() for x in case["inputs"]]
    mask = None if case["mask"] is None else case["mask"].to(device)
    options = dict(
        mask=mask,
        causal=case["causal"],
        dropout_p=case["p"],
        seed=case["seed"],
        offset=case["offset"],
    )
    out = implementation.attention(q, k, v, **options)
    grads = torch.autograd.grad(out, (q, k, v), case["dy"].to(device))
    for name, value, expected in zip(
        ("O", "dQ", "dK", "dV"), (out, *grads), case["expected"]
    ):
        identifier = prefix + case["id"] + "/" + name
        check(value, expected, q.dtype, identifier, checks)
        if outputs is not None:
            outputs[identifier] = value.detach().cpu()
    return (q, k, v), options, (out, *grads)


class Capture:
    def __init__(self, kernel, name, records, output, backend):
        self.kernel, self.name, self.records, self.output, self.backend = (
            kernel,
            name,
            records,
            output,
            backend,
        )

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            compiled = self.kernel[grid](*args, **kwargs)
            tag = self.name + "/" + compiled.hash
            if tag not in self.records:
                suffix = "amdgcn" if self.backend == "hip" else "ptx"
                assembly = compiled.asm[suffix]
                instructions = (
                    sorted(set(re.findall(r"v_wmma_[a-z0-9_]+", assembly)))
                    if self.backend == "hip"
                    else sorted(set(re.findall(r"mma\.sync[^\s;]+", assembly)))
                )
                zero_compute = (
                    kwargs.get("NQ", 1) == 0 or kwargs.get("NK", 1) == 0
                )
                if not zero_compute:
                    assert instructions, (
                        self.name,
                        "missing matrix instructions",
                    )
                dest = self.output / f"{self.name}_{compiled.hash}.{suffix}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(assembly)
                binary = compiled.asm[
                    "hsaco" if self.backend == "hip" else "cubin"
                ]
                binary_path = dest.with_suffix(
                    ".hsaco" if self.backend == "hip" else ".cubin"
                )
                binary_path.write_bytes(binary)
                self.records[tag] = dict(
                    kernel=self.name,
                    empty_sequence_zero_fill=zero_compute,
                    instructions=instructions,
                    assembly=str(dest),
                    assembly_sha256=sha(dest),
                    binary_sha256=sha(binary_path),
                    registers=compiled.n_regs,
                    spills=compiled.n_spills,
                    shared_bytes=compiled.metadata.shared,
                )
            return compiled

        return launch


@triton.jit
def _random_probe(Out, Seed, Offset, N: tl.constexpr):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    y = tl.rand(Seed, (x + Offset).to(tl.uint32), n_rounds=10)
    tl.store(Out + x, y, x < N)


def contracts(case, device, guard_device, checks):
    inputs, options, results = evaluate(
        case, device, checks, prefix="contracts/base/"
    )
    q, k, v = inputs
    call = lambda **kw: implementation.attention(q, k, v, **{**options, **kw})
    for label, fn in [
        ("cpu", lambda: implementation.attention(q.cpu(), k.cpu(), v.cpu())),
        ("dtype", lambda: implementation.attention(q.float(), k, v)),
        ("rank", lambda: implementation.attention(q[0], k, v)),
        (
            "head_ratio",
            lambda: implementation.attention(q[:, :3].contiguous(), k, v),
        ),
        ("stride", lambda: implementation.attention(q.transpose(-1, -2), k, v)),
        (
            "mask_shape",
            lambda: call(
                mask=torch.ones(3, 7, device=device, dtype=torch.bool)
            ),
        ),
        (
            "mask_grad",
            lambda: call(
                mask=torch.ones(
                    q.shape[2], k.shape[2], device=device, requires_grad=True
                )
            ),
        ),
        (
            "mask_cpu",
            lambda: call(
                mask=torch.ones(q.shape[2], k.shape[2], dtype=torch.bool)
            ),
        ),
        ("negative_dropout", lambda: call(dropout_p=-0.1)),
        ("unit_dropout", lambda: call(dropout_p=1.0)),
        ("nan_dropout", lambda: call(dropout_p=float("nan"))),
        ("scale_inf", lambda: call(scale=float("inf"))),
        ("seed_negative", lambda: call(seed=-1)),
        ("seed_overflow", lambda: call(seed=2**63)),
        ("offset_wrap", lambda: call(offset=2**32 - 1)),
        ("causal_type", lambda: call(causal=1)),
    ]:
        try:
            fn()
        except (ValueError, TypeError):
            passed = True
        else:
            passed = False
        checks.append(dict(id="reject/" + label, passed=passed))
    # Signed zeros, all 10 Philox rounds, high seed bits and near-wrap offsets.
    for seed, offset in (
        (0, 0),
        (17 + (1 << 40), 37),
        (2**63 - 1, 2**32 - 4096),
    ):
        rng = torch.empty(4096, device=device)
        _random_probe[(16,)](rng, seed, offset, 4096)
        expected = torch.from_numpy(random_cpu(seed, 4096, offset))
        checks.append(
            dict(
                id=f"philox_exact/{seed}/{offset}",
                passed=torch.equal(rng.cpu(), expected),
            )
        )
    # Reproducibility includes gradients; new offsets must change dropout.
    y = call()
    g = torch.autograd.grad(y, (q, k, v), case["dy"].to(device))
    checks.append(
        dict(
            id="dropout_repeat",
            passed=all(torch.equal(a, b) for a, b in zip((y, *g), results)),
        )
    )
    changed = call(
        offset=options["offset"]
        + q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2]
    )
    checks.append(
        dict(id="dropout_new_offset", passed=not torch.equal(y, changed))
    )
    # Capture forward and all gradients, then poison every result before replay.
    dy = case["dy"].to(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_out = call()
        graph_grads = torch.autograd.grad(graph_out, (q, k, v), dy)
    with torch.cuda.stream(capture_stream), torch.no_grad():
        for tensor in (graph_out, *graph_grads):
            tensor.fill_(float("nan"))
        graph.replay()
    capture_stream.synchronize()
    for name, actual, expected in zip(
        ("O", "dQ", "dK", "dV"), (graph_out, *graph_grads), results
    ):
        checks.append(
            dict(
                id="training_graph_replay/" + name,
                passed=torch.equal(actual, expected),
            )
        )
    # Valid storage offsets and noncontiguous broadcastable mask.
    shifted = []
    for x in inputs:
        storage = torch.empty(x.numel() + 1, device=device, dtype=x.dtype)
        t = storage[1:].view_as(x)
        t.copy_(x.detach())
        shifted.append(t.requires_grad_())
    y = implementation.attention(*shifted, **options)
    check(y, case["expected"][0], q.dtype, "unaligned_storage", checks)
    if options["mask"] is not None:
        mask = options["mask"]
        larger = torch.empty(
            (*mask.shape[:-1], mask.shape[-1] * 2),
            device=device,
            dtype=mask.dtype,
        )
        strided = larger[..., ::2]
        strided.copy_(mask)
        check(
            call(mask=strided),
            case["expected"][0],
            q.dtype,
            "strided_mask",
            checks,
        )
    # Keep outputs alive until all streams complete. Exact deterministic replay.
    streams = [torch.cuda.Stream(device=device) for _ in range(6)]
    batches = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            batches.append([call() for _ in range(3)])
    for stream, batch in zip(streams, batches):
        stream.synchronize()
        checks.append(
            dict(
                id="concurrent_stream",
                passed=all(torch.equal(y, results[0]) for y in batch),
            )
        )
    if guard_device is not None:
        old = torch.cuda.current_device()
        torch.cuda.set_device(guard_device)
        try:
            y = call()
            check(
                y, case["expected"][0], q.dtype, "device_guard/output", checks
            )
            checks.append(
                dict(
                    id="device_guard/restored",
                    passed=torch.cuda.current_device() == guard_device,
                )
            )
            for label, fn in (
                (
                    "input",
                    lambda: implementation.attention(q, k.to(guard_device), v),
                ),
                ("mask", lambda: call(mask=options["mask"].to(guard_device))),
            ):
                try:
                    fn()
                except ValueError:
                    passed = True
                else:
                    passed = False
                checks.append(dict(id="mixed_device/" + label, passed=passed))
        finally:
            torch.cuda.set_device(old)


def run(args):
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    if args.backend == "hip":
        from .environment import verify

        environment = verify()
    else:
        # Explicit same-algorithm CUDA comparison, not an upstream CUDA port.
        assert torch.version.cuda and torch.cuda.get_device_capability() == (
            12,
            0,
        )
        implementation._check_environment = lambda device: None
        import os

        environment = dict(
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )
    args.output = args.output.resolve()
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        environment=environment,
        source_kind=(
            "native_rocm_training_extension"
            if args.backend == "hip"
            else "same_triton_algorithm_cuda_reference"
        ),
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        implementation_sha256=sha(implementation.__file__),
        checks=[],
        isa={},
        passed=False,
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
    fixture = torch.load(args.fixtures, weights_only=True)
    outputs = {}
    device = torch.device("cuda", args.device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        for case in fixture["cases"]:
            evaluate(case, device, report["checks"], outputs)
            print(case["id"], "checked", flush=True)
        chosen = next(
            c for c in fixture["cases"] if c["id"] == "torch.float16/32/17/1"
        )
        contracts(chosen, device, args.guard_device, report["checks"])
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
                isa=len(report["isa"]),
            )
        )
    )
    return int(not report["passed"])


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("fixtures")
    g.add_argument("--fixtures", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--fixtures", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--backend", choices=("hip", "cuda"), default="hip")
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--guard-device", type=int)
    args = p.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
