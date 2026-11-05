"""Shared-input CPU-reference qualification of the four Triton examples."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from .cross_validate import tensor_sha, timed
from .environment import ROOT

SOURCES = {
    "add": "vector-add/triton_vector_add.py",
    "softmax": "fused-softmax/triton_fused_softmax.py",
    "layer_norm": "layer-norm/triton_layer_norm.py",
    "merge": "merge-attn-states/triton_merge_attn_states.py",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures(path):
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261105)
    cases = []

    def rand(shape, dtype):
        return torch.randn(shape, generator=generator).to(dtype)

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for n in (0, 1, 31, 98432):
            x, y = rand((n,), dtype), rand((n,), dtype)
            cases.append(
                dict(
                    op="add", inputs=[x, y], expected=[x.double() + y.double()]
                )
            )
        for m, n in ((0, 31), (17, 0), (1, 1), (17, 31), (257, 781), (9, 8192)):
            x = rand((m, n), dtype) * 10
            cases.append(
                dict(
                    op="softmax", inputs=[x], expected=[x.double().softmax(-1)]
                )
            )
        for m, n in ((1, 1), (17, 31), (257, 256), (9, 2049)):
            x, w, b = rand((m, n), dtype), rand((n,), dtype), rand((n,), dtype)
            dy = rand((m, n), dtype) * 0.1
            xd, wd, bd = [t.double().requires_grad_() for t in (x, w, b)]
            # Explicit CPU FP64 formula, independent of fused GPU layer norm.
            center = xd - xd.mean(-1, keepdim=True)
            y = (
                center
                * torch.rsqrt(center.square().mean(-1, keepdim=True) + 1e-5)
                * wd
                + bd
            )
            grads = torch.autograd.grad(y, (xd, wd, bd), dy.double())
            cases.append(
                dict(
                    op="layer_norm",
                    inputs=[x, w, b, dy],
                    expected=[y.detach(), *grads],
                )
            )
        for kind in (
            "normal",
            "empty_prefix",
            "empty_suffix",
            "both_empty",
            "fa2_inf",
        ):
            p, s = rand((17, 3, 33), dtype), rand((17, 3, 33), dtype)
            pl, sl = (
                rand((3, 17), torch.float32) * 80,
                rand((3, 17), torch.float32) * 80,
            )
            if kind in ("empty_prefix", "both_empty", "fa2_inf"):
                pl.fill_(float("inf") if kind == "fa2_inf" else -float("inf"))
                p.fill_(float("nan"))
            if kind in ("empty_suffix", "both_empty"):
                sl.fill_(-float("inf"))
                s.fill_(float("nan"))
            l1, l2 = [
                torch.where(t == float("inf"), -float("inf"), t).double()
                for t in (pl, sl)
            ]
            lse = torch.logaddexp(l1, l2)
            w1 = torch.where(l1.isneginf(), 0.0, (l1 - lse).exp()).T[..., None]
            w2 = torch.where(l2.isneginf(), 0.0, (l2 - lse).exp()).T[..., None]
            value = torch.where(w1 == 0, 0.0, p.double() * w1) + torch.where(
                w2 == 0, 0.0, s.double() * w2
            )
            cases.append(
                dict(
                    op="merge",
                    kind=kind,
                    inputs=[p, pl, s, sl],
                    expected=[value, lse],
                )
            )
    for i, case in enumerate(cases):
        case["id"] = f"{case['op']}/{i}"
        case["input_sha256"] = [tensor_sha(t) for t in case["inputs"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, path)


def load_modules():
    modules, hashes = {}, {}
    for name, rel in SOURCES.items():
        path = ROOT / "kernels/openai-triton" / rel
        spec = importlib.util.spec_from_file_location("example_" + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
        hashes[str(path.relative_to(ROOT))] = sha(path)
    return modules, hashes


def run(args):
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    if torch.version.hip:
        from .environment import verify

        env = verify()
    else:
        env = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            triton=triton.__version__,
            gpu=torch.cuda.get_device_name(),
            gpu_uuid=str(torch.cuda.get_device_properties(args.device).uuid),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
        )
    mods, hashes = load_modules()
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        environment=env,
        fixture_sha256=sha(args.fixtures),
        source_sha256=hashes,
        harness_sha256=sha(__file__),
        checks=[],
        performance=[],
        passed=False,
    )
    saved = {}
    device = torch.device("cuda", args.device)
    caller = args.guard_device if args.guard_device is not None else args.device
    torch.cuda.set_device(caller)
    stream = torch.cuda.Stream(device=device)
    for c in torch.load(args.fixtures, weights_only=True):
        assert c["input_sha256"] == [tensor_sha(t) for t in c["inputs"]]
        with torch.cuda.stream(stream):
            inputs = [t.to(device) for t in c["inputs"]]
            stream.synchronize()
        # Keep a non-default stream on the target device even with a different
        # current device. The wrappers must select the target and restore caller.
        torch.cuda.set_stream(stream)
        torch.cuda.set_device(caller)
        op = c["op"]
        with torch.enable_grad():
            if op == "add":
                call = lambda: mods[op].add(*inputs)
                actual = [call()]
            elif op == "softmax":
                call = lambda: mods[op].triton_softmax(inputs[0])
                actual = [call()]
            elif op == "layer_norm":
                x, w, b, dy = inputs
                x.requires_grad_()
                w.requires_grad_()
                b.requires_grad_()
                call = lambda: mods[op].layer_norm(
                    x, (x.shape[-1],), w, b, 1e-5
                )
                y = call()
                actual = [y, *torch.autograd.grad(y, (x, w, b), dy)]
            else:
                p, pl, s, sl = inputs
                out, lse = torch.empty_like(p), torch.empty_like(pl)
                call = lambda: mods[op].merge_attn_states_triton(
                    out, p, pl, s, sl, lse
                )
                call()
                actual = [out, lse]
        report["checks"].append(
            dict(
                id=c["id"] + "/device_guard",
                passed=torch.cuda.current_device() == caller,
            )
        )
        stream.synchronize()
        for index, (value, ref) in enumerate(zip(actual, c["expected"])):
            value = value.detach().cpu()
            expected = ref.to(value.dtype)
            tol = {
                torch.float16: 0.004,
                torch.bfloat16: 0.03,
                torch.float32: 0.0001,
            }[value.dtype]
            row = dict(
                id=c["id"] + f"/{index}",
                passed=torch.allclose(value, expected, atol=tol, rtol=tol),
                atol=tol,
                rtol=tol,
                input_sha256=c["input_sha256"],
            )
            finite = value.isfinite() & ref.isfinite()
            row["max_abs_error_fp64"] = (
                (value.double()[finite] - ref[finite]).abs().max().item()
                if finite.any()
                else 0.0
            )
            saved[row["id"]] = value
            report["checks"].append(row)
        if op == "merge":
            # The optional autotuned kernel is not used by the public wrapper;
            # verify it separately because it is also part of this example.
            with torch.cuda.device(device), torch.cuda.stream(stream):
                mods[op].merge_attn_states_kernel_opt[(p.shape[0], p.shape[1])](
                    out,
                    lse,
                    p,
                    pl,
                    s,
                    sl,
                    p.shape[2],
                    triton.next_power_of_2(p.shape[2]),
                    True,
                )
            stream.synchronize()
            for index, value in enumerate((out, lse)):
                tol = 0.03 if value.dtype == torch.bfloat16 else 0.004
                report["checks"].append(
                    dict(
                        id=c["id"] + f"/autotuned/{index}",
                        passed=torch.allclose(
                            value.cpu(),
                            c["expected"][index].to(value.dtype),
                            atol=tol,
                            rtol=tol,
                        ),
                    )
                )
        if (
            args.performance
            and inputs[0].dtype == torch.float32
            and inputs[0].numel() > 1000
            and op != "merge"
        ):
            with torch.cuda.device(device), torch.cuda.stream(
                stream
            ), torch.no_grad():
                if op == "add":
                    ref_call = lambda: inputs[0] + inputs[1]
                elif op == "softmax":
                    ref_call = lambda: torch.softmax(inputs[0], -1)
                else:
                    ref_call = lambda: torch.nn.functional.layer_norm(
                        x, (x.shape[-1],), w, b, 1e-5
                    )
                report["performance"].append(
                    dict(
                        id=c["id"],
                        shape=list(inputs[0].shape),
                        kernel=timed(call),
                        torch=timed(ref_call),
                    )
                )
        print(c["id"], flush=True)
    report["passed"] = all(r["passed"] for r in report["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output.with_suffix(".pt"))
    report["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(dict(passed=report["passed"], checks=len(report["checks"])))
    )
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["fixtures", "run"])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--guard-device", type=int)
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    if args.command == "fixtures":
        fixtures(args.fixtures)
        return 0
    if args.output is None:
        parser.error("run requires --output")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
