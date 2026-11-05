"""Source-equivalent unbound tiling-QK RR entry, original CUDA and native AMD."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .environment import ROOT
from .cross_validate import sha, tensor_sha
from .unbound import (
    ATTENTION_ENTRY as NAME,
    ATTENTION_SOURCES,
    attention_equivalence,
)
from . import flash_tiled_check as tiled
from .benchmark import graph_pair


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    proof = attention_equivalence()
    assert proof["equal"]
    if args.backend == "hip":
        from .unbound import load_unbound
        from .loader import source_paths
        from .environment import verify

        lib = load_unbound("flash_attn")
        identity = verify()
        meta = dict(
            source_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in source_paths("flash_attn")
            },
            binary_sha256=sha(lib.__file__),
        )
    else:
        from .cuda_pending_check import build

        args.source_root = args.original
        args.cuda_current_stream = True
        group = dict(
            source=ATTENTION_SOURCES[0],
            module="flash_attn",
            functions=[
                dict(
                    name=NAME,
                    declaration=f"void {NAME}(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int);",
                    argc=5,
                )
            ],
        )
        lib, meta = build(group, args)
        identity = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
            gpu=torch.cuda.get_device_name(),
        )
    fn = getattr(lib, NAME)
    rows = torch.load(args.fixtures, weights_only=True)
    r = dict(
        backend=args.backend,
        environment=identity,
        build=meta,
        equivalence=proof,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        helpers_sha256={
            p: sha(ROOT / p)
            for p in (
                "leetcuda_rocm/unbound.py",
                "leetcuda_rocm/flash_tiled_check.py",
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/cuda_pending_check.py",
            )
        },
        checks=[],
        boundaries=[],
        storage=[],
        performance=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def compare(x, ref):
        return bool(
            torch.isfinite(x).all()
            and torch.allclose(x, ref.half(), atol=0.003, rtol=0.003)
        )

    for case in rows:
        assert case["input_sha256"] == [tensor_sha(x) for x in case["inputs"]]
        with torch.cuda.stream(stream):
            inputs = [x.cuda() for x in case["inputs"]]
            shape = inputs[0].shape
            backing = torch.full(
                (inputs[0].numel() + 16,),
                -123,
                device="cuda",
                dtype=torch.float16,
            )
            out = backing[8:-8].view(shape)
            for stage in (1, 2):
                key = case["id"] + f"/s{stage}"
                out.fill_(float("nan"))
                fn(*inputs, out, stage)
                stream.synchronize()
                got = out.cpu()
                r["checks"].append(
                    dict(
                        id=key,
                        passed=compare(got, case["reference"]),
                        max_abs=float(
                            (got.double() - case["reference"]).abs().max()
                        ),
                        canary=bool(
                            (backing[:8] == -123).all()
                            and (backing[-8:] == -123).all()
                        ),
                    )
                )
                outcomes[key] = got
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn(*inputs, out, stage)
                out.fill_(float("nan"))
                graph.replay()
                stream.synchronize()
                replayed = out.cpu()
                r["checks"].append(
                    dict(
                        id=key + "/graph",
                        passed=compare(replayed, case["reference"])
                        and torch.equal(got, replayed),
                        canary=bool(
                            (backing[:8] == -123).all()
                            and (backing[-8:] == -123).all()
                        ),
                    )
                )
                outcomes[key + "/graph"] = replayed
    tiled.NAMES = (NAME,)
    r["storage"] = tiled.storage_checks({NAME: fn})
    if args.backend == "hip":
        r["boundaries"] = tiled.boundaries(lib)
    if args.performance:
        for case in rows:
            if not case["performance"]:
                continue
            inputs = [x.cuda() for x in case["inputs"]]
            out = torch.empty_like(inputs[0])
            for stage in (1, 2):
                op = lambda: fn(*inputs, out, stage)
                pair = graph_pair(
                    op, lambda: F.scaled_dot_product_attention(*inputs)
                )
                pair.update(
                    id=case["id"] + f"/s{stage}",
                    passed=compare(out.cpu(), case["reference"]),
                )
                r["performance"].append(pair)
    torch.save(outcomes, args.output.with_suffix(".pt"))
    r["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    if args.backend == "hip":
        from .isa_objects import hip_isa_objects

        r["isa"] = hip_isa_objects(
            lib.__file__, args.output.with_suffix("").resolve() / "isa"
        )
    else:
        binary = Path(lib.__file__)
        dest = args.output.parent / "binaries" / f"{sha(binary)}.so"
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            dest.write_bytes(binary.read_bytes())
        assert sha(dest) == meta["binary_sha256"]
        r["frozen_binary"] = str(dest.resolve().relative_to(ROOT))
    r["counts"] = {
        k: len(r[k]) for k in ("checks", "storage", "boundaries", "performance")
    }
    r["passed"] = all(
        x["passed"] and x.get("canary", True) for k in r["counts"] for x in r[k]
    )
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print(r["counts"], "PASSED", r["passed"])
    assert r["passed"]


def compare(args):
    a, b = (torch.load(p, weights_only=True) for p in (args.hip, args.cuda))
    assert a.keys() == b.keys()
    checks = [
        dict(
            id=k,
            passed=bool(torch.allclose(a[k], b[k], atol=0.003, rtol=0.003)),
            exact=torch.equal(a[k], b[k]),
            max_abs=float((a[k].float() - b[k].float()).abs().max()),
        )
        for k in a
    ]
    r = dict(
        checks=checks,
        passed=all(x["passed"] for x in checks),
        count=len(checks),
        exact=sum(x["exact"] for x in checks),
        hip_sha256=sha(args.hip),
        cuda_sha256=sha(args.cuda),
    )
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print("COMPARE", r["count"], r["exact"], r["passed"])
    assert r["passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("run")
    a.add_argument("--backend", choices=("hip", "cuda"), required=True)
    a.add_argument("--device", type=int, default=0)
    a.add_argument("--fixtures", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--original", type=Path, default=ROOT / "original")
    a.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    a.add_argument("--performance", action="store_true")
    a = sub.add_parser("compare")
    a.add_argument("--hip", type=Path, required=True)
    a.add_argument("--cuda", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args) if args.command == "run" else compare(args)
