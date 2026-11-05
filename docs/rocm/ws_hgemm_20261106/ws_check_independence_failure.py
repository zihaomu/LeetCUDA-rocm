"""B7 independent producer/consumer GEMM: sequence audit and synchronous control."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha
from .mma_check import numerical, boundaries, fixtures as base_fixtures, packed

VARIANTS = {
    "ws_hgemm_naive_cute": dict(
        tile=[64, 128, 32], stages=3, producers=32, consumers=128
    )
}


def fixtures(path):
    base_fixtures(path)
    rows = torch.load(path, weights_only=True)
    g = torch.Generator().manual_seed(73)
    for k in (16, 32, 48, 64, 96, 544, 1024):
        a = (torch.randn(128, k, generator=g) * 0.25).half()
        b = (torch.randn(k, 256, generator=g) * 0.25).half()
        # All CUDA variants require full BK32 tiles; K16/48 are AMD-only tails.
        if k % 32:
            continue
        rows.append(
            dict(
                id=f"pipeline_k{k}",
                a=a,
                b=b,
                reference=a.double() @ b.double(),
                input_sha256=[tensor_sha(a), tensor_sha(b)],
            )
        )
    torch.save(rows, path)


def controls(identity):
    from .loader import _load, _prepare
    from .environment import cache_key

    bind = """#include <torch/extension.h>
void ws_synchronous_control(torch::Tensor,torch::Tensor,torch::Tensor);
void ws_audit(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("ws_synchronous_control",&ws_synchronous_control);m.def("ws_audit",&ws_audit);}
"""
    source = (
        "#define LEETCUDA_WS_CONTROL 1\n"
        + (ROOT / "kernels/rocm/ws_hgemm.cu").read_text()
    )
    return _load(
        "ws_control",
        tuple(_prepare(t) for t in (bind, source)),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )


def isa(lib, folder):
    from .isa_objects import hip_isa_objects
    import yaml

    report = hip_isa_objects(lib.__file__, folder.resolve())
    metadata = {}
    for obj in report["objects"]:
        notes = subprocess.check_output(
            [
                "/opt/rocm/llvm/bin/llvm-readobj",
                "--notes",
                str(ROOT / obj["hsaco"]),
            ],
            text=True,
        )
        for entry in yaml.safe_load(
            notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
        )["amdhsa.kernels"]:
            metadata[entry[".name"]] = entry
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n",
        (ROOT / report["assembly"]).read_text(),
        flags=re.M,
    )
    kernels = []
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        if "ws_pipeline" not in symbol:
            continue
        e = metadata[symbol]
        kernels.append(
            dict(
                symbol=symbol,
                wmma=body.count("v_wmma_f32_16x16x16_f16"),
                barrier_signals=body.count("s_barrier_signal"),
                barrier_waits=body.count("s_barrier_wait"),
                poll_sleeps=body.count("s_sleep"),
                global_load_b128=body.count("global_load_b128"),
                vgpr_count=e[".vgpr_count"],
                vgpr_spill_count=e[".vgpr_spill_count"],
                scratch_bytes=e[".private_segment_fixed_size"],
                lds_bytes=e[".group_segment_fixed_size"],
            )
        )
    assert len(kernels) == 1
    report["kernels"] = kernels
    report["passed"] = all(
        k["wmma"]
        and k["lds_bytes"] <= 65536
        and k["scratch_bytes"] == 0
        and k["barrier_signals"] == 1
        and k["barrier_waits"] == 1
        and k["poll_sleeps"]
        for k in kernels
    )
    return report


def audit(control, checks):
    for seed in (17, 93):
        torch.manual_seed(seed)
        for m, n, k in (
            (1, 1, 1),
            (65, 129, 97),
            (128, 256, 128),
            (256, 256, 4096),
            (17, 33, 8193),
            (3, 2, 0),
        ):
            a = (torch.randn(m, k, device="cuda") * 0.25).half()
            b = (torch.randn(k, n, device="cuda") * 0.25).half()
            bp = packed(b, "ws_hgemm_naive_cute")
            ref = (a.double() @ b.double()).half()
            for delay in (0, 1, 2):
                c = torch.full(
                    (m, n), float("nan"), device="cuda", dtype=torch.float16
                )
                metrics = torch.zeros(
                    ((m + 63) // 64) * ((n + 127) // 128),
                    8,
                    device="cuda",
                    dtype=torch.int32,
                )
                control.ws_audit(a, bp, c, metrics, delay)
                values = metrics.cpu()
                tiles = (k + 31) // 32
                progressed = bool((values[:, :5] == tiles).all())
                bounded = bool(
                    (
                        (values[:, 5] <= 3) & (values[:, 5] >= min(tiles, 1))
                    ).all()
                )
                independent = (not (delay == 2 and tiles >= 3)) or bool(
                    (values[:, 5] >= 2).all()
                )
                passed = (
                    torch.allclose(c, ref, atol=0.01, rtol=0.01)
                    and progressed
                    and bounded
                    and independent
                    and not bool(values[:, 6].any())
                )
                checks.append(
                    dict(
                        id=f"pipeline/{seed}/{m}_{n}_{k}/delay{delay}",
                        passed=passed,
                        tiles=tiles,
                        counts=values.tolist(),
                        independent_progress=independent,
                    )
                )
                print(
                    checks[-1]["id"],
                    passed,
                    "lead",
                    values[:, 5].tolist(),
                    flush=True,
                )


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    if args.backend == "hip":
        from .loader import load, source_paths
        from .environment import verify

        identity = verify()
        lib = load("ws_hgemm")
        funcs = {n: getattr(lib, n) for n in VARIANTS}
        control = controls(identity)
        builds = dict(
            hip=dict(
                source_sha256={
                    str(p.relative_to(ROOT)): sha(p)
                    for p in source_paths("ws_hgemm")
                },
                binary_sha256=sha(lib.__file__),
            ),
            control=dict(binary_sha256=sha(control.__file__)),
        )
    else:
        from .cuda_pending_check import build

        manifest = json.loads(args.manifest.read_text())
        groups = [
            g
            for g in manifest["groups"]
            if any(f["name"] in VARIANTS for f in g["functions"])
        ]
        assert len(groups) == 1
        args.cuda_current_stream = True
        lib, metadata = build(groups[0], args)
        funcs = {n: getattr(lib, n) for n in VARIANTS}
        identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            uuid=str(torch.cuda.get_device_properties(args.device).uuid),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
        )
        builds = {groups[0]["source"]: metadata}
    r = dict(
        environment=identity,
        backend=args.backend,
        variants=VARIANTS,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        helper_sha256={
            p: sha(ROOT / p)
            for p in (
                "leetcuda_rocm/mma_check.py",
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/loader.py",
                "leetcuda_rocm/cuda_pending_check.py",
            )
        },
        builds=builds,
        original_accumulator="FP16",
        rocm_accumulator="FP32",
        checks=[],
        boundaries=[],
        scheduling_controls=[],
        pipeline_audits=[],
        performance=[],
        passed=False,
    )
    outputs = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, fn in funcs.items():
        for c in torch.load(args.fixtures, weights_only=True):
            a, b = c["a"].cuda(), c["b"].cuda()
            identifier = name + "/" + c["id"]
            outputs[identifier] = numerical(
                fn, a, b, (), r["checks"], identifier, c["reference"]
            )
            r["checks"][-1].update(
                function=name, input_sha256=c["input_sha256"]
            )
            if args.backend == "hip":
                out = torch.empty_like(outputs[identifier], device="cuda")
                control.ws_synchronous_control(a, packed(b, name), out)
                r["scheduling_controls"].append(
                    dict(
                        id=identifier,
                        passed=torch.equal(out.cpu(), outputs[identifier]),
                    )
                )
        print(
            name,
            "shared",
            sum(not c["passed"] for c in r["checks"]),
            flush=True,
        )
    if args.backend == "hip":
        boundaries(lib, r["boundaries"], VARIANTS)
        audit(control, r["pipeline_audits"])
        r["isa"] = isa(lib, args.output.parent / (args.output.stem + "_isa"))
    if args.performance:
        torch.manual_seed(20261105)
        for size in (256, 1024, 2048):
            a, b = [
                (torch.randn(size, size, device="cuda") * 0.25).half()
                for _ in range(2)
            ]
            bp = packed(b, "ws_hgemm_naive_cute")
            reference = (a.double() @ b.double()).half()
            out = torch.empty_like(a)
            baseline = torch.empty_like(a)
            for name, fn in funcs.items():
                fn(a, bp, out)
                prior = out.clone()
                accuracy = torch.allclose(out, reference, atol=0.01, rtol=0.01)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn(a, bp, out)
                with torch.cuda.stream(stream):
                    out.fill_(float("nan"))
                    graph.replay()
                stream.synchronize()
                replay = torch.equal(out, prior)
                pair = graph_pair(
                    lambda: fn(a, bp, out),
                    lambda: torch.mm(a, b, out=baseline),
                    calls=8,
                    sample_ms=20,
                    trials=5,
                )
                row = dict(
                    function=name,
                    shape_mnk=[size] * 3,
                    accuracy_passed=accuracy,
                    graph_replay_verified=replay,
                    **pair,
                )
                if args.backend == "hip":
                    sync = control.ws_synchronous_control
                    sync(a, bp, baseline)
                    row["synchronous_bitwise_equal"] = torch.equal(
                        out, baseline
                    )
                    row["synchronous_pair"] = graph_pair(
                        lambda: fn(a, bp, out),
                        lambda: sync(a, bp, baseline),
                        calls=8,
                        sample_ms=20,
                        trials=5,
                    )
                r["performance"].append(row)
            print("performance", size, flush=True)
    r["passed"] = all(
        c["passed"]
        for key in (
            "checks",
            "boundaries",
            "scheduling_controls",
            "pipeline_audits",
        )
        for c in r[key]
    )
    if args.backend == "hip":
        r["passed"] &= r["isa"]["passed"] and all(
            c["accuracy_passed"]
            and c["graph_replay_verified"]
            and c["synchronous_bitwise_equal"]
            for c in r["performance"]
        )
    torch.save(outputs, args.output.with_suffix(".pt"))
    r["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print(
        "DONE",
        r["passed"],
        {
            k: len(r[k])
            for k in (
                "checks",
                "boundaries",
                "scheduling_controls",
                "pipeline_audits",
                "performance",
            )
        },
        flush=True,
    )
    return int(not r["passed"])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=["fixtures", "run"])
    p.add_argument("--backend", choices=["hip", "cuda"], default="hip")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--performance", action="store_true")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "docs/rocm/cross_5090_20261105/pending56_manifest.json",
    )
    p.add_argument("--source-root", type=Path, default=ROOT / "original")
    p.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    args = p.parse_args()
    if args.command == "fixtures":
        fixtures(args.fixtures)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
