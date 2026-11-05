"""B5 SIMT FP16 I/O GEMM: FP32 state, true prefetch, synchronous control."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha
from .mma_check import numerical, boundaries, fixtures as base_fixtures

VARIANTS = {
    f"hgemm_t_{tm}x8_sliced_k{bk}_f16x8_pack_dbuf_async": dict(
        bk=bk, tm=tm, bm=128, bn=128, tn=8
    )
    for bk, tm in ((16, 8), (32, 8), (32, 16))
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

    names = [n + "_synchronous_control" for n in VARIANTS]
    bind = (
        "#include <torch/extension.h>\n"
        + "\n".join(
            "void " + n + "(torch::Tensor,torch::Tensor,torch::Tensor);"
            for n in names
        )
        + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){"
        + "\n".join('m.def("' + n + '",&' + n + ");" for n in names)
        + "\n}\n"
    )
    source = (
        "#define LEETCUDA_ASYNC_HGEMM_CONTROL 1\n"
        + (ROOT / "kernels/rocm/hgemm_async.cu").read_text()
    )
    from .environment import cache_key

    return _load(
        "async_hgemm_control",
        tuple(_prepare(s) for s in (bind, source)),
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
    fma = r"v_(?:dual_)?fma(?:c|_mix)?_f32"
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        match = re.search(r"async_hgemmILi(\d+)ELi(\d+)ELb1E", symbol)
        if not match:
            continue
        bk, tm = map(int, match.groups())
        entry = metadata[symbol]
        lines = body.splitlines()
        overlap = 0
        for j, line in enumerate(lines):
            if "global_load_b128" not in line:
                continue
            for after in lines[j + 1 :]:
                if "s_wait_loadcnt" in after:
                    break
                if re.search(fma, after):
                    overlap += 1
                    break
        kernels.append(
            dict(
                symbol=symbol,
                tile_mnk=[128, 128, bk],
                thread_microtile=[tm, 8],
                float32_fma=len(re.findall(fma, body)),
                prefetch_order_sites=overlap,
                global_load_b128=body.count("global_load_b128"),
                vgpr_count=entry[".vgpr_count"],
                vgpr_spill_count=entry[".vgpr_spill_count"],
                scratch_bytes=entry[".private_segment_fixed_size"],
                lds_bytes=entry[".group_segment_fixed_size"],
            )
        )
    assert len(kernels) == 3
    report["kernels"] = kernels
    report["passed"] = all(
        k["float32_fma"]
        and k["prefetch_order_sites"]
        and k["lds_bytes"] <= 65536
        for k in kernels
    )
    report["ordering_note"] = (
        "Static load -> FP32 FMA -> wait ordering verifies scheduled prefetch; no claim of cycle-level overlap measurement."
    )
    return report


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    if args.backend == "hip":
        from .loader import load, source_paths
        from .environment import verify

        identity = verify()
        lib = load("hgemm")
        funcs = {n: getattr(lib, n) for n in VARIANTS}
        control = controls(identity)
        builds = dict(
            hip=dict(
                source_sha256={
                    str(p.relative_to(ROOT)): sha(p)
                    for p in source_paths("hgemm")
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
                getattr(control, name + "_synchronous_control")(a, b, out)
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
        r["isa"] = isa(lib, args.output.parent / (args.output.stem + "_isa"))
    if args.performance:
        torch.manual_seed(20261105)
        for size in (256, 1024, 2048):
            a, b = [
                (torch.randn(size, size, device="cuda") * 0.25).half()
                for _ in range(2)
            ]
            reference = (a.double() @ b.double()).half()
            out = torch.empty_like(a)
            baseline = torch.empty_like(a)
            for name, fn in funcs.items():
                fn(a, b, out)
                prior = out.clone()
                accuracy = torch.allclose(out, reference, atol=0.01, rtol=0.01)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn(a, b, out)
                with torch.cuda.stream(stream):
                    out.fill_(float("nan"))
                    graph.replay()
                stream.synchronize()
                replay = torch.equal(out, prior)
                pair = graph_pair(
                    lambda: fn(a, b, out),
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
                    sync = getattr(control, name + "_synchronous_control")
                    sync(a, b, baseline)
                    row["synchronous_bitwise_equal"] = torch.equal(
                        out, baseline
                    )
                    row["synchronous_pair"] = graph_pair(
                        lambda: fn(a, b, out),
                        lambda: sync(a, b, baseline),
                        calls=8,
                        sample_ms=20,
                        trials=5,
                    )
                r["performance"].append(row)
            print("performance", size, flush=True)
    r["passed"] = all(
        c["passed"]
        for key in ("checks", "boundaries", "scheduling_controls")
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
