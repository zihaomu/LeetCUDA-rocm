"""U3: eleven distinct SIMT layouts with explicit FP32 accumulator upgrades."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch

from .benchmark import graph_pair
from .cross_validate import ROOT, HGEMM, sha, tensor_sha, half_round
from .mma_check import numerical, boundaries

VARIANTS = {
    name: dict(
        bm=32 if i < 2 else 128,
        bn=32 if i < 2 else 128,
        bk=1 if i == 0 else 32 if i == 1 or i >= 9 else 16 if i == 8 else 8,
        tm=1 if i < 2 else 16 if i == 10 else 8,
        tn=1 if i < 2 else 8,
        alignment=(
            2 if i < 2 else 4 if i in (2, 4) else 8 if i in (3, 5) else 16
        ),
        double_buffer=i >= 7,
    )
    for i, name in enumerate(HGEMM)
}


def fixtures(path):
    torch.set_num_threads(4)
    g = torch.Generator()
    rows = []
    for seed in (17, 93):
        g.manual_seed(seed)
        for m, n, k in (
            (128, 128, 32),
            (128, 256, 96),
            (256, 128, 256),
            (128, 128, 4096),
        ):
            a = (torch.randn(m, k, generator=g) * 0.25).half()
            b = (torch.randn(k, n, generator=g) * 0.25).half()
            rows.append(dict(id=f"s{seed}_{m}_{n}_{k}", a=a, b=b))
    for kind in ("zero", "cancellation", "reciprocal-range"):
        a = (torch.randn(128, 256, generator=g) * 0.25).half()
        b = (torch.randn(256, 128, generator=g) * 0.25).half()
        if kind == "zero":
            a.zero_()
        elif kind == "cancellation":
            a[:, 1::2] = a[:, ::2]
            b[1::2] = -b[::2]
        else:
            a[:, ::2] *= 256
            b[::2] /= 256
            a[:, 1::2] /= 256
            b[1::2] *= 256
        rows.append(dict(id=kind, a=a, b=b))
    for c in rows:
        a, b = c["a"], c["b"]
        c["reference"] = a.double() @ b.double()
        acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.half)
        for k in range(a.shape[1]):
            acc = half_round(
                acc.double() + a[:, k, None].double() * b[k, None, :].double()
            )
        c["half_model"] = acc
        c["input_sha256"] = [tensor_sha(a), tensor_sha(b)]
        print("fixture", c["id"], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)


def isa(lib, folder):
    from .isa_objects import hip_isa_objects
    import yaml

    r = hip_isa_objects(lib.__file__, folder.resolve())
    meta = {}
    for obj in r["objects"]:
        notes = subprocess.check_output(
            [
                "/opt/rocm/llvm/bin/llvm-readobj",
                "--notes",
                str(ROOT / obj["hsaco"]),
            ],
            text=True,
        )
        meta.update(
            {
                k[".name"]: k
                for k in yaml.safe_load(
                    notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
                )["amdhsa.kernels"]
            }
        )
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n",
        (ROOT / r["assembly"]).read_text(),
        flags=re.M,
    )
    rows = []
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        names = [n for n in VARIANTS if n + "_fp32_kernel" in symbol]
        if not names:
            continue
        assert len(names) == 1
        name = names[0]
        e = meta[symbol]
        g = VARIANTS[name]
        expected_lds = (
            0
            if name == HGEMM[0]
            else (g["bm"] + g["bn"])
            * g["bk"]
            * 2
            * (2 if g["double_buffer"] else 1)
        )
        rows.append(
            dict(
                function=name,
                symbol=symbol,
                fp32_fma=len(
                    re.findall(r"v_(?:dual_)?fma(?:c|_mix)?_f32", body)
                ),
                global_load_b128=body.count("global_load_b128"),
                global_load_b64=body.count("global_load_b64"),
                lds_load_b128=body.count("ds_load_b128"),
                lds_load_b64=body.count("ds_load_b64"),
                global_store_b128=body.count("global_store_b128"),
                global_store_b64=body.count("global_store_b64"),
                lds_bytes=e[".group_segment_fixed_size"],
                expected_lds_bytes=expected_lds,
                vgpr_count=e[".vgpr_count"],
                spill=e[".vgpr_spill_count"],
                scratch=e[".private_segment_fixed_size"],
                barrier_signal=body.count("s_barrier_signal"),
                barrier_wait=body.count("s_barrier_wait"),
            )
        )
    assert {x["function"] for x in rows} == set(VARIANTS) and len(rows) == 11
    r["kernels"] = rows
    r["passed"] = all(
        x["fp32_fma"]
        and x["lds_bytes"] == x["expected_lds_bytes"]
        and x["lds_bytes"] <= 65536
        for x in rows
    )
    r["resource_target_passed"] = all(
        x["spill"] == x["scratch"] == 0 for x in rows
    )
    return r


def extra_boundaries(lib, checks):
    for name, g in VARIANTS.items():
        fn = getattr(lib, name)
        # Full tiles test the actual fast layout, including odd LDS-slot parity.
        for count in (1, 3, 5):
            k = g["bk"] * count
            a = torch.randn(128, k, device="cuda", dtype=torch.half) * 0.25
            b = torch.randn(k, 256, device="cuda", dtype=torch.half) * 0.25
            before = (a.clone(), b.clone())
            numerical(fn, a, b, (), checks, f"{name}/fast-tiles-{count}")
            checks.append(
                dict(
                    id=f"{name}/immutable-{count}",
                    passed=torch.equal(a, before[0])
                    and torch.equal(b, before[1]),
                )
            )
        a = torch.randn(128, 96, device="cuda", dtype=torch.half) * 0.25
        b = torch.randn(96, 256, device="cuda", dtype=torch.half) * 0.25
        ref = (a.double() @ b.double()).half()
        for target in ("a", "b", "c"):
            x = a if target == "a" else b if target == "b" else ref
            storage = torch.full(
                (x.numel() + 2,), 42, device="cuda", dtype=torch.half
            )
            view = storage[1:-1].view_as(x)
            view.copy_(x)
            aa = view if target == "a" else a
            bb = view if target == "b" else b
            cc = view if target == "c" else torch.empty_like(ref)
            if target == "c":
                cc.fill_(float("nan"))
            fn(aa, bb, cc)
            checks.append(
                dict(
                    id=f"{name}/unaligned-full-{target}",
                    passed=torch.allclose(cc, ref, atol=0.01, rtol=0.01)
                    and bool(storage[0] == 42 and storage[-1] == 42),
                )
            )
        streams = []
        for i in range(8):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                aa = torch.full(
                    (128, 96), i + 1.0, device="cuda", dtype=torch.half
                )
                bb = torch.full(
                    (96, 256), 0.125, device="cuda", dtype=torch.half
                )
                cc = torch.empty((128, 256), device="cuda", dtype=torch.half)
                for _ in range(4):
                    fn(aa, bb, cc)
                streams.append((stream, aa, bb, cc, i))
        for stream, aa, bb, cc, i in streams:
            stream.synchronize()
            checks.append(
                dict(
                    id=f"{name}/fast-stream-{i}",
                    passed=bool((cc == (i + 1) * 12).all()),
                )
            )


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    if args.backend == "hip":
        from .environment import verify
        from .loader import load, source_paths
        from .cross_validate import build

        identity = verify()
        lib = load("hgemm")
        funcs = {n: getattr(lib, n) for n in VARIANTS}
        control, control_meta = build("hgemm", "hip", None, None)
        builds = dict(
            hip=dict(
                source_sha256={
                    str(p.relative_to(ROOT)): sha(p)
                    for p in source_paths("hgemm")
                },
                binary_sha256=sha(lib.__file__),
            ),
            control=control_meta,
        )
    else:
        from .cuda_pending_check import build

        identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            uuid=str(torch.cuda.get_device_properties(args.device).uuid),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
        )
        args.cuda_current_stream = True
        funcs = {}
        builds = {}
        for source, names in [
            ("hgemm.cu", HGEMM[:8]),
            ("hgemm_async.cu", HGEMM[8:]),
        ]:
            group = dict(
                source="kernels/hgemm/naive/" + source,
                module="hgemm",
                functions=[
                    dict(
                        name=n,
                        declaration=f"void {n}(torch::Tensor,torch::Tensor,torch::Tensor);",
                        argc=3,
                    )
                    for n in names
                ],
            )
            lib, metadata = build(group, args)
            builds[source] = metadata
            funcs.update({n: getattr(lib, n) for n in names})
    r = dict(
        environment=identity,
        backend=args.backend,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        helper_sha256={
            p: sha(ROOT / p)
            for p in (
                "leetcuda_rocm/mma_check.py",
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/cross_validate.py",
                "leetcuda_rocm/cuda_pending_check.py",
                "leetcuda_rocm/loader.py",
            )
        },
        builds=builds,
        variants=VARIANTS,
        original_accumulator="FP16",
        rocm_accumulator="FP32",
        checks=[],
        boundaries=[],
        half_controls=[],
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
                function=name,
                input_sha256=c["input_sha256"],
                inputs_unchanged=torch.equal(a.cpu(), c["a"])
                and torch.equal(b.cpu(), c["b"]),
            )
            r["checks"][-1]["passed"] &= r["checks"][-1]["inputs_unchanged"]
            if args.backend == "hip":
                old = torch.empty_like(outputs[identifier], device="cuda")
                getattr(control, name)(a, b, old)
                old = old.cpu()
                r["half_controls"].append(
                    dict(
                        id=identifier,
                        passed=torch.equal(old, c["half_model"]),
                        fp64_accuracy_passed=torch.allclose(
                            old, c["reference"].half(), atol=0.01, rtol=0.01
                        ),
                    )
                )
        print(
            "shared",
            name,
            sum(not c["passed"] for c in r["checks"]),
            flush=True,
        )
    if args.backend == "hip":
        boundaries(lib, r["boundaries"], VARIANTS)
        extra_boundaries(lib, r["boundaries"])
        r["isa"] = isa(lib, args.output.parent / (args.output.stem + "_isa"))
    if args.performance:
        torch.manual_seed(20261105)
        for size in (256, 1024, 2048):
            a, b = [
                (torch.randn(size, size, device="cuda") * 0.25).half()
                for _ in range(2)
            ]
            ref = (a.double() @ b.double()).half()
            out = torch.empty_like(a)
            baseline = torch.empty_like(a)
            for name, fn in funcs.items():
                fn(a, b, out)
                prior = out.clone()
                accuracy = torch.allclose(out, ref, atol=0.01, rtol=0.01)
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
                r["performance"].append(
                    dict(
                        function=name,
                        shape_mnk=[size] * 3,
                        accuracy_passed=accuracy,
                        graph_replay_verified=replay,
                        **pair,
                    )
                )
            print("performance", size, flush=True)
    r["passed"] = all(
        c["passed"]
        for field in ("checks", "boundaries", "half_controls")
        for c in r[field]
    )
    if args.backend == "hip":
        r["passed"] &= (
            r["isa"]["passed"]
            and r["isa"]["resource_target_passed"]
            and all(
                p["accuracy_passed"] and p["graph_replay_verified"]
                for p in r["performance"]
            )
        )
    torch.save(outputs, args.output.with_suffix(".pt"))
    r["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print(
        "DONE",
        r["passed"],
        {
            k: len(r[k])
            for k in ("checks", "boundaries", "half_controls", "performance")
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
    p.add_argument("--source-root", type=Path, default=ROOT / "original")
    p.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    args = p.parse_args()
    if args.command == "fixtures":
        fixtures(args.fixtures)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
