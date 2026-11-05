"""U2 BF16 reducers: packing, compensated FP32, original BF16 controls."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch
from .benchmark import graph_pair
from .cross_validate import ROOT, REDUCE, reduce_model, sha, tensor_sha

PACKS = dict(zip(REDUCE, (1, 2, 8)))


def model(x, pack):
    width = x.shape[1]
    if width // pack not in (32, 64, 128, 256, 512, 1024):
        width = 1024
    return reduce_model(x.reshape(-1, width), pack)


def fixtures(path):
    torch.set_num_threads(4)
    cases = []
    for seed in (17, 93):
        g = torch.Generator().manual_seed(seed)
        for rows, k in ((8, 256), (16, 1024), (128, 1024), (8, 8192)):
            cases.append(
                dict(
                    id=f"seed{seed}_{rows}_{k}",
                    x=(torch.randn(rows, k, generator=g) * 0.25).bfloat16(),
                )
            )
    for pattern in ("zero", "cancellation", "small-residual", "wide-range"):
        x = torch.ones(16, 1024, dtype=torch.bfloat16)
        if pattern == "zero":
            x.zero_()
        elif pattern == "cancellation":
            x[:, 1::2] = -x[:, ::2]
        elif pattern == "small-residual":
            x[:, ::4] = 2**24
            x[:, 2::4] = -(2**24)
        else:
            x[:, ::4] = 2**30
            x[:, 2::4] = -(2**30)
            x[:, 1::2] = 2**-10
        cases.append(dict(id=pattern, x=x))
    for c in cases:
        c.update(
            reference=c["x"].double().sum().float().reshape(1),
            models={n: model(c["x"], p) for n, p in PACKS.items()},
            input_sha256=tensor_sha(c["x"]),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, path)


def numerical(fn, x, checks, identifier, reference=None, strict=True):
    before = x.clone()
    y = fn(x)
    ref = (
        x.double().sum().float().reshape(1)
        if reference is None
        else reference.to(x.device)
    )
    atol, rtol = (0.0002, 0.00002) if strict else (0.15, 0.01)
    actual = y.cpu()
    passed = torch.allclose(
        y, ref, atol=atol, rtol=rtol, equal_nan=True
    ) and torch.equal(x.view(torch.int16), before.view(torch.int16))
    checks.append(
        dict(
            id=identifier,
            function=fn.__name__,
            passed=passed,
            shape=list(x.shape),
            atol=atol,
            rtol=rtol,
            output_contract=list(y.shape) == [1]
            and y.dtype == torch.float32
            and y.device == x.device,
            finite=bool(y.isfinite().all()),
            max_error=(
                float((y - ref).abs().max())
                if bool(y.isfinite().all() and ref.isfinite().all())
                else None
            ),
            nonfinite_expected=not bool(ref.isfinite().all()),
        )
    )
    checks[-1]["passed"] &= checks[-1]["output_contract"]
    return actual


def boundaries(lib, checks):
    for name, pack in PACKS.items():
        fn = getattr(lib, name)
        for seed in (17, 93):
            torch.manual_seed(seed)
            for shape in (
                (),
                (0,),
                (1,),
                (3,),
                (31,),
                (33,),
                (257,),
                (1025,),
                (2**20,),
                (7, 31),
                (2, 4097),
                (3, 0),
                (0, 16),
            ):
                x = (torch.randn(shape, device="cuda") * 0.25).bfloat16()
                numerical(fn, x, checks, f"{name}/{seed}/{shape}")
            # Exercise all six source-specialized thread counts and both dispatch forms.
            for threads in (32, 64, 128, 256, 512, 1024):
                x = (
                    torch.randn(3, threads * pack, device="cuda") * 0.25
                ).bfloat16()
                numerical(fn, x, checks, f"{name}/{seed}/threads{threads}")
            x = (torch.randn(4098, device="cuda") * 0.25).bfloat16()[1:]
            numerical(fn, x, checks, f"{name}/{seed}/unaligned")
        for mode in (
            "wide-range",
            "subnormal",
            "positive-inf",
            "negative-inf",
            "mixed-inf",
            "nan",
        ):
            x = torch.ones(2**16, device="cuda", dtype=torch.bfloat16)
            if mode == "wide-range":
                x[::4] = 2**30
                x[2::4] = -(2**30)
                x[1::2] = 2**-10
            elif mode == "subnormal":
                x.fill_(2**-133)
            elif mode == "positive-inf":
                x[2] = float("inf")
            elif mode == "negative-inf":
                x[3] = -float("inf")
            elif mode == "mixed-inf":
                x[2] = float("inf")
                x[3] = -float("inf")
            else:
                x[7] = float("nan")
            actual = numerical(fn, x, checks, name + "/" + mode)
            if mode == "subnormal":
                exact = torch.equal(
                    actual, x.double().sum().float().reshape(1).cpu()
                )
                checks[-1].update(
                    passed=checks[-1]["passed"] and exact, subnormal_exact=exact
                )
        x = torch.ones(8, 1024, device="cuda", dtype=torch.bfloat16)
        bad = {
            "cpu": lambda: fn(x.cpu()),
            "dtype": lambda: fn(x.half()),
            "strides": lambda: fn(x[:, ::2]),
            "grad": lambda: fn(x.clone().requires_grad_()),
        }
        for key, call in bad.items():
            try:
                call()
            except RuntimeError:
                passed = True
            else:
                passed = False
            checks.append(dict(id=name + "/reject-" + key, passed=passed))
        if torch.cuda.device_count() > 1:
            other = 1 - torch.cuda.current_device()
            with torch.cuda.device(other):
                y = fn(x)
                restored = torch.cuda.current_device() == other
            checks.append(
                dict(
                    id=name + "/device-guard",
                    passed=restored
                    and y.device == x.device
                    and float(y) == 8192,
                )
            )
        pending = []
        for index in range(8):
            st = torch.cuda.Stream()
            with torch.cuda.stream(st):
                xx = torch.full(
                    (65, 257), index + 1, device="cuda", dtype=torch.bfloat16
                )
                ys = [fn(xx) for _ in range(4)]
                pending.append((st, xx, ys, index))
        for st, xx, ys, index in pending:
            st.synchronize()
            checks.append(
                dict(
                    id=f"{name}/stream{index}",
                    passed=all(float(y) == 65 * 257 * (index + 1) for y in ys),
                )
            )
        torch.manual_seed(55)
        x = (torch.randn(2**20, device="cuda") * 0.25).bfloat16()
        first = fn(x)
        checks.append(
            dict(
                id=name + "/deterministic-20-repeats",
                passed=all(torch.equal(fn(x), first) for _ in range(20)),
            )
        )
        print(
            "boundaries", name, sum(not c["passed"] for c in checks), flush=True
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
        for e in yaml.safe_load(
            notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
        )["amdhsa.kernels"]:
            metadata[e[".name"]] = e
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n",
        (ROOT / report["assembly"]).read_text(),
        flags=re.M,
    )
    kernels = []
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        m = re.search(r"reduce_blocksILi(\d+)ELi(\d+)E", symbol)
        if not m:
            continue
        pack, threads = map(int, m.groups())
        e = metadata[symbol]
        kernels.append(
            dict(
                symbol=symbol,
                pack=pack,
                threads=threads,
                global_load_b128=body.count("global_load_b128"),
                float_add=body.count("v_add_f32"),
                vgpr=e[".vgpr_count"],
                spill=e[".vgpr_spill_count"],
                scratch=e[".private_segment_fixed_size"],
                lds=e[".group_segment_fixed_size"],
            )
        )
    assert len(kernels) == 18
    report["kernels"] = kernels
    report["passed"] = all(
        k["float_add"]
        and not k["spill"]
        and not k["scratch"]
        and k["lds"] <= 65536
        and (k["pack"] != 8 or k["global_load_b128"])
        for k in kernels
    )
    return report


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    if args.backend == "hip":
        from .environment import verify
        from .loader import load, source_paths
        from .cross_validate import build

        identity = verify()
        lib = load("reduce")
        control, control_meta = build(
            "reduce", "hip", args.source_root, args.cutlass
        )
        builds = dict(
            hip=dict(
                source_sha256={
                    str(p.relative_to(ROOT)): sha(p)
                    for p in source_paths("reduce")
                },
                binary_sha256=sha(lib.__file__),
            ),
            control=control_meta,
        )
    else:
        from .cuda_pending_check import build

        group = dict(
            source="kernels/reduce/block_all_reduce.cu",
            module="reduce",
            functions=[
                dict(
                    name=n,
                    declaration=f"torch::Tensor {n}(torch::Tensor);",
                    argc=1,
                )
                for n in PACKS
            ],
        )
        args.cuda_current_stream = True
        lib, meta = build(group, args)
        identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            uuid=str(torch.cuda.get_device_properties(args.device).uuid),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
        )
        builds = {group["source"]: meta}
    r = dict(
        environment=identity,
        backend=args.backend,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        helper_sha256={
            f: sha(ROOT / f)
            for f in (
                "leetcuda_rocm/cross_validate.py",
                "leetcuda_rocm/loader.py",
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/cuda_pending_check.py",
            )
        },
        builds=builds,
        variants=PACKS,
        original_accumulator="BF16",
        rocm_accumulator="compensated FP32",
        checks=[],
        boundaries=[],
        bf16_controls=[],
        performance=[],
        passed=False,
    )
    outputs = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in PACKS:
        fn = getattr(lib, name)
        for c in torch.load(args.fixtures, weights_only=True):
            x = c["x"].cuda()
            identifier = name + "/" + c["id"]
            outputs[identifier] = numerical(
                fn,
                x,
                r["checks"],
                identifier,
                c["reference"],
                args.backend == "hip",
            )
            r["checks"][-1]["input_sha256"] = c["input_sha256"]
            if args.backend == "hip":
                old = getattr(control, name)(x).cpu()
                r["bf16_controls"].append(
                    dict(
                        id=identifier,
                        passed=torch.equal(old, c["models"][name]),
                        fp64_accuracy_passed=torch.allclose(
                            old, c["reference"], atol=0.15, rtol=0.01
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
        boundaries(lib, r["boundaries"])
        r["isa"] = isa(lib, args.output.parent / (args.output.stem + "_isa"))
    if args.performance:
        for shape in ((8, 1024), (128, 1024), (1024, 1024)):
            torch.manual_seed(55)
            x = (torch.randn(shape, device="cuda") * 0.25).bfloat16()
            ref = x.double().sum().float().reshape(1)
            for name in PACKS:
                fn = getattr(lib, name)
                prior = fn(x)
                tol = (
                    (0.0002, 0.00002) if args.backend == "hip" else (0.15, 0.01)
                )
                accuracy = torch.allclose(prior, ref, atol=tol[0], rtol=tol[1])
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=st):
                    captured = fn(x)
                with torch.cuda.stream(st):
                    captured.fill_(float("nan"))
                    g.replay()
                st.synchronize()
                replay = torch.allclose(
                    captured, prior, atol=tol[0], rtol=tol[1]
                )
                r["performance"].append(
                    dict(
                        function=name,
                        shape=list(shape),
                        accuracy_passed=accuracy,
                        graph_replay_verified=replay,
                        **graph_pair(
                            lambda: fn(x),
                            lambda: torch.sum(x, dtype=torch.float32),
                            calls=8,
                            sample_ms=20,
                            trials=5,
                        ),
                    )
                )
            print("performance", shape, flush=True)
    r["passed"] = all(
        c["passed"]
        for f in ("checks", "boundaries", "bf16_controls")
        for c in r[f]
    )
    if args.backend == "hip":
        r["passed"] &= r["isa"]["passed"] and all(
            p["accuracy_passed"] and p["graph_replay_verified"]
            for p in r["performance"]
        )
    torch.save(outputs, args.output.with_suffix(".pt"))
    r["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print(
        "DONE",
        r["passed"],
        {
            f: len(r[f])
            for f in ("checks", "boundaries", "bf16_controls", "performance")
        },
        flush=True,
    )
    return int(not r["passed"])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=("fixtures", "run"))
    p.add_argument("--backend", choices=("hip", "cuda"), default="hip")
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
