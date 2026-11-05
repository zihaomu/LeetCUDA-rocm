"""U1 scalar/packed HGEMV: explicit FP32 state, original half controls."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch

from .benchmark import graph_pair
from .cross_validate import ROOT, binding, gemv_models, sha, tensor_sha

NAMES = ("hgemv_f16_cute", "hgemv_f16x8_cute")


def fixtures(path):
    torch.set_num_threads(4)
    cases = []
    for seed in (17, 93):
        g = torch.Generator().manual_seed(seed)
        for m, k in (
            (64, 128),
            (65, 256),
            (129, 512),
            (64, 1024),
            (64, 8192),
            (17, 65536),
        ):
            a = (torch.randn(m, k, generator=g) * 0.25).half()
            b = (torch.randn(k, 1, generator=g) * 0.25).half()
            cases.append(dict(id=f"seed{seed}_{m}_{k}", a=a, b=b))
    g = torch.Generator().manual_seed(55)
    for pattern in ("zero", "cancellation", "reciprocal"):
        a = (torch.randn(65, 4096, generator=g) * 0.25).half()
        b = (torch.randn(4096, 1, generator=g) * 0.25).half()
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
        cases.append(dict(id=pattern, a=a, b=b))
    for c in cases:
        c.update(
            reference=c["a"].double() @ c["b"].double(),
            models=gemv_models(c["a"], c["b"]),
            input_sha256=[tensor_sha(c["a"]), tensor_sha(c["b"])],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, path)


def controls(identity):
    from .loader import _load, _prepare
    from .environment import cache_key

    return _load(
        "hgemv_half_control",
        tuple(
            _prepare(s)
            for s in (
                binding(NAMES),
                (ROOT / "kernels/rocm/hgemv_layouts.cu").read_text(),
            )
        ),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )


def numerical(fn, a, b, checks, identifier, reference=None):
    storage = torch.full((a.shape[0] + 64,), 42, device=a.device, dtype=a.dtype)
    c = storage[32:-32].reshape(a.shape[0], 1)
    c.fill_(float("nan"))
    aa, bb = a.clone(), b.clone()
    fn(a, b, c)
    actual = c.cpu()
    ref = (
        (a.double() @ b.double()).cpu()
        if reference is None
        else reference.cpu()
    )
    immutable = torch.equal(a, aa) and torch.equal(b, bb)
    canary = bool((storage[:32] == 42).all() and (storage[-32:] == 42).all())
    passed = (
        torch.allclose(actual, ref.half(), atol=0.003, rtol=0.003)
        and immutable
        and canary
    )
    checks.append(
        dict(
            id=identifier,
            function=fn.__name__,
            passed=passed,
            shape_mk=list(a.shape),
            atol=0.003,
            rtol=0.003,
            finite=bool(actual.isfinite().all()),
            input_immutable=immutable,
            canary=canary,
            max_error_fp64=(
                float((actual.double() - ref).abs().max())
                if actual.numel()
                else 0
            ),
        )
    )
    return actual


def boundaries(lib, checks):
    for name in NAMES:
        fn = getattr(lib, name)
        for seed in (17, 93):
            torch.manual_seed(seed)
            for m, k in (
                (0, 8),
                (3, 0),
                (1, 1),
                (3, 8),
                (17, 19),
                (65, 257),
                (129, 1025),
                (17, 65537),
                (4097, 128),
            ):
                a = (torch.randn(m, k, device="cuda") * 0.25).half()
                b = (torch.randn(k, 1, device="cuda") * 0.25).half()
                numerical(fn, a, b, checks, f"{name}/{seed}/{m}_{k}")
            for target in ("a", "b"):
                a = (torch.randn(65, 256, device="cuda") * 0.25).half()
                b = (torch.randn(256, 1, device="cuda") * 0.25).half()
                t = a if target == "a" else b
                raw = torch.empty(t.numel() + 1, device="cuda", dtype=t.dtype)
                v = raw[1:].view_as(t)
                v.copy_(t)
                if target == "a":
                    a = v
                else:
                    b = v
                numerical(fn, a, b, checks, f"{name}/{seed}/unaligned-{target}")
        a = torch.ones(32, 64, device="cuda", dtype=torch.float16)
        b = torch.ones(64, 1, device="cuda", dtype=torch.float16)
        c = torch.empty(32, 1, device="cuda", dtype=torch.float16)
        bad = {
            "cpu": lambda: fn(a.cpu(), b, c),
            "a-type": lambda: fn(a.float(), b, c),
            "b-type": lambda: fn(a, b.float(), c),
            "c-type": lambda: fn(a, b, c.float()),
            "rank": lambda: fn(a.flatten(), b, c),
            "shape": lambda: fn(a, b[:63], c),
            "strides": lambda: fn(a[:, ::2], b[:32], c),
            "alias-a": lambda: fn(a, b, a.flatten()[:32].view_as(c)),
            "alias-b": lambda: fn(a, b, b[:32]),
            "grad-a": lambda: fn(a.clone().requires_grad_(), b, c),
            "grad-b": lambda: fn(a, b.clone().requires_grad_(), c),
            "grad-c": lambda: fn(a, b, c.clone().requires_grad_()),
        }
        if torch.cuda.device_count() > 1:
            selected = torch.cuda.current_device()
            other = 1 - selected
            bad["mixed"] = lambda: fn(a, b.to(other), c)
            with torch.cuda.device(other):
                fn(a, b, c)
                restored = torch.cuda.current_device() == other
            checks.append(
                dict(
                    id=name + "/device-guard",
                    passed=restored and bool((c == 64).all()),
                )
            )
        for kind, call in bad.items():
            try:
                call()
            except RuntimeError:
                passed = True
            else:
                passed = False
            checks.append(dict(id=name + "/reject-" + kind, passed=passed))
        # Every lane receives a leading 16 and many representable small terms;
        # a half lane accumulator loses them before the XOR tree.
        aa = torch.ones(4, 65536, device="cuda", dtype=torch.float16)
        bb = torch.full((65536, 1), 2**-12, device="cuda", dtype=torch.float16)
        bb[:256] = 16
        cc = torch.empty(4, 1, device="cuda", dtype=torch.float16)
        fn(aa, bb, cc)
        expected = torch.tensor(256 * 16 + (65536 - 256) * 2**-12).half().item()
        checks.append(
            dict(
                id=name + "/FP32-state",
                passed=bool((cc == expected).all()),
                expected=expected,
            )
        )
        pending = []
        for index in range(8):
            st = torch.cuda.Stream()
            with torch.cuda.stream(st):
                aa = torch.full(
                    (65, 1025), index + 1, device="cuda", dtype=torch.float16
                )
                bb = torch.full(
                    (1025, 1), 0.125, device="cuda", dtype=torch.float16
                )
                cc = torch.empty(65, 1, device="cuda", dtype=torch.float16)
                for _ in range(4):
                    fn(aa, bb, cc)
                pending.append((st, aa, bb, cc, index))
        for st, aa, bb, cc, index in pending:
            st.synchronize()
            expected = torch.tensor((index + 1) * 1025 * 0.125).half().item()
            checks.append(
                dict(
                    id=f"{name}/stream{index}",
                    passed=bool((cc == expected).all()),
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
        if "vector_fp32" not in symbol:
            continue
        e = metadata[symbol]
        kernels.append(
            dict(
                symbol=symbol,
                packed="ILb1E" in symbol,
                fma_f32=len(
                    re.findall(r"v_(?:dual_)?fma(?:c|_mix)?_f32", body)
                ),
                global_load_b128=body.count("global_load_b128"),
                vgpr=e[".vgpr_count"],
                spill=e[".vgpr_spill_count"],
                scratch=e[".private_segment_fixed_size"],
                lds=e[".group_segment_fixed_size"],
            )
        )
    assert len(kernels) == 2
    report["kernels"] = kernels
    report["passed"] = all(
        k["fma_f32"]
        and not k["spill"]
        and not k["scratch"]
        and (not k["packed"] or k["global_load_b128"] >= 2)
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

        identity = verify()
        lib = load("hgemv")
        control = controls(identity)
        builds = dict(
            hip=dict(
                source_sha256={
                    str(p.relative_to(ROOT)): sha(p)
                    for p in source_paths("hgemv")
                },
                binary_sha256=sha(lib.__file__),
            ),
            control=dict(
                source_sha256=sha(ROOT / "kernels/rocm/hgemv_layouts.cu"),
                binary_sha256=sha(control.__file__),
            ),
        )
    else:
        from .cuda_pending_check import build

        group = json.loads(args.manifest.read_text())["groups"][0]
        args.cuda_current_stream = True
        lib, metadata = build(group, args)
        identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            uuid=str(torch.cuda.get_device_properties(args.device).uuid),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
        )
        builds = {group["source"]: metadata}
    r = dict(
        environment=identity,
        backend=args.backend,
        source_root=str(args.source_root),
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
        variants=NAMES,
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
    for name in NAMES:
        fn = getattr(lib, name)
        for c in torch.load(args.fixtures, weights_only=True):
            a, b = c["a"].cuda(), c["b"].cuda()
            identifier = name + "/" + c["id"]
            outputs[identifier] = numerical(
                fn, a, b, r["checks"], identifier, c["reference"]
            )
            r["checks"][-1]["input_sha256"] = c["input_sha256"]
            if args.backend == "hip":
                out = torch.empty_like(outputs[identifier], device="cuda")
                getattr(control, name)(a, b, out)
                r["half_controls"].append(
                    dict(
                        id=identifier,
                        passed=torch.equal(out.cpu(), c["models"][name]),
                        fp64_accuracy_passed=torch.allclose(
                            out.cpu(),
                            c["reference"].half(),
                            atol=0.003,
                            rtol=0.003,
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
        for m, k in ((64, 128), (1024, 1024), (4096, 8192)):
            torch.manual_seed(55)
            a = (torch.randn(m, k, device="cuda") * 0.25).half()
            b = (torch.randn(k, 1, device="cuda") * 0.25).half()
            ref = (a.double() @ b.double()).half()
            out = torch.empty_like(ref)
            base = torch.empty_like(ref)
            for name in NAMES:
                fn = getattr(lib, name)
                fn(a, b, out)
                prior = out.clone()
                accuracy = torch.allclose(out, ref, atol=0.003, rtol=0.003)
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=st):
                    fn(a, b, out)
                with torch.cuda.stream(st):
                    out.fill_(float("nan"))
                    g.replay()
                st.synchronize()
                replay = torch.equal(out, prior)
                r["performance"].append(
                    dict(
                        function=name,
                        shape_mk=[m, k],
                        accuracy_passed=accuracy,
                        graph_replay_verified=replay,
                        **graph_pair(
                            lambda: fn(a, b, out),
                            lambda: torch.mm(a, b, out=base),
                            calls=8,
                            sample_ms=20,
                            trials=5,
                        ),
                    )
                )
            print("performance", m, k, flush=True)
    r["passed"] = all(
        c["passed"]
        for f in ("checks", "boundaries", "half_controls")
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
            for f in ("checks", "boundaries", "half_controls", "performance")
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
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts/rocm/continuation_all/u1/manifest.json",
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
