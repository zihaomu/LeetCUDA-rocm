"""TF32x3 qualification, original RNA-TF32 oracle and input-precision controls."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch
from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha

STATIC = "sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages"
DYNAMIC = STATIC + "_dsmem"
BLAS = "sgemm_cublas_tf32"
VARIANTS = {
    STATIC: "static LDS / BF16 WMMA",
    DYNAMIC: "dynamic LDS / BF16 WMMA",
    BLAS: "three quantized hipBLAS products",
}
ATOL = RTOL = 2e-4
CUDA_ATOL = CUDA_RTOL = (
    0.005  # Unchanged from the original pending-interface audit.
)


def quantize(x):
    """Independent CPU bit reference: nearest, ties away from zero."""
    assert x.device.type == "cpu" and x.dtype == torch.float32
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (u + 4096) & 0xFFFFE000
    rounded = torch.where((u & 0x7FFFFFFF) >= 0x7F800000, u, rounded)
    return rounded.to(torch.int32).view(torch.float32)


def configs(name):
    return (
        [()]
        if name == BLAS
        else [
            (s, w, 256)
            for s in range(2, 6 if name == DYNAMIC else 5)
            for w in (False, True)
        ]
    )


def fixtures(path):
    rows = []
    for seed in (17, 20261106):
        g = torch.Generator().manual_seed(seed)
        for m, n, k in (
            (256, 256, 128),
            (256, 512, 256),
            (384, 256, 136),
            (256, 1024, 160),
            (256, 256, 4096),
        ):
            a = torch.randn(m, k, generator=g) * 0.25
            b = torch.randn(k, n, generator=g) * 0.25
            rows.append(dict(id=f"{seed}_{m}_{n}_{k}", a=a, b=b))
    for pattern in (
        "zero",
        "cancellation",
        "reciprocal-range",
        "rounding-ties",
        "uniform",
    ):
        a = torch.randn(256, 128, generator=g) * 0.25
        b = torch.randn(128, 256, generator=g) * 0.25
        if pattern == "zero":
            a.zero_()
        elif pattern == "cancellation":
            a[:, 1::2] = a[:, ::2]
            b[1::2] = -b[::2]
        elif pattern == "reciprocal-range":
            a *= 2.0**80
            b *= 2.0**-80
        elif pattern == "rounding-ties":
            a.fill_(1 + 2**-11)
            b.fill_(2**-7)
        else:
            a.fill_(0.125)
            b.fill_(0.25)
        rows.append(dict(id=pattern, a=a, b=b))
    for c in rows:
        c["reference"] = c["a"].double() @ c["b"].double()
        c["tf32_reference"] = (
            quantize(c["a"]).double() @ quantize(c["b"]).double()
        )
        c["input_sha256"] = [tensor_sha(c["a"]), tensor_sha(c["b"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)


def controls(identity):
    from .loader import _load, _prepare
    from .environment import cache_key

    names = ["tf32_strict_static", "tf32_strict_dynamic"]
    source = (
        "#define LEETCUDA_TF32_CONTROL 1\n"
        + (ROOT / "kernels/rocm/sgemm_tf32.cu").read_text()
    )
    binding = "#include <torch/extension.h>\n" + "\n".join(
        "void "
        + n
        + "(torch::Tensor,torch::Tensor,torch::Tensor,int,bool,int);"
        for n in names
    )
    binding += (
        "\nvoid tf32_round_probe(torch::Tensor,torch::Tensor);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){"
        + "".join(
            'm.def("' + n + '",&' + n + ");"
            for n in names + ["tf32_round_probe"]
        )
        + "}\n"
    )
    return _load(
        "tf32_control",
        tuple(_prepare(s) for s in (binding, source)),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )


def numerical(
    fn, a, b, config, checks, identifier, reference=None, backend="hip"
):
    ref = (
        (a.cpu().double() @ b.cpu().double()).float()
        if reference is None
        else reference.float()
    )
    storage = torch.full(
        (a.shape[0] * b.shape[1] + 16,),
        731.25,
        dtype=torch.float32,
        device=a.device,
    )
    out = storage[8:-8].view(a.shape[0], b.shape[1])
    out.fill_(float("nan"))
    before = [a.clone(), b.clone()]
    fn(a, b, out, *config)
    actual = out.cpu()
    canary = bool(
        (storage[:8] == 731.25).all() and (storage[-8:] == 731.25).all()
    )
    atol, rtol = (ATOL, RTOL) if backend == "hip" else (CUDA_ATOL, CUDA_RTOL)
    error = (
        float((actual.double() - ref.double()).abs().max())
        if actual.numel()
        else 0.0
    )
    unchanged = torch.equal(a, before[0]) and torch.equal(b, before[1])
    passed = (
        bool(torch.isfinite(actual).all())
        and torch.allclose(actual, ref, atol=atol, rtol=rtol)
        and canary
    )
    if backend == "hip":
        passed &= unchanged
    checks.append(
        dict(
            id=identifier,
            shape_mnk=[a.shape[0], b.shape[1], a.shape[1]],
            config=config,
            passed=passed,
            finite=bool(torch.isfinite(actual).all()),
            canary_passed=canary,
            inputs_unchanged=unchanged,
            atol=atol,
            rtol=rtol,
            max_abs_error=error,
        )
    )
    return actual


def boundaries(lib, rows):
    for name in VARIANTS:
        fn = getattr(lib, name)
        for cfg in configs(name):
            for seed in (19, 71):
                torch.manual_seed(seed)
                for m, n, k in (
                    (0, 3, 8),
                    (3, 0, 8),
                    (3, 5, 0),
                    (1, 1, 1),
                    (17, 33, 7),
                    (65, 129, 9),
                    (129, 257, 17),
                    (257, 263, 65),
                    (17, 19, 8193),
                ):
                    a = torch.randn(m, k, device="cuda") * 0.25
                    b = torch.randn(k, n, device="cuda") * 0.25
                    numerical(
                        fn, a, b, cfg, rows, f"{name}/{cfg}/{seed}/{m}_{n}_{k}"
                    )
            for exponent in (-140, -120, -100, 100, 120):
                a = torch.full((17, 97), 2.0**exponent, device="cuda")
                b = torch.full(
                    (97, 19), 2.0 ** min(127, -exponent), device="cuda"
                )
                numerical(
                    fn, a, b, cfg, rows, f"{name}/{cfg}/exponent-{exponent}"
                )
            a = torch.randn(65, 97, device="cuda") * 0.25
            b = torch.randn(97, 129, device="cuda") * 0.25
            v = torch.empty(a.numel() + 1, device="cuda")[1:].view_as(a)
            v.copy_(a)
            numerical(fn, v, b, cfg, rows, f"{name}/{cfg}/unaligned-A")
            v = torch.empty(b.numel() + 1, device="cuda")[1:].view_as(b)
            v.copy_(b)
            numerical(fn, a, v, cfg, rows, f"{name}/{cfg}/unaligned-B")
        cfg = configs(name)[0]
        a = torch.ones(32, 32, device="cuda")
        b = a.clone()
        out = torch.empty_like(a)
        invalid = {
            "cpu": lambda: fn(a.cpu(), b, out, *cfg),
            "dtype": lambda: fn(a.half(), b, out, *cfg),
            "output-type": lambda: fn(a, b, out.half(), *cfg),
            "rank": lambda: fn(a.flatten(), b, out, *cfg),
            "shape": lambda: fn(a, b, out[:31], *cfg),
            "stride": lambda: fn(a.T, b, out, *cfg),
            "alias-a": lambda: fn(a, b, a, *cfg),
            "alias-b": lambda: fn(a, b, b, *cfg),
            "grad": lambda: fn(a.clone().requires_grad_(), b, out, *cfg),
        }
        if cfg:
            invalid.update(
                {
                    "swizzle-zero": lambda: fn(a, b, out, 2, True, 0),
                    "swizzle-128": lambda: fn(a, b, out, 2, True, 128),
                }
            )
        for label, call in invalid.items():
            try:
                call()
            except RuntimeError:
                passed = True
            else:
                passed = False
            rows.append(dict(id=name + "/reject-" + label, passed=passed))
        if cfg:
            fn(a, b, out, 2, False, 256)
            base = out.clone()
            for stage in (-3, 0, 1, 6, 99, *([5] if name == STATIC else [])):
                fn(a, b, out, stage, False, 256)
                rows.append(
                    dict(
                        id=f"{name}/stage-alias-{stage}",
                        passed=torch.equal(base, out),
                    )
                )
        if torch.cuda.device_count() > 1:
            selected = torch.cuda.current_device()
            other = 1 - selected
            try:
                fn(a, b.to(other), out, *cfg)
            except RuntimeError:
                passed = True
            else:
                passed = False
            rows.append(dict(id=name + "/mixed-device", passed=passed))
            with torch.cuda.device(other):
                fn(a, b, out, *cfg)
                restored = torch.cuda.current_device() == other
            rows.append(
                dict(
                    id=name + "/guard",
                    passed=restored and bool((out == 32).all()),
                )
            )
        queued = []
        for i in range(8):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                aa = torch.full((65, 97), float(i + 1), device="cuda")
                bb = torch.full((97, 129), 0.125, device="cuda")
                cc = torch.empty(65, 129, device="cuda")
                for _ in range(4):
                    fn(aa, bb, cc, *cfg)
                queued.append((stream, aa, bb, cc, i))
        for st, aa, bb, cc, i in queued:
            st.synchronize()
            rows.append(
                dict(
                    id=f"{name}/stream-{i}",
                    passed=bool((cc == (i + 1) * 97 * 0.125).all()),
                )
            )
        print(
            "boundaries", name, sum(not x["passed"] for x in rows), flush=True
        )


def isa(lib, folder):
    from .isa_objects import hip_isa_objects
    import yaml

    result = hip_isa_objects(lib.__file__, folder.resolve())
    metadata = {}
    for obj in result["objects"]:
        text = subprocess.check_output(
            [
                "/opt/rocm/llvm/bin/llvm-readobj",
                "--notes",
                str(ROOT / obj["hsaco"]),
            ],
            text=True,
        )
        for entry in yaml.safe_load(
            text.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
        )["amdhsa.kernels"]:
            metadata[entry[".name"]] = entry
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n",
        (ROOT / result["assembly"]).read_text(),
        flags=re.M,
    )
    rows = []
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        match = re.search(r"tf32_gemmILb([01])ELi([2345])ELb1E", symbol)
        if not match:
            continue
        dyn, stage = map(int, match.groups())
        entry = metadata[symbol]
        lines = body.splitlines()
        overlap = 0
        for j, line in enumerate(lines):
            if "global_load_b128" not in line:
                continue
            for after in lines[j + 1 :]:
                if "s_wait_loadcnt" in after:
                    break
                if "v_wmma_" in after:
                    overlap += 1
                    break
        rows.append(
            dict(
                symbol=symbol,
                dynamic=bool(dyn),
                stages=stage,
                bf16_wmma=body.count("v_wmma_f32_16x16x16_bf16"),
                global_load_b128=body.count("global_load_b128"),
                prefetch_order_sites=overlap,
                vgpr=entry[".vgpr_count"],
                spill=entry[".vgpr_spill_count"],
                scratch_bytes=entry[".private_segment_fixed_size"],
                static_lds=entry[".group_segment_fixed_size"],
                dynamic_lds=stage * 8192 if dyn else 0,
            )
        )
    assert len(rows) == 7
    result.update(
        kernels=rows,
        passed=all(
            x["bf16_wmma"] and x["static_lds"] + x["dynamic_lds"] <= 65536
            for x in rows
        ),
    )
    return result


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    if args.backend == "hip":
        from .loader import load, source_paths
        from .environment import verify

        identity = verify()
        lib = load("sgemm")
        funcs = {n: getattr(lib, n) for n in VARIANTS}
        control = controls(identity)
        builds = dict(
            hip=dict(
                source_sha256={
                    str(p.relative_to(ROOT)): sha(p)
                    for p in source_paths("sgemm")
                },
                binary_sha256=sha(lib.__file__),
            ),
            control=dict(binary_sha256=sha(control.__file__)),
        )
    else:
        from .cuda_pending_check import build

        args.cuda_current_stream = True
        funcs = {}
        builds = {}
        for group in json.loads(args.manifest.read_text())["groups"]:
            if not any(x["name"] in VARIANTS for x in group["functions"]):
                continue
            lib, record = build(group, args)
            builds[group["source"]] = record
            funcs.update(
                {x["name"]: getattr(lib, x["name"]) for x in group["functions"]}
            )
        identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
            uuid=str(torch.cuda.get_device_properties(args.device).uuid),
        )
    report = dict(
        backend=args.backend,
        environment=identity,
        builds=builds,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        variants=VARIANTS,
        input_precision_changed=True,
        original_model="Single RNA-TF32 product (cuBLAS may choose its implementation)",
        rocm_model="Explicit TF32x3 residual correction; BF16 WMMA or three quantized hipBLAS passes; exceptional exponents use full values",
        input_mutation="AMD leaves inputs unchanged; original WMMA quantizes them in place",
        checks=[],
        boundaries=[],
        strict_controls=[],
        rounding=[],
        performance=[],
        passed=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    cases = torch.load(args.fixtures, weights_only=True)
    for name, fn in funcs.items():
        for cfg in configs(name):
            for c in cases:
                assert c["input_sha256"] == [
                    tensor_sha(c["a"]),
                    tensor_sha(c["b"]),
                ]
                a, b = c["a"].cuda(), c["b"].cuda()
                identifier = f'{name}/{cfg}/{c["id"]}'
                outputs[identifier] = numerical(
                    fn,
                    a,
                    b,
                    cfg,
                    report["checks"],
                    identifier,
                    c["reference"],
                    args.backend,
                )
                report["checks"][-1].update(
                    function=name, input_sha256=c["input_sha256"]
                )
                if args.backend == "cuda" and name != BLAS:
                    rounded = all(
                        torch.equal(
                            t.cpu().view(torch.int32),
                            quantize(c[key]).view(torch.int32),
                        )
                        for t, key in ((a, "a"), (b, "b"))
                    )
                    report["rounding"].append(
                        dict(id=identifier, passed=rounded)
                    )
                if args.backend == "hip" and name != BLAS:
                    strict = getattr(
                        control,
                        (
                            "tf32_strict_dynamic"
                            if name == DYNAMIC
                            else "tf32_strict_static"
                        ),
                    )
                    numerical(
                        strict,
                        a,
                        b,
                        cfg,
                        report["strict_controls"],
                        identifier,
                        c["tf32_reference"],
                    )
        print(
            "shared",
            name,
            sum(not x["passed"] for x in report["checks"]),
            flush=True,
        )
        args.output.with_name(args.output.stem + "_progress.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    if args.backend == "hip":
        boundaries(lib, report["boundaries"])
        # Include both ties, signs, exponent transitions and random finite bits.
        g = torch.Generator().manual_seed(91)
        bits = torch.randint(
            0, 2**32, (262144,), dtype=torch.int64, generator=g
        )
        bits = bits[(bits & 0x7F800000) != 0x7F800000]
        ties = torch.tensor(
            [
                sign + base + tail
                for sign in (0, 0x80000000)
                for base in (0, 0x00800000, 0x3F800000, 0x7F000000)
                for tail in (0, 0xFFF, 0x1000, 0x1001, 0x1FFF, 0x2000)
            ],
            dtype=torch.int64,
        )
        x = torch.cat((bits, ties)).to(torch.int32).view(torch.float32)
        out = torch.empty_like(x, device="cuda")
        control.tf32_round_probe(x.cuda(), out)
        report["rounding"].append(
            dict(
                id="RNA-TF32-bits",
                elements=x.numel(),
                passed=torch.equal(
                    out.cpu().view(torch.int32), quantize(x).view(torch.int32)
                ),
            )
        )
        report["isa"] = isa(
            lib, args.output.parent / (args.output.stem + "_isa")
        )
    if args.performance:
        torch.manual_seed(20261106)
        for size in (256, 1024, 2048):
            # Exact TF32 inputs prevent CUDA's in-place quantization from changing a timed workload.
            a, b = [
                (torch.randn(size, size, device="cuda") * 0.25).half().float()
                for _ in range(2)
            ]
            reference = (a.double() @ b.double()).float()
            out = torch.empty_like(a)
            base = torch.empty_like(a)
            for name, fn in funcs.items():
                for cfg in configs(name):
                    fn(a, b, out, *cfg)
                    prior = out.clone()
                    accuracy = torch.allclose(
                        out, reference, atol=ATOL, rtol=RTOL
                    )
                    if args.backend == "cuda" and name == BLAS:
                        # The immutable original creates/destroys a cuBLAS
                        # handle per call. Capture was attempted and failed;
                        # retain that evidence and measure eager execution.
                        from .cross_validate import timed

                        ours = timed(lambda: fn(a, b, out))
                        baseline = timed(lambda: torch.mm(a, b, out=base))
                        report["performance"].append(
                            dict(
                                function=name,
                                config=cfg,
                                shape_mnk=[size] * 3,
                                accuracy_passed=accuracy,
                                graph_capture_supported=False,
                                graph_replay_verified=None,
                                method="eager events including per-call handle overhead; not comparable to graph-only latency",
                                ours_ms=ours["median_ms"],
                                baseline_ms=baseline["median_ms"],
                                samples_ms={
                                    "ours": ours["samples_ms"],
                                    "baseline": baseline["samples_ms"],
                                },
                                baseline_over_ours=baseline["median_ms"]
                                / ours["median_ms"],
                            )
                        )
                        continue
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        fn(a, b, out, *cfg)
                    with torch.cuda.stream(stream):
                        out.fill_(float("nan"))
                        graph.replay()
                    stream.synchronize()
                    replay = torch.equal(out, prior)
                    pair = graph_pair(
                        lambda: fn(a, b, out, *cfg),
                        lambda: torch.mm(a, b, out=base),
                        calls=8,
                        sample_ms=20,
                        trials=5,
                    )
                    report["performance"].append(
                        dict(
                            function=name,
                            config=cfg,
                            shape_mnk=[size] * 3,
                            accuracy_passed=accuracy,
                            graph_replay_verified=replay,
                            **pair,
                        )
                    )
            print("performance", size, flush=True)
    report["passed"] = all(
        x["passed"]
        for field in ("checks", "boundaries", "strict_controls", "rounding")
        for x in report[field]
    )
    if args.backend == "hip":
        report["passed"] &= report["isa"]["passed"]
    report["passed"] &= all(
        x["accuracy_passed"]
        and (
            x["graph_replay_verified"]
            or (
                args.backend == "cuda"
                and x["function"] == BLAS
                and x.get("graph_capture_supported") is False
            )
        )
        for x in report["performance"]
    )
    torch.save(outputs, args.output.with_suffix(".pt"))
    report["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "DONE",
        report["passed"],
        {
            k: len(report[k])
            for k in (
                "checks",
                "boundaries",
                "strict_controls",
                "rounding",
                "performance",
            )
        },
        flush=True,
    )
    return int(not report["passed"])


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
