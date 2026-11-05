"""MMA HGEMM qualification: explicit FP32 upgrade and independent CUDA evidence."""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha, timed

PREFIX = "hgemm_mma_"
VARIANTS = {
    PREFIX + "m16n8k16_naive": -1,
    PREFIX + "m16n8k16_mma2x4_warp4x4": 0,
    PREFIX + "m16n8k16_mma2x4_warp4x4_stages": 1,
    PREFIX + "m16n8k16_mma2x4_warp4x4_stages_dsmem": 2,
    PREFIX + "m16n8k16_mma2x4_warp4x4x2_stages_dsmem": 3,
    PREFIX + "m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4": 4,
    PREFIX + "m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr": 5,
    PREFIX + "m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle": 6,
    PREFIX + "m16n8k16_mma2x4_warp4x4_stages_dsmem_tn": 7,
    PREFIX + "m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4": 8,
    PREFIX + "stages_block_swizzle_tn_cute": 9,
}
ATOL = RTOL = 0.01


def cuda_resource_error(name, config):
    if config and config[0] == 5 and VARIANTS[name] == 8:
        return "upstream TN x4 stage5 requests 102400 bytes, above its 98304-byte opt-in"
    if config and config[0] == 5 and VARIANTS[name] in (3, 4, 5):
        # Actual upstream padding is A_PAD=B_PAD=8. The stage5 allocation is
        # 104960 bytes, but the wrapper opts in to only 98304. Its launch error
        # was reproduced before adding this preflight; retain it as a failure.
        return (
            "upstream stage5 requests 104960 bytes, above its 98304-byte opt-in"
        )
    return None


def inspect_specializations(isa):
    import yaml

    meta = {}
    for obj in isa["objects"]:
        notes = subprocess.check_output(
            [
                "/opt/rocm/llvm/bin/llvm-readobj",
                "--notes",
                str(ROOT / obj["hsaco"]),
            ],
            text=True,
        )
        block = notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
        meta.update(
            {k[".name"]: k for k in yaml.safe_load(block)["amdhsa.kernels"]}
        )
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n",
        (ROOT / isa["assembly"]).read_text(),
        flags=re.M,
    )
    kernels = []
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        match = re.search(r"mma_tiledILi(\d+)ELi(\d+)EEE", symbol)
        if not match and "mma_naive" not in symbol:
            continue
        mode, stages = map(int, match.groups()) if match else (-1, 1)
        bk = 32 if mode >= 3 and mode != 7 else 16
        bm = 16 if mode == -1 else (64 if bk == 32 and stages == 5 else 128)
        bn = 8 if mode == -1 else (256 if mode == 9 and stages == 2 else 128)
        lds = 0 if mode == -1 else (bm + bn) * bk * stages * 2
        entry = meta[symbol]
        rows = body.splitlines()
        overlaps = 0
        for index, row in enumerate(rows):
            if "global_load_b128" not in row:
                continue
            for after in rows[index + 1 :]:
                if "s_wait_loadcnt" in after:
                    break
                if "v_wmma_" in after:
                    overlaps += 1
                    break
        kernels.append(
            dict(
                symbol=symbol,
                mode=mode,
                stages=stages,
                tile_mnk=[bm, bn, bk],
                lds_bytes=lds,
                dynamic_lds=mode >= 2,
                wmma=body.count("v_wmma_f32_16x16x16_f16"),
                scratch=body.count("scratch_"),
                load_before_wmma_before_wait_sites=overlaps,
                **{
                    key[1:]: entry[key]
                    for key in (
                        ".vgpr_count",
                        ".sgpr_count",
                        ".vgpr_spill_count",
                        ".private_segment_fixed_size",
                    )
                },
            )
        )
    expected = {
        (mode, s)
        for name, mode in VARIANTS.items()
        for cfg in options(name)
        for s in [cfg[0] if cfg else 1]
    }
    assert {(k["mode"], k["stages"]) for k in kernels} == expected
    assert len(kernels) == 36 and all(
        k["wmma"] and k["lds_bytes"] <= 65536 for k in kernels
    )
    return kernels


def options(name, exhaustive=False):
    if "_stages" not in name:
        return [()]
    maximum = 4 if VARIANTS[name] in (1, 9) else 5
    return [
        (s, swizzle, stride)
        for s in range(2, maximum + 1)
        for swizzle, stride in (
            ((False, 256), (True, 256), (True, 512))
            if exhaustive
            else ((s % 2 == 1, 256),)
        )
    ]


def packed(b, name):
    return (
        b.T.contiguous().reshape_as(b)
        if "_tn" in name or name.startswith("ws_hgemm")
        else b
    )


def fixtures(path):
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(20261105)
    cases = []
    for seed in (20261105, 17):
        g.manual_seed(seed)
        for m, n, k in (
            (256, 256, 256),
            (256, 512, 512),
            (512, 512, 128),
            (256, 1024, 160),
            (256, 256, 4096),
        ):
            a = (torch.randn(m, k, generator=g) * 0.25).half()
            b = (torch.randn(k, n, generator=g) * 0.25).half()
            cases.append(
                dict(
                    id=f"s{seed}_{m}_{n}_{k}",
                    a=a,
                    b=b,
                    reference=a.double() @ b.double(),
                    input_sha256=[tensor_sha(a), tensor_sha(b)],
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, path)


def numerical(fn, a, b, config, checks, identifier, reference=None):
    storage = torch.full(
        (a.shape[0] * b.shape[1] + 64,), 42, device=a.device, dtype=a.dtype
    )
    out = storage[32:-32].view(a.shape[0], b.shape[1])
    out.fill_(float("nan"))
    fn(a, packed(b, fn.__name__), out, *config)
    if reference is None:
        reference = a.double() @ b.double()
    actual = out.cpu()
    ref = reference.cpu()
    canary = bool((storage[:32] == 42).all() and (storage[-32:] == 42).all())
    passed = torch.allclose(actual, ref.half(), atol=ATOL, rtol=RTOL) and canary
    checks.append(
        dict(
            id=identifier,
            shape_mnk=[a.shape[0], b.shape[1], a.shape[1]],
            config=config,
            passed=passed,
            atol=ATOL,
            rtol=RTOL,
            finite=bool(actual.isfinite().all()),
            canary_passed=canary,
            max_abs_error_fp64=(
                (actual.double() - ref).abs().max().item()
                if actual.numel()
                else 0
            ),
        )
    )
    return actual


def boundaries(lib, checks, variants=None):
    for name in VARIANTS if variants is None else variants:
        fn = getattr(lib, name)
        for config in options(name, True):
            for seed in (17, 93):
                torch.manual_seed(seed)
                for m, n, k in (
                    (0, 17, 8),
                    (17, 0, 8),
                    (3, 5, 0),
                    (1, 1, 1),
                    (17, 33, 15),
                    (65, 129, 17),
                    (129, 257, 65),
                    (257, 263, 257),
                    (17, 19, 8193),
                ):
                    a, b = [
                        (torch.randn(shape, device="cuda") * 0.25).half()
                        for shape in ((m, k), (k, n))
                    ]
                    numerical(
                        fn,
                        a,
                        b,
                        config,
                        checks,
                        f"{name}/{config}/{seed}/{m}_{n}_{k}",
                    )
            for pattern in (
                "zero",
                "cancellation",
                "reciprocal-range",
                "unaligned-contiguous",
            ):
                a, b = [
                    (torch.randn(shape, device="cuda") * 0.25).half()
                    for shape in ((65, 256), (256, 129))
                ]
                if pattern == "zero":
                    a.zero_()
                elif pattern == "cancellation":
                    a[:, 1::2] = a[:, ::2]
                    b[1::2] = -b[::2]
                elif pattern == "reciprocal-range":
                    a[:, ::2] *= 256
                    b[::2] /= 256
                    a[:, 1::2] /= 256
                    b[1::2] *= 256
                else:
                    storage = torch.empty(
                        a.numel() + 1, device="cuda", dtype=a.dtype
                    )
                    view = storage[1:].view_as(a)
                    view.copy_(a)
                    a = view
                numerical(
                    fn, a, b, config, checks, f"{name}/{config}/{pattern}"
                )
        a, b, out = [
            torch.ones(32, 32, device="cuda", dtype=torch.float16)
            for _ in range(3)
        ]
        conf = options(name)[0]
        invalid = {
            "cpu": lambda: fn(a.cpu(), b, out, *conf),
            "type": lambda: fn(a.float(), b, out, *conf),
            "output-type": lambda: fn(a, b, out.float(), *conf),
            "rank": lambda: fn(a.flatten(), b, out, *conf),
            "shape": lambda: fn(a, b, out[:31], *conf),
            "strides": lambda: fn(a.T, b, out, *conf),
            "alias-a": lambda: fn(a, b, a, *conf),
            "alias-b": lambda: fn(a, b, b, *conf),
            "gradient": lambda: fn(a.clone().requires_grad_(), b, out, *conf),
        }
        if conf:
            invalid.update(
                {
                    "stride-zero": lambda: fn(a, b, out, 2, True, 0),
                }
            )
        for kind, call in invalid.items():
            try:
                call()
            except RuntimeError:
                checks.append(dict(id=name + "/reject-" + kind, passed=True))
            else:
                checks.append(dict(id=name + "/reject-" + kind, passed=False))
        if conf:
            first = torch.empty_like(out)
            fn(a, b, first, 2, False, 256)
            max_stage = 4 if VARIANTS[name] in (1, 9) else 5
            for stage in (-3, 0, 1, max_stage + 1, 99):
                fn(a, b, out, stage, False, 256)
                checks.append(
                    dict(
                        id=f"{name}/stage-alias-{stage}",
                        passed=torch.equal(out, first),
                    )
                )
        if torch.cuda.device_count() > 1:
            selected = torch.cuda.current_device()
            other = 1 - selected
            try:
                fn(a, b.to(other), out, *conf)
            except RuntimeError:
                checks.append(
                    dict(id=f"{name}/reject-mixed-device", passed=True)
                )
            else:
                checks.append(
                    dict(id=f"{name}/reject-mixed-device", passed=False)
                )
            with torch.cuda.device(other):
                fn(a, b, out, *conf)
                restored = torch.cuda.current_device() == other
            checks.append(
                dict(
                    id=f"{name}/device-guard",
                    passed=restored and bool((out == 32).all()),
                )
            )
        # Exact state test distinguishes the explicit upgrade from FP16 MMA.
        for cfg in options(name):
            aa = torch.ones(128, 512, device="cuda", dtype=torch.float16)
            bb = torch.full(
                (512, 256), 2**-12, device="cuda", dtype=torch.float16
            )
            bb[:16] = 1
            cc = torch.empty(128, 256, device="cuda", dtype=torch.float16)
            fn(aa, packed(bb, name), cc, *cfg)
            expected = torch.tensor(16 + 496 * 2**-12).half().item()
            checks.append(
                dict(
                    id=f"{name}/{cfg}/FP32-accumulator",
                    passed=bool((cc == expected).all()),
                    expected=expected,
                    fp16_accumulation=16.0,
                )
            )
        # Independent outputs across streams, several queued calls before sync.
        stream_cases = []
        for index in range(8):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                aa = torch.full(
                    (65, 97), index + 1.0, device="cuda", dtype=torch.float16
                )
                bb = torch.full(
                    (97, 129), 0.125, device="cuda", dtype=torch.float16
                )
                cc = torch.empty(65, 129, device="cuda", dtype=torch.float16)
                for _ in range(4):
                    fn(aa, bb, cc, *conf)
                stream_cases.append((stream, aa, bb, cc, index))
        for stream, aa, bb, cc, index in stream_cases:
            stream.synchronize()
            checks.append(
                dict(
                    id=f"{name}/concurrent-stream-{index}",
                    passed=bool((cc == (index + 1) * 97 * 0.125).all()),
                )
            )
        print(
            name,
            "boundary failures",
            sum(not c["passed"] for c in checks),
            flush=True,
        )


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    builds = {}
    if args.backend == "hip":
        from .environment import verify
        from .loader import load, source_paths

        identity = verify()
        paths = source_paths("hgemm")
        lib = load("hgemm")
        funcs = {name: getattr(lib, name) for name in VARIANTS}
        builds["hip"] = dict(
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
            binary_sha256=sha(lib.__file__),
        )
    else:
        from .cuda_pending_check import build

        args.cuda_current_stream = True

        identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
        )
        manifest = json.loads(args.manifest.read_text())
        funcs = {}
        for group in manifest["groups"]:
            if not any(f["name"] in VARIANTS for f in group["functions"]):
                continue
            lib, metadata = build(group, args)
            builds[group["source"]] = metadata
            funcs.update(
                {f["name"]: getattr(lib, f["name"]) for f in group["functions"]}
            )
        assert set(funcs) == set(VARIANTS)
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        environment=identity,
        backend=args.backend,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        variants=VARIANTS,
        builds=builds,
        original_accumulator="FP16",
        rocm_accumulator="FP32",
        checks=[],
        boundaries=[],
        performance=[],
        passed=False,
    )
    values = {}
    for name, fn in funcs.items():
        for conf in options(name):
            for c in torch.load(args.fixtures, weights_only=True):
                assert c["input_sha256"] == [
                    tensor_sha(c["a"]),
                    tensor_sha(c["b"]),
                ]
                a, b = c["a"].cuda(), c["b"].cuda()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                identifier = f"{name}/{conf}/{c['id']}"
                error = (
                    cuda_resource_error(name, conf)
                    if args.backend == "cuda"
                    else None
                )
                if error:
                    actual = torch.full_like(
                        c["reference"], float("nan"), dtype=torch.float16
                    )
                    report["checks"].append(
                        dict(
                            id=identifier,
                            passed=False,
                            executed=False,
                            execution_error=error,
                            shape_mnk=[a.shape[0], b.shape[1], a.shape[1]],
                            config=conf,
                            atol=ATOL,
                            rtol=RTOL,
                        )
                    )
                else:
                    with torch.cuda.stream(stream):
                        actual = numerical(
                            fn,
                            a,
                            b,
                            conf,
                            report["checks"],
                            identifier,
                            c["reference"],
                        )
                stream.synchronize()
                report["checks"][-1]["input_sha256"] = c["input_sha256"]
                values[identifier] = actual
        print(name, "shared cases done", flush=True)
    if args.backend == "hip":
        boundaries(lib, report["boundaries"])
        from .isa_objects import hip_isa_objects

        isa = hip_isa_objects(
            lib.__file__,
            (args.output.parent / (args.output.stem + "_isa")).resolve(),
        )
        report["isa"] = isa
        report["isa"]["kernels"] = inspect_specializations(isa)
        assembly = (ROOT / isa["assembly"]).read_text()
        assert "v_wmma_f32_16x16x16_f16" in assembly
        assert "global_load_b128" in assembly and "s_wait_loadcnt" in assembly
        report["isa_counts"] = {
            key: len(re.findall(key, assembly))
            for key in (
                "v_wmma_f32_16x16x16_f16",
                "global_load_b128",
                "s_wait_loadcnt",
                "scratch_load",
                "scratch_store",
            )
        }
    if args.performance:
        torch.manual_seed(20261105)
        for size in (256, 1024, 2048):
            a, b = [
                (torch.randn(size, size, device="cuda") * 0.25).half()
                for _ in range(2)
            ]
            out, ref = torch.empty_like(a), torch.empty_like(a)
            for name, fn in funcs.items():
                for conf in options(name) if "_stages" in name else [()]:
                    packed_b = packed(b, name)
                    error = (
                        cuda_resource_error(name, conf)
                        if args.backend == "cuda"
                        else None
                    )
                    if error:
                        report["performance"].append(
                            dict(
                                function=name,
                                shape_mnk=[size] * 3,
                                config=conf,
                                accuracy_passed=False,
                                graph_replay_verified=False,
                                executed=False,
                                execution_error=error,
                            )
                        )
                        continue
                    fn(a, packed_b, out, *conf)
                    torch.mm(a, b, out=ref)
                    passed = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
                    prior = out.clone()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        fn(a, packed_b, out, *conf)
                    with torch.cuda.stream(stream):
                        out.fill_(float("nan"))
                        graph.replay()
                    stream.synchronize()
                    graph_passed = torch.equal(out, prior)
                    assert (
                        graph_passed
                    ), "empty or incorrect captured kernel graph"
                    perf = graph_pair(
                        lambda: fn(a, packed_b, out, *conf),
                        lambda: torch.mm(a, b, out=ref),
                        calls=8,
                        sample_ms=10,
                        trials=3,
                    )
                    report["performance"].append(
                        dict(
                            function=name,
                            shape_mnk=[size] * 3,
                            config=conf,
                            accuracy_passed=passed,
                            graph_replay_verified=graph_passed,
                            **perf,
                        )
                    )
            print("performance", size, "done", flush=True)
    report["passed"] = all(
        c["passed"] for c in report["checks"] + report["boundaries"]
    )
    if args.backend == "hip":
        report["passed"] &= all(
            c["accuracy_passed"] for c in report["performance"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(values, args.output.with_suffix(".pt"))
    report["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                shared=len(report["checks"]),
                boundaries=len(report["boundaries"]),
                performance=len(report["performance"]),
            )
        )
    )
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["fixtures", "run"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--backend", choices=["hip", "cuda"], default="hip")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "docs/rocm/cross_5090_20261105/pending56_manifest.json",
    )
    parser.add_argument("--source-root", type=Path, default=ROOT / "original")
    parser.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    args = parser.parse_args()
    if args.command == "fixtures":
        fixtures(args.fixtures)
        return 0
    if args.output is None:
        parser.error("run requires --output")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
