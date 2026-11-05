"""Nine original WMMA HGEMMs: shared CUDA inputs, HIP boundaries, ISA and timing."""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha, timed

PREFIX = "hgemm_wmma_"
VARIANTS = {
    PREFIX + "m16n16k16_naive": (16, 16, 1, 1, 1, 1, "global"),
    PREFIX + "m16n16k16_mma4x2": (64, 32, 4, 2, 1, 1, "static"),
    PREFIX + "m16n16k16_mma4x2_warp2x4": (128, 128, 4, 2, 2, 4, "static"),
    PREFIX
    + "m16n16k16_mma4x2_warp2x4_dbuf_async": (
        128,
        128,
        4,
        2,
        2,
        4,
        "prefetch2",
    ),
    PREFIX
    + "m32n8k16_mma2x4_warp2x4_dbuf_async": (128, 128, 2, 4, 4, 2, "prefetch2"),
    PREFIX
    + "m16n16k16_mma4x2_warp2x4_stages": (128, 128, 4, 2, 2, 4, "static-ring"),
    PREFIX
    + "m16n16k16_mma4x2_warp2x4_stages_dsmem": (
        128,
        128,
        4,
        2,
        2,
        4,
        "dynamic-ring",
    ),
    PREFIX
    + "m16n16k16_mma4x2_warp4x4_stages_dsmem": (
        256,
        128,
        4,
        2,
        4,
        4,
        "dynamic-ring",
    ),
    PREFIX
    + "m16n16k16_mma4x4_warp4x4_stages_dsmem": (
        256,
        256,
        4,
        4,
        4,
        4,
        "dynamic-ring",
    ),
}
ATOL = RTOL = 0.01


def options(name, exhaustive=False):
    if "_stages" not in name:
        return [()]
    maximum = 4 if "mma4x4" in name else 5
    configs = [(2, False, 256), (3, True, 256), (4, True, 512)]
    if maximum == 5:
        configs.append((5, True, 256))
    if exhaustive:
        configs = [
            (s, swizzle, stride)
            for s in range(2, maximum + 1)
            for swizzle, stride in ((False, 256), (True, 256), (True, 512))
        ]
    return configs


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
            (256, 1024, 80),
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
    out = torch.full(
        (a.shape[0], b.shape[1]), float("nan"), device=a.device, dtype=a.dtype
    )
    fn(a, b, out, *config)
    if reference is None:
        reference = a.double() @ b.double()
    actual = out.cpu()
    ref = reference.cpu()
    passed = torch.allclose(actual, ref.half(), atol=ATOL, rtol=RTOL)
    checks.append(
        dict(
            id=identifier,
            shape_mnk=[a.shape[0], b.shape[1], a.shape[1]],
            config=config,
            passed=passed,
            atol=ATOL,
            rtol=RTOL,
            finite=bool(actual.isfinite().all()),
            max_abs_error_fp64=(
                (actual.double() - ref).abs().max().item()
                if actual.numel()
                else 0
            ),
        )
    )
    return actual


def boundaries(lib, checks):
    for name in VARIANTS:
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
                    "stage-zero": lambda: fn(a, b, out, 0, False, 256),
                    "stage-six": lambda: fn(a, b, out, 6, False, 256),
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
    builds = {}
    if args.backend == "hip":
        from .environment import verify
        from .loader import load, source_paths

        identity = verify()
        lib = load("hgemm")
        funcs = {name: getattr(lib, name) for name in VARIANTS}
        builds["hip"] = dict(
            source_sha256={
                str(p.relative_to(ROOT)): sha(p) for p in source_paths("hgemm")
            },
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
            if not group["source"].startswith("kernels/hgemm/wmma/"):
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

        isa = hip_isa_objects(lib.__file__, ROOT / "artifacts/rocm/wmma/isa")
        report["isa"] = isa
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
                    fn(a, b, out, *conf)
                    torch.mm(a, b, out=ref)
                    passed = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
                    prior = out.clone()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        fn(a, b, out, *conf)
                    with torch.cuda.stream(stream):
                        out.fill_(float("nan"))
                        graph.replay()
                    stream.synchronize()
                    graph_passed = torch.equal(out, prior)
                    assert (
                        graph_passed
                    ), "empty or incorrect captured kernel graph"
                    perf = graph_pair(
                        lambda: fn(a, b, out, *conf),
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
    parser.add_argument("--backend", choices=["hip", "cuda"], default="hip")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts/cross/pending_manifest.json",
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
