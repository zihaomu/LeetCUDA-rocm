"""Original shared-KV/QKV attention: CPU oracle, CUDA parity, LDS/ISA and timing."""

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha

NAMES = tuple(
    "flash_attn_mma_stages_split_q_shared_" + kind + "_acc_f32"
    for kind in ("kv", "qkv")
)
TOL = 0.003


def reference(q, k, v):
    return (
        q.double() @ k.double().transpose(-1, -2) / math.sqrt(q.shape[-1])
    ).softmax(-1) @ v.double()


def fixtures(path):
    torch.set_num_threads(4)
    rows = []
    g = torch.Generator()

    def add(identifier, shape, pattern="random", performance=False):
        q, k, v = [
            (torch.randn(shape, generator=g) * 0.5).half() for _ in range(3)
        ]
        if pattern == "zero-v":
            v.zero_()
        elif pattern == "uniform":
            q.zero_()
        elif pattern == "large-logits":
            q *= 16
            k *= 16
        elif pattern == "cancellation":
            q.zero_()
            v[..., 1::2, :] = -v[..., ::2, :]
        rows.append(
            dict(
                id=identifier,
                inputs=(q, k, v),
                reference=reference(q, k, v),
                input_sha256=[tensor_sha(t) for t in (q, k, v)],
                performance=performance,
            )
        )

    for seed in (20261105, 17):
        g.manual_seed(seed)
        for n in (128, 256, 384):
            for d in (32, 64, 96, 128, 256):
                add(f"seed{seed}_n{n}_d{d}", (1, 2, n, d))
    for pattern in ("zero-v", "uniform", "large-logits", "cancellation"):
        add(pattern, (2, 3, 256, 64), pattern)
    for n, d in (
        (128, 32),
        (256, 64),
        (512, 64),
        (1024, 64),
        (1024, 128),
        (256, 256),
    ):
        add(f"perf_n{n}_d{d}", (1, 4, n, d), performance=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)
    print(f"Saved {len(rows)} CPU fixtures: {sha(path)}")


def numerical(fn, inputs, stage, identifier, ref=None):
    out = torch.full_like(inputs[0], float("nan"))
    fn(*inputs, out, stage)
    actual = out.cpu()
    ref = (reference(*inputs) if ref is None else ref).cpu()
    return actual, dict(
        id=identifier,
        stage=stage,
        shape=list(out.shape),
        passed=bool(torch.isfinite(actual).all())
        and torch.allclose(actual, ref.half(), atol=TOL, rtol=TOL),
        atol=TOL,
        rtol=TOL,
        max_abs_error_fp64=(
            (actual.double() - ref).abs().max().item() if actual.numel() else 0
        ),
    )


def boundaries(lib):
    checks = []
    for name in NAMES:
        fn = getattr(lib, name)
        for d in (32, 64, 96, 128, 256):
            for stage in (1,) if d == 256 else (1, 2):
                for seed in (17, 93):
                    torch.manual_seed(seed)
                    for n in (1, 17, 63, 65, 127, 129, 257):
                        inputs = [
                            (
                                torch.randn(1, 2, n, d, device="cuda") * 0.5
                            ).half()
                            for _ in range(3)
                        ]
                        _, row = numerical(
                            fn,
                            inputs,
                            stage,
                            f"{name}/d{d}/s{stage}/n{n}/seed{seed}",
                        )
                        checks.append(row)
                for shape in (
                    (0, 2, 17, d),
                    (1, 0, 17, d),
                    (1, 2, 0, d),
                    (2, 3, 65, d),
                ):
                    inputs = [
                        torch.randn(shape, device="cuda", dtype=torch.float16)
                        * 0.5
                        for _ in range(3)
                    ]
                    _, row = numerical(
                        fn, inputs, stage, f"{name}/s{stage}/shape{shape}"
                    )
                    checks.append(row)
                inputs = []
                for _ in range(3):
                    raw = (
                        torch.randn(
                            65 * d + 1, device="cuda", dtype=torch.float16
                        )
                        * 0.5
                    )
                    inputs.append(raw[1:].view(1, 1, 65, d))
                _, row = numerical(
                    fn, inputs, stage, f"{name}/d{d}/s{stage}/unaligned"
                )
                checks.append(row)
        inputs = [
            torch.randn(1, 2, 65, 64, device="cuda", dtype=torch.float16) * 0.5
            for _ in range(3)
        ]
        for stage in (-3, 0, 3, 5):
            _, row = numerical(fn, inputs, stage, f"{name}/stage-alias-{stage}")
            checks.append(row)
        q, k, v = inputs
        out = torch.empty_like(q)
        invalid = {
            "cpu": lambda: fn(q.cpu(), k, v, out, 1),
            "dtype-q": lambda: fn(q.float(), k, v, out, 1),
            "dtype-v": lambda: fn(q, k, v.float(), out, 1),
            "dtype-o": lambda: fn(q, k, v, out.float(), 1),
            "shape": lambda: fn(q, k[:, :, :64], v, out, 1),
            "rank": lambda: fn(q.flatten(), k, v, out, 1),
            "stride": lambda: fn(
                q.transpose(-1, -2),
                k.transpose(-1, -2),
                v.transpose(-1, -2),
                out.transpose(-1, -2),
                1,
            ),
            "alias-q": lambda: fn(q, k, v, q, 1),
            "alias-k": lambda: fn(q, k, v, k, 1),
            "alias-v": lambda: fn(q, k, v, v, 1),
            "gradient": lambda: fn(q.clone().requires_grad_(), k, v, out, 1),
            "head-16": lambda: fn(
                *[
                    torch.empty(
                        1, 1, 65, 16, device="cuda", dtype=torch.float16
                    )
                    for _ in range(4)
                ],
                1,
            ),
            "head-256-stage2": lambda: fn(
                *[
                    torch.empty(
                        1, 1, 65, 256, device="cuda", dtype=torch.float16
                    )
                    for _ in range(4)
                ],
                2,
            ),
        }
        if torch.cuda.device_count() > 1:
            other = 1 - torch.cuda.current_device()
            invalid["mixed-device"] = lambda: fn(q, k.to(other), v, out, 1)
            selected = torch.cuda.current_device()
            with torch.cuda.device(other):
                fn(q, k, v, out, 2)
                restored = torch.cuda.current_device() == other
            passed = restored and torch.allclose(
                out, reference(q, k, v).half(), atol=TOL, rtol=TOL
            )
            checks.append(
                dict(
                    id=name + "/device-guard",
                    passed=passed,
                    tensor_device=selected,
                    caller_device=other,
                )
            )
        for label, call in invalid.items():
            try:
                call()
            except RuntimeError:
                checks.append(dict(id=name + "/reject-" + label, passed=True))
            else:
                checks.append(dict(id=name + "/reject-" + label, passed=False))
        queued = []
        for i in range(6):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                q = torch.zeros(
                    1, 2, 65, 64, device="cuda", dtype=torch.float16
                )
                k = torch.zeros_like(q)
                v = torch.full_like(q, (i + 1) * 0.125)
                out = torch.empty_like(q)
                for _ in range(4):
                    fn(q, k, v, out, 1 + i % 2)
                queued.append((stream, q, k, v, out, i))
        for stream, q, k, v, out, i in queued:
            stream.synchronize()
            checks.append(
                dict(
                    id=f"{name}/stream{i}",
                    passed=bool((out == (i + 1) * 0.125).all()),
                )
            )
        print(
            name,
            "boundaries",
            sum(not c["passed"] for c in checks),
            "failures",
            flush=True,
        )
    return checks


def inspect_isa(lib, destination):
    from .isa_objects import hip_isa_objects

    isa = hip_isa_objects(lib.__file__, destination.resolve())
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n",
        (ROOT / isa["assembly"]).read_text(),
        flags=re.M,
    )
    kernels = []
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        match = re.search(
            r"attention_sharedILi(\d+)ELi([12])ELb([01])EEE", symbol
        )
        if not match:
            continue
        d, stage, shareq = map(int, match.groups())
        br = 64 if d == 256 else 128
        bc = 32 if not shareq and d >= 128 else 64
        data = (
            max(br * d, stage * bc * d) if shareq else br * d + stage * bc * d
        )
        overlap = 0
        lines = body.splitlines()
        for idx, line in enumerate(lines):
            if "global_load_b128" not in line:
                continue
            for later in lines[idx + 1 :]:
                if "s_wait_loadcnt" in later:
                    break
                if "v_wmma_" in later:
                    overlap += 1
                    break
        kernels.append(
            dict(
                symbol=symbol,
                d=d,
                stage=stage,
                shared_qkv=bool(shareq),
                query_tile=br,
                key_tile=bc,
                dynamic_lds_bytes=(data + br * bc) * 2,
                wmma=body.count("v_wmma_f32_16x16x16_f16"),
                scratch=body.count("scratch_"),
                prefetch_order_sites=overlap,
            )
        )
    assert len(kernels) == 18
    assert all(k["wmma"] > 0 for k in kernels)
    isa["kernels"] = kernels
    isa["ordering_note"] = (
        "Static linear load -> WMMA -> wait sites, not a cycle-level overlap measurement."
    )
    return isa


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    builds = {}
    if args.backend == "hip":
        from .environment import verify
        from .loader import load, source_paths

        identity = verify()
        lib = load("flash_attn")
        functions = {name: getattr(lib, name) for name in NAMES}
        builds["hip"] = dict(
            source_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in source_paths("flash_attn")
            },
            binary_sha256=sha(lib.__file__),
        )
    else:
        from .cuda_pending_check import build

        args.cuda_current_stream = True
        functions = {}
        identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            gpu_uuid=str(torch.cuda.get_device_properties(args.device).uuid),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
        )
        for group in json.loads(args.manifest.read_text())["groups"]:
            names = [f["name"] for f in group["functions"]]
            if not set(names) & set(NAMES):
                continue
            assert set(names) <= set(NAMES)
            lib, metadata = build(group, args)
            builds[group["source"]] = metadata
            functions.update({name: getattr(lib, name) for name in names})
        assert set(functions) == set(NAMES)
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        environment=identity,
        logical_device=args.device,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        builds=builds,
        checks=[],
        boundaries=[],
        performance=[],
        passed=False,
    )
    outputs = {}
    fixture = torch.load(args.fixtures, weights_only=True)
    for name, fn in functions.items():
        for case in fixture:
            assert case["input_sha256"] == [
                tensor_sha(t) for t in case["inputs"]
            ]
            inputs = [t.cuda() for t in case["inputs"]]
            for stage in (1,) if inputs[0].shape[-1] == 256 else (1, 2):
                identifier = f"{name}/s{stage}/{case['id']}"
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual, row = numerical(
                        fn, inputs, stage, identifier, case["reference"]
                    )
                stream.synchronize()
                row["input_sha256"] = case["input_sha256"]
                row["function"] = name
                report["checks"].append(row)
                outputs[identifier] = actual
                if args.performance and case["performance"]:
                    out = torch.empty_like(inputs[0])
                    fn(*inputs, out, stage)
                    baseline = F.scaled_dot_product_attention(*inputs)
                    baseline_passed = torch.allclose(
                        baseline.cpu(),
                        case["reference"].half(),
                        atol=TOL,
                        rtol=TOL,
                    )
                    prior = out.clone()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        fn(*inputs, out, stage)
                    with torch.cuda.stream(stream):
                        out.fill_(float("nan"))
                        graph.replay()
                    stream.synchronize()
                    replay = torch.equal(out, prior)
                    assert replay
                    perf = graph_pair(
                        lambda: fn(*inputs, out, stage),
                        lambda: F.scaled_dot_product_attention(*inputs),
                        calls=8,
                        sample_ms=10,
                        trials=3,
                    )
                    report["performance"].append(
                        dict(
                            id=identifier,
                            function=name,
                            shape=list(out.shape),
                            stage=stage,
                            input_sha256=case["input_sha256"],
                            accuracy_passed=row["passed"],
                            baseline_accuracy_passed=baseline_passed,
                            graph_replay_verified=replay,
                            **perf,
                        )
                    )
        print(name, "shared complete", flush=True)
    if args.backend == "hip":
        report["boundaries"] = boundaries(lib)
        report["isa"] = inspect_isa(
            lib, args.output.parent / ("isa_device" + str(args.device))
        )
    if args.performance:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            F.scaled_dot_product_attention(*inputs)
            torch.cuda.synchronize()
        report["sdpa_dispatch_sample"] = [
            e.key for e in prof.key_averages() if "scaled_dot_product" in e.key
        ]
    report["passed"] = all(
        c["passed"] for c in report["checks"] + report["boundaries"]
    ) and all(
        p["accuracy_passed"] and p["baseline_accuracy_passed"]
        for p in report["performance"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.output.with_suffix(".pt"))
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


def compare(args):
    paths = (args.cuda, args.hip)
    reports = [json.loads(p.read_text()) for p in paths]
    assert [r["backend"] for r in reports] == ["cuda", "hip"]
    assert reports[0]["fixture_sha256"] == reports[1]["fixture_sha256"]
    assert reports[0]["harness_sha256"] == reports[1]["harness_sha256"]
    values = []
    for p, r in zip(paths, reports):
        assert sha(p.with_suffix(".pt")) == r["outputs_sha256"]
        values.append(torch.load(p.with_suffix(".pt"), weights_only=True))
    checks = [{c["id"]: c for c in r["checks"]} for r in reports]
    assert (
        values[0].keys()
        == values[1].keys()
        == checks[0].keys()
        == checks[1].keys()
    )
    rows = []
    for key, cuda in values[0].items():
        hip = values[1][key]
        assert checks[0][key]["input_sha256"] == checks[1][key]["input_sha256"]
        rows.append(
            dict(
                id=key,
                cuda_accuracy=checks[0][key]["passed"],
                hip_accuracy=checks[1][key]["passed"],
                bitwise_equal=torch.equal(cuda, hip),
                parity_within_gate=torch.allclose(
                    cuda, hip, atol=TOL, rtol=TOL
                ),
                max_abs_cuda_hip=(cuda.float() - hip.float())
                .abs()
                .max()
                .item(),
            )
        )
    summary = {
        k: sum(r[k] for r in rows)
        for k in (
            "cuda_accuracy",
            "hip_accuracy",
            "bitwise_equal",
            "parity_within_gate",
        )
    }
    summary["shared_cases"] = len(rows)
    performance = []
    cp = {p["id"]: p for p in reports[0]["performance"]}
    for hp in reports[1]["performance"]:
        p = cp[hp["id"]]
        assert p["input_sha256"] == hp["input_sha256"]
        performance.append(
            dict(
                id=hp["id"],
                cuda_ms=p["ours_ms"],
                hip_ms=hp["ours_ms"],
                hip_torch_ms=hp["baseline_ms"],
                cuda_accuracy=p["accuracy_passed"],
                hip_accuracy=hp["accuracy_passed"],
            )
        )
    passed = all(r["passed"] for r in reports) and summary[
        "parity_within_gate"
    ] == len(rows)
    result = dict(
        report_sha256={p.name: sha(p) for p in paths},
        fixture_sha256=reports[0]["fixture_sha256"],
        summary=summary,
        checks=rows,
        performance=performance,
        passed=passed,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary))
    return int(not passed)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["fixtures", "run", "compare"])
    parser.add_argument("--backend", choices=["hip", "cuda"], default="hip")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--hip", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT / "original")
    parser.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "docs/rocm/cross_5090_20261105/pending56_manifest.json",
    )
    args = parser.parse_args()
    if args.command == "fixtures":
        fixtures(args.fixtures)
        return 0
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
