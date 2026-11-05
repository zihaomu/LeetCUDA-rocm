"""Shared attention rr: explicit FP32 state, native register pressure and parity."""

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha
from .flash_shared_check import reference
from .flash_swizzle_check import compare, devices, numerical, telemetry

NAMES = tuple(
    "flash_attn_mma_stages_split_q_shared_" + kind + "_acc_f32_rr"
    for kind in ("kv", "qkv")
)
TOL = 0.003


def stages(name, d):
    return (1,) if d == 256 and "_shared_kv_" in name else (1, 2)


def fixtures(path):
    from .flash_shared_check import fixtures as create

    create(path)
    rows = torch.load(path, weights_only=True)
    for seed in (20261106, 19):
        g = torch.Generator().manual_seed(seed)
        for d in (32, 64, 96, 128, 256):
            inputs = [
                (torch.randn(1, 1, 4096, d, generator=g) * 0.5).half()
                for _ in range(3)
            ]
            rows.append(
                dict(
                    id=f"long_seed{seed}_d{d}",
                    inputs=inputs,
                    reference=reference(*inputs),
                    input_sha256=[tensor_sha(t) for t in inputs],
                    performance=False,
                )
            )
    torch.save(rows, path)
    print(f"Saved {len(rows)} rr CPU fixtures: {sha(path)}", flush=True)


def storage_checks(functions):
    rows = []
    for name, fn in functions.items():
        for d in (32, 64, 96, 128, 256):
            for stage in stages(name, d):
                bc = (
                    32
                    if ("_shared_kv_" in name and d >= 128)
                    or (d == 256 and stage == 2)
                    else 64
                )
                n = bc * 32
                q = torch.zeros(1, 1, n, d, dtype=torch.float16, device="cuda")
                k, v = torch.zeros_like(q), torch.full_like(q, 2**-11)
                v[:, :, :bc] = 1
                state = torch.tensor(0.0)
                low = torch.tensor(0.0)
                for block in range(32):
                    add = bc if block == 0 else bc * 2**-11
                    state = state + add
                    low = (low + add).half().float()
                expected, wrong = (state / n).half().item(), (
                    low / n
                ).half().item()
                out = torch.empty_like(q)
                fn(q, k, v, out, stage)
                rows.append(
                    dict(
                        id=f"{name}/d{d}/s{stage}/FP32-state",
                        passed=bool((out == expected).all()),
                        expected=expected,
                        fp16_state_result=wrong,
                        output_state="FP32",
                    )
                )
    return rows


def inspect_isa(lib, output):
    import yaml
    from .isa_objects import hip_isa_objects

    isa = hip_isa_objects(lib.__file__, output.resolve())
    metadata = {}
    for obj in isa["objects"]:
        notes = subprocess.check_output(
            [
                "/opt/rocm/llvm/bin/llvm-readobj",
                "--notes",
                str(ROOT / obj["hsaco"]),
            ],
            text=True,
        )
        doc = notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
        metadata.update(
            {v[".name"]: v for v in yaml.safe_load(doc)["amdhsa.kernels"]}
        )
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n",
        (ROOT / isa["assembly"]).read_text(),
        flags=re.M,
    )
    kernels = []
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        match = re.search(
            r"attention_sharedILi(\d+)ELi([12])ELb([01])ELb([01])EEE", symbol
        )
        if not match:
            continue
        d, stage, share, rr = map(int, match.groups())
        br = 64 if d == 256 else 128
        bc = (
            32
            if (not share and d >= 128) or (rr and d == 256 and stage == 2)
            else 64
        )
        planes = 2 if rr and d > 64 else 1
        od = ((d + planes * 16 - 1) // (planes * 16)) * 16
        data = max(br * d, stage * bc * d) if share else br * d + stage * bc * d
        entry = metadata[symbol]
        kernels.append(
            dict(
                symbol=symbol,
                d=d,
                stage=stage,
                shared_qkv=bool(share),
                rr=bool(rr),
                query_tile=br,
                key_tile=bc,
                output_panel=od,
                cooperating_output_groups=planes,
                threads=br * 2 * planes,
                output_state="FP32",
                output_state_floats_per_lane=od // 2,
                dynamic_lds_bytes=2 * (data + br * bc),
                wmma=body.count("v_wmma_f32_16x16x16_f16"),
                scratch=body.count("scratch_"),
                **{
                    key[1:]: entry[key]
                    for key in [
                        ".vgpr_count",
                        ".sgpr_count",
                        ".vgpr_spill_count",
                        ".private_segment_fixed_size",
                    ]
                },
            )
        )
    assert sum(k["rr"] for k in kernels) == 19
    assert all(k["wmma"] and k["dynamic_lds_bytes"] <= 65536 for k in kernels)
    isa["kernels"] = kernels
    return isa


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    parents, functions, builds = {}, {}, {}
    if args.backend == "hip":
        from .environment import verify
        from .loader import load, source_paths

        identity = verify()
        lib = load("flash_attn")
        functions = {name: getattr(lib, name) for name in NAMES}
        base = json.loads(args.frozen_baseline.read_text())
        binary = ROOT / base["isa"]["binary"]
        assert sha(binary) == base["isa"]["binary_sha256"]
        spec = importlib.util.spec_from_file_location(
            Path(base["isa"]["build_cache_binary"]).stem, binary
        )
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        parents = {name: getattr(frozen, name[:-3]) for name in NAMES}
        builds["frozen_parent"] = base["builds"]["hip"]
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
        identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            gpu_uuid=str(torch.cuda.get_device_properties(args.device).uuid),
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
        )
        wanted = set(NAMES) | {n[:-3] for n in NAMES}
        found = {}
        for group in json.loads(args.manifest.read_text())["groups"]:
            names = [r["name"] for r in group["functions"]]
            if not wanted.intersection(names):
                continue
            assert set(names) <= wanted
            lib, metadata = build(group, args)
            builds[group["source"]] = metadata
            found.update({name: getattr(lib, name) for name in names})
        functions = {name: found[name] for name in NAMES}
        parents = {name: found[name[:-3]] for name in NAMES}
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        environment=identity,
        logical_device=args.device,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        helper_sha256={
            p: sha(ROOT / p)
            for p in [
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/cross_validate.py",
                "leetcuda_rocm/cuda_pending_check.py",
                "leetcuda_rocm/flash_shared_check.py",
                "leetcuda_rocm/flash_swizzle_check.py",
            ]
        },
        builds=builds,
        checks=[],
        boundaries=[],
        performance=[],
        passed=False,
    )
    fixture = torch.load(args.fixtures, weights_only=True)
    outputs = {}
    for name, fn in functions.items():
        for case in fixture:
            assert case["input_sha256"] == [
                tensor_sha(t) for t in case["inputs"]
            ]
            inputs = [t.cuda() for t in case["inputs"]]
            for stage in stages(name, inputs[0].shape[-1]):
                ident = f"{name}/s{stage}/{case['id']}"
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual, row = numerical(
                        fn, inputs, stage, ident, case["reference"]
                    )
                stream.synchronize()
                row.update(function=name, input_sha256=case["input_sha256"])
                if not (inputs[0].shape[-1] == 256 and stage == 2):
                    out = torch.empty_like(inputs[0])
                    parents[name](*inputs, out, stage)
                    row["parent_bitwise_equal"] = torch.equal(out.cpu(), actual)
                    if args.backend == "hip":
                        row["passed"] &= row["parent_bitwise_equal"]
                else:
                    row["parent_bitwise_equal"] = None
                    row["parent_note"] = (
                        "The frozen AMD parent rejects D256/stage2; rr uses BC32 to fit LDS."
                    )
                outputs[ident] = actual
                report["checks"].append(row)
        print(name, "shared complete", flush=True)
    if args.performance:
        cases = [c for c in fixture if c["performance"]]
        for rd in range(3):
            before = telemetry(args.backend)
            for name in list(functions)[rd % 2 :] + list(functions)[: rd % 2]:
                fn, parent = functions[name], parents[name]
                for case in cases[rd:] + cases[:rd]:
                    inputs = [t.cuda() for t in case["inputs"]]
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as prof:
                        baseline = F.scaled_dot_product_attention(*inputs)
                        torch.cuda.synchronize()
                    dispatch = [
                        e.key
                        for e in prof.key_averages()
                        if "scaled_dot_product" in e.key
                    ]
                    for stage in stages(name, inputs[0].shape[-1]):
                        ps = 1 if inputs[0].shape[-1] == 256 else stage
                        out, po = torch.empty_like(inputs[0]), torch.empty_like(
                            inputs[0]
                        )
                        fn(*inputs, out, stage)
                        parent(*inputs, po, ps)
                        correct = all(
                            torch.allclose(
                                t.cpu(),
                                case["reference"].half(),
                                atol=TOL,
                                rtol=TOL,
                            )
                            for t in [out, po, baseline]
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
                        assert correct and replay
                        pp = graph_pair(
                            lambda: fn(*inputs, out, stage),
                            lambda: parent(*inputs, po, ps),
                            calls=8,
                            sample_ms=40,
                            trials=5,
                        )
                        sp = graph_pair(
                            lambda: fn(*inputs, out, stage),
                            lambda: F.scaled_dot_product_attention(*inputs),
                            calls=8,
                            sample_ms=40,
                            trials=5,
                        )
                        report["performance"].append(
                            dict(
                                id=f"{name}/s{stage}/{case['id']}/round{rd}",
                                function=name,
                                round=rd,
                                stage=stage,
                                parent_stage=ps,
                                same_parent_contract=ps == stage,
                                shape=list(out.shape),
                                input_sha256=case["input_sha256"],
                                accuracy_passed=correct,
                                baseline_accuracy_passed=correct,
                                graph_replay_verified=replay,
                                ours_ms=pp["ours_ms"],
                                parent_ms=pp["baseline_ms"],
                                baseline_ms=sp["baseline_ms"],
                                parent_pair=pp,
                                sdpa_pair=sp,
                                sdpa_dispatch=dispatch,
                            )
                        )
            report.setdefault("telemetry", []).append(
                dict(round=rd, before=before, after=telemetry(args.backend))
            )
            print(f"rr performance round {rd+1}/3 complete", flush=True)
    report["storage_checks"] = storage_checks(functions)
    if args.backend == "hip":
        report["boundaries"] = boundaries(lib)
        report["device_checks"] = devices(lib)
        report["isa"] = inspect_isa(
            lib, args.output.parent / (args.output.stem + "_isa")
        )
    report["passed"] = all(
        c["passed"]
        for key in ["checks", "boundaries", "storage_checks", "device_checks"]
        for c in report.get(key, [])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.output.with_suffix(".pt"))
    report["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                **{
                    k: len(report.get(k, []))
                    for k in [
                        "checks",
                        "boundaries",
                        "storage_checks",
                        "device_checks",
                        "performance",
                    ]
                },
            }
        )
    )
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["fixtures", "run", "compare"])
    parser.add_argument("--backend", choices=["hip", "cuda"], default="hip")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument(
        "--frozen-baseline",
        type=Path,
        default=ROOT
        / "artifacts/rocm/continuation_all/b2/baseline/attention.json",
    )
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


def boundaries(lib):
    checks = []
    for name in NAMES:
        fn = getattr(lib, name)
        for d in (32, 64, 96, 128, 256):
            for stage in stages(name, d):
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
        }
        if "_shared_kv_" in name:
            invalid["head-256-stage2"] = lambda: fn(
                *[
                    torch.empty(
                        1, 1, 65, 256, device="cuda", dtype=torch.float16
                    )
                    for _ in range(4)
                ],
                2,
            )
        invalid["grid-limit"] = lambda: fn(
            *[
                torch.empty(65536, 1, 0, 64, device="cuda", dtype=torch.float16)
                for _ in range(4)
            ],
            1,
        )
        invalid["index-limit"] = lambda: fn(
            *[
                torch.empty(
                    0, 1, 2147483647, 64, device="cuda", dtype=torch.float16
                )
                for _ in range(4)
            ],
            1,
        )
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


if __name__ == "__main__":
    raise SystemExit(main())
