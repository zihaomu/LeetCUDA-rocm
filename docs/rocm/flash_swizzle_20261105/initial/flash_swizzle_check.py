"""Swizzled tiling-QKV: frozen CUDA/AMD fixtures, layout and paired timing."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha

PARENT = "flash_attn_mma_stages_split_q_tiling_qkv_acc_f32"
VARIANTS = {kind: PARENT + "_swizzle_" + kind for kind in ("q", "qk", "qkv")}
NAMES = tuple(VARIANTS.values())
TOL = 0.003


def reference(q, k, v):
    return (
        q.double() @ k.double().transpose(-1, -2) / math.sqrt(q.shape[-1])
    ).softmax(-1) @ v.double()


def fixtures(path):
    from .flash_tiled_check import fixtures as create

    create(path)


def numerical(fn, inputs, stage, identifier, ref=None):
    raw = torch.full(
        (inputs[0].numel() + 32,),
        123.0,
        device=inputs[0].device,
        dtype=inputs[0].dtype,
    )
    out = raw[16:-16].view_as(inputs[0])
    out.fill_(float("nan"))
    fn(*inputs, out, stage)
    actual = out.cpu()
    ref = (reference(*inputs) if ref is None else ref).cpu()
    return actual, dict(
        id=identifier,
        stage=stage,
        shape=list(out.shape),
        passed=bool(torch.isfinite(actual).all())
        and torch.allclose(actual, ref.half(), atol=TOL, rtol=TOL)
        and bool((raw[:16] == 123).all() and (raw[-16:] == 123).all()),
        canary_passed=bool(
            (raw[:16] == 123).all() and (raw[-16:] == 123).all()
        ),
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
        for d in (32, 64, 96, 128, 256, 512, 1024):
            for stage in (1, 2):
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
            "grid-limit": lambda: fn(
                *[
                    torch.empty(
                        65536, 1, 0, 64, device="cuda", dtype=torch.float16
                    )
                    for _ in range(4)
                ],
                1,
            ),
            "index-limit": lambda: fn(
                *[
                    torch.empty(
                        0, 1, 2147483647, 64, device="cuda", dtype=torch.float16
                    )
                    for _ in range(4)
                ],
                1,
            ),
            "head-16": lambda: fn(
                *[
                    torch.empty(
                        1, 1, 65, 16, device="cuda", dtype=torch.float16
                    )
                    for _ in range(4)
                ],
                1,
            ),
            "head-192": lambda: fn(
                *[
                    torch.empty(
                        1, 1, 65, 192, device="cuda", dtype=torch.float16
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
            r"attention_tiledILi(\d+)ELi([12])ELb([01])ELi([137])EEE", symbol
        )
        if not match:
            continue
        d, stage, tile_v, mask = map(int, match.groups())
        br = bc = 64 if d < 128 else 128
        od = min(d, 128)
        data = stage * max((br + bc) * 16, bc * 16 if tile_v else 16 * od)
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
                tile_v=bool(tile_v),
                swizzle_mask=mask,
                code_sha256=hashlib.sha256(body.encode()).hexdigest(),
                query_tile=br,
                key_tile=bc,
                output_panel=od,
                output_state="FP16" if d >= 256 else "FP32",
                dynamic_lds_bytes=(data + br * bc) * 2,
                wmma=body.count("v_wmma_f32_16x16x16_f16"),
                scratch=body.count("scratch_"),
                prefetch_order_sites=overlap,
            )
        )
    assert len(kernels) >= 14
    metadata = {}
    import yaml

    for obj in isa["objects"]:
        notes = subprocess.check_output(
            [
                "/opt/rocm/llvm/bin/llvm-readobj",
                "--notes",
                str(ROOT / obj["hsaco"]),
            ],
            text=True,
        )
        document = notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
        for entry in yaml.safe_load(document)["amdhsa.kernels"]:
            metadata[entry[".name"]] = entry
    for row in kernels:
        entry = metadata[row["symbol"]]
        row.update(
            {
                key[1:]: entry[key]
                for key in (
                    ".vgpr_count",
                    ".sgpr_count",
                    ".vgpr_spill_count",
                    ".private_segment_fixed_size",
                )
            }
        )
    assert all(k["wmma"] > 0 for k in kernels)
    isa["kernels"] = kernels
    isa["ordering_note"] = (
        "Static linear load -> WMMA -> wait sites, not a cycle-level overlap measurement."
    )
    return isa


def storage_checks(functions):
    # Exact dyadic inputs separate the original running-FP16 state from an
    # accidentally strengthened FP32 state, without relying on loose tolerances.
    checks = []
    for name, fn in functions.items():
        for d in (32, 64, 96, 128, 256, 512, 1024):
            bc = 64 if d < 128 else 128
            n = bc * 32
            q = torch.zeros(1, 1, n, d, device="cuda", dtype=torch.float16)
            k = torch.zeros_like(q)
            v = torch.full_like(q, 2.0**-11)
            v[:, :, :bc] = 1
            state = torch.tensor(0.0, dtype=torch.float32)
            for block in range(32):
                state = state + (bc if block == 0 else bc * 2.0**-11)
                if d >= 256:
                    state = state.half().float()
            expected = (state / n).half().item()
            fp32_value = (
                torch.tensor((bc + 31 * bc * 2.0**-11) / n).half().item()
            )
            for stage in (1, 2):
                out = torch.empty_like(q)
                fn(q, k, v, out, stage)
                checks.append(
                    dict(
                        id=f"{name}/storage/d{d}/s{stage}",
                        output_state="FP16" if d >= 256 else "FP32",
                        expected=expected,
                        fp32_state_result=fp32_value,
                        passed=bool((out == expected).all()),
                        distinguishes_fp16_state=(
                            (expected != fp32_value) if d >= 256 else None
                        ),
                    )
                )
    return checks


def devices(lib):
    if torch.cuda.device_count() < 2:
        return [dict(id="two-visible-devices", passed=False)]
    selected, other = (
        torch.cuda.current_device(),
        1 - torch.cuda.current_device(),
    )
    rows = []
    for name in sorted(n for n in dir(lib) if n.startswith("flash_attn_")):
        fn = getattr(lib, name)
        for device in (selected, other):
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    q = torch.zeros(
                        1, 2, 65, 64, device="cuda", dtype=torch.float16
                    )
                    k, v, out = (
                        torch.zeros_like(q),
                        torch.full_like(q, 0.375),
                        torch.empty_like(q),
                    )
                    with torch.cuda.device(1 - device):
                        fn(q, k, v, out, 2)
                        restored = torch.cuda.current_device() == 1 - device
                stream.synchronize()
                rows.append(
                    dict(
                        id=f"{name}/device{device}/foreign-current-nondefault-stream",
                        passed=restored and bool((out == 0.375).all()),
                    )
                )
                try:
                    fn(q, k.to(1 - device), v, out, 1)
                except RuntimeError:
                    rejected = True
                else:
                    rejected = False
                rows.append(
                    dict(
                        id=f"{name}/device{device}/mixed-device",
                        passed=rejected,
                    )
                )
    assert torch.cuda.current_device() == selected
    return rows


def layout_checks():
    from .environment import cache_key, verify
    from .loader import _load, _prepare

    files = [
        ROOT / "kernels/rocm/flash_attention_tiled.cu",
        ROOT / "leetcuda_rocm/native/flash_swizzle_probe.cu",
    ]
    source = "\n".join(p.read_text() for p in files)
    source += '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("probe", &layout_probe); }\n'
    identity = verify()
    lib = _load(
        "flash_swizzle_probe",
        (_prepare(source),),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )
    checks = []
    for rows in (64, 128):
        x = torch.arange(rows * 16, dtype=torch.float16).reshape(rows, 16)
        inputs = torch.stack((x, -x - 1))
        for transpose in (False, True):
            expected = (
                inputs.transpose(1, 2).contiguous() if transpose else inputs
            )
            nr, nc = expected.shape[1:]
            for swizzle in (False, True):
                indices = torch.tensor(
                    [
                        r * nc + (c ^ ((r & 7) << 1) if swizzle else c)
                        for r in range(nr)
                        for c in range(nc)
                    ]
                )
                valid = torch.equal(
                    indices.sort().values, torch.arange(nr * nc)
                ) and torch.equal(indices[indices], torch.arange(nr * nc))
                physical_expected = torch.empty_like(expected).reshape(2, -1)
                physical_expected[:, indices] = expected.reshape(2, -1)
                physical, logical = lib.probe(inputs.cuda(), transpose, swizzle)
                passed = (
                    valid
                    and torch.equal(
                        physical.cpu().reshape(2, -1), physical_expected
                    )
                    and torch.equal(
                        logical.cpu().reshape_as(expected), expected
                    )
                )
                checks.append(
                    dict(
                        id=f"layout/rows{rows}/transpose{transpose}/swizzle{swizzle}",
                        passed=passed,
                        slots=2,
                        elements_checked=inputs.numel(),
                        source_sha256={
                            str(p.relative_to(ROOT)): sha(p) for p in files
                        },
                        binary_sha256=sha(lib.__file__),
                    )
                )
    return checks


def telemetry(backend):
    if backend == "hip":
        cards = {}
        for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
            if not (device / "gpu_busy_percent").exists():
                continue
            paths = [
                device / n
                for n in (
                    "gpu_busy_percent",
                    "mem_info_vram_used",
                    "pp_dpm_sclk",
                    "pp_dpm_mclk",
                )
            ]
            paths += list(device.glob("hwmon/hwmon*/temp*_input"))
            values = {}
            for path in paths:
                try:
                    values[path.name] = path.read_text().strip()
                except OSError as exc:
                    values[path.name] = str(exc)
            cards[device.resolve().name] = values
        assert cards, "GPU sysfs telemetry unavailable"
        return dict(
            method="read-only sysfs; temp*_input in milli-Celsius", cards=cards
        )
    command = [
        "nvidia-smi",
        "--query-gpu=pci.bus_id,temperature.gpu,clocks.sm,utilization.gpu,memory.used",
        "--format=csv,noheader",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    return dict(
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def performance(functions, parent, fixture, args, report):
    rows = []
    cases = [c for c in fixture if c["performance"]]
    for round_id in range(args.rounds):
        before = telemetry(args.backend)
        ordered = cases[round_id:] + cases[:round_id]
        for name in functions:
            fn = functions[name]
            for case in ordered:
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
                for stage in (1, 2):
                    out, parent_out = torch.empty_like(
                        inputs[0]
                    ), torch.empty_like(inputs[0])
                    fn(*inputs, out, stage)
                    parent(*inputs, parent_out, stage)
                    accuracy = torch.allclose(
                        out.cpu(), case["reference"].half(), atol=TOL, rtol=TOL
                    )
                    parent_accuracy = torch.allclose(
                        parent_out.cpu(),
                        case["reference"].half(),
                        atol=TOL,
                        rtol=TOL,
                    )
                    baseline_accuracy = torch.allclose(
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
                    assert (
                        replay
                        and accuracy
                        and parent_accuracy
                        and baseline_accuracy
                    )
                    # Both comparisons are paired and alternate their execution order.
                    pp = graph_pair(
                        lambda: fn(*inputs, out, stage),
                        lambda: parent(*inputs, parent_out, stage),
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
                    rows.append(
                        dict(
                            id=f"{name}/s{stage}/{case['id']}/round{round_id}",
                            function=name,
                            round=round_id,
                            stage=stage,
                            shape=list(out.shape),
                            input_sha256=case["input_sha256"],
                            accuracy_passed=accuracy,
                            parent_accuracy_passed=parent_accuracy,
                            baseline_accuracy_passed=baseline_accuracy,
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
            dict(round=round_id, before=before, after=telemetry(args.backend))
        )
        print(
            f"performance round {round_id + 1}/{args.rounds} complete",
            flush=True,
        )
    return rows


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
        parent = getattr(lib, PARENT)
        if args.frozen_baseline:
            baseline = json.loads(args.frozen_baseline.read_text())
            artifact = ROOT / baseline["isa"]["binary"]
            assert sha(artifact) == baseline["isa"]["binary_sha256"]
            spec = importlib.util.spec_from_file_location(
                Path(baseline["isa"]["build_cache_binary"]).stem, artifact
            )
            frozen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(frozen)
            parent = getattr(frozen, PARENT)
            builds["frozen_parent"] = baseline["builds"]["hip"]
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
            if not set(names) & (set(NAMES) | {PARENT}):
                continue
            assert set(names) <= set(NAMES) | {PARENT}
            lib, metadata = build(group, args)
            builds[group["source"]] = metadata
            functions.update({name: getattr(lib, name) for name in names})
        parent = functions.pop(PARENT)
        assert set(functions) == set(NAMES)
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        environment=identity,
        logical_device=args.device,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        helper_sha256={
            p: sha(ROOT / p)
            for p in (
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/cross_validate.py",
                "leetcuda_rocm/cuda_pending_check.py",
            )
        },
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
            for stage in (1, 2):
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
                parent_out = torch.empty_like(inputs[0])
                parent(*inputs, parent_out, stage)
                row["parent_bitwise_equal"] = torch.equal(
                    parent_out.cpu(), actual
                )
                if args.backend == "hip":
                    row["passed"] &= row["parent_bitwise_equal"]
                report["checks"].append(row)
                outputs[identifier] = actual
        print(name, "shared complete", flush=True)
    if args.performance:
        report["performance"] = performance(
            functions, parent, fixture, args, report
        )
    report["storage_checks"] = storage_checks(functions)
    if args.backend == "hip":
        report["boundaries"] = boundaries(lib)
        report["device_checks"] = devices(lib)
        report["layout_checks"] = layout_checks()
        report["isa"] = inspect_isa(
            lib, args.output.parent / ("isa_device" + str(args.device))
        )
    report["passed"] = all(
        c["passed"]
        for c in report["checks"]
        + report["boundaries"]
        + report["storage_checks"]
        + report.get("device_checks", [])
        + report.get("layout_checks", [])
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
                storage=len(report["storage_checks"]),
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
    parser.add_argument(
        "--names", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--frozen-baseline", type=Path)
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
    global NAMES
    NAMES = tuple(VARIANTS[n] for n in args.names)
    if args.command == "fixtures":
        fixtures(args.fixtures)
        return 0
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
