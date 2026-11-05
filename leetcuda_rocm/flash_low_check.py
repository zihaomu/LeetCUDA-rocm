"""B4 attention: explicit QK upgrades, original V layouts/state, AMD WMMA."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha
from .flash_shared_check import reference
from .flash_swizzle_check import devices, telemetry

CONTRACTS = json.loads(
    (Path(__file__).with_name("flash_low_contracts.json")).read_text()
)
TOL = 0.003


def transposed_v(name):
    return name.endswith("_swizzle_qkv") and "tiling_qkv_" not in name


def configure(spec, fn):
    def call(q, k, v, out, stage):
        args = () if spec["argc"] == 4 else (stage,)
        return fn(q, k, v, out, *args)

    return call


def inputs_for(name, tensors):
    q, k, v = tensors
    return (q, k, v.transpose(-1, -2).contiguous() if transposed_v(name) else v)


def geometry(name, d, stage):
    if name == "flash_attn_cute":
        br = bc = 128 if d <= 64 else 64 if d == 128 else 32
        return br, bc, "FP32"
    if transposed_v(name) and "tiling_qk_" in name:
        return (32 if d == 256 else d), d, "FP16" if d >= 256 else "FP32"
    if "tiling_" in name:
        br = bc = 64 if d < 128 else 128
    elif name.endswith("split_kv"):
        br = bc = 64
    elif name.endswith("split_q"):
        br = 64
        bc = 64 if d < 128 else 16
    else:
        br = 64 if d == 256 else 128
        bc = 32 if "_shared_kv" in name and d >= 128 else 64
    half = name.endswith(("split_q", "split_kv", "Os2g")) or d >= 256
    return br, bc, "FP16" if half else "FP32"


def numerical(fn, inputs, stage, identifier, ref, unaligned=False):
    # Canary storage also covers empty and offset-one output views.
    prefix = 17 if unaligned else 16
    raw = torch.full(
        (inputs[0].numel() + prefix + 16,),
        123.0,
        device=inputs[0].device,
        dtype=torch.float16,
    )
    out = raw[prefix:-16].view_as(inputs[0])
    out.fill_(float("nan"))
    before = [x.clone() for x in inputs]
    fn(*inputs, out, stage)
    actual = out.cpu()
    canary = bool((raw[:prefix] == 123).all() and (raw[-16:] == 123).all())
    unchanged = all(torch.equal(a, b) for a, b in zip(inputs, before))
    return actual, dict(
        id=identifier,
        stage=stage,
        shape=list(out.shape),
        passed=bool(torch.isfinite(actual).all())
        and torch.allclose(actual, ref.half().cpu(), atol=TOL, rtol=TOL)
        and canary
        and unchanged,
        canary_passed=canary,
        inputs_unchanged=unchanged,
        atol=TOL,
        rtol=TOL,
        max_abs_error_fp64=(
            (actual.double() - ref.cpu()).abs().max().item()
            if actual.numel()
            else 0
        ),
    )


def fixtures(path):
    from .flash_tiled_check import fixtures as create

    create(path)
    rows = torch.load(path, weights_only=True)
    g = torch.Generator().manual_seed(20261106)
    for n in (128, 256, 384, 1024):
        qkv = [
            (torch.randn(1, 2, n, 16, generator=g) * 0.5).half()
            for _ in range(3)
        ]
        rows.append(
            dict(
                id=f"cute_d16_n{n}",
                inputs=qkv,
                reference=reference(*qkv),
                input_sha256=[tensor_sha(t) for t in qkv],
                performance=n == 1024,
            )
        )
    torch.save(rows, path)


def boundaries(lib):
    rows = []
    for spec in CONTRACTS:
        name = spec["function"]
        fn = configure(spec, getattr(lib, name))
        for stage, dimensions in spec["dimensions"].items():
            stage = int(stage)
            for d in dimensions:
                for seed in (17, 93):
                    torch.manual_seed(seed)
                    for n in (1, 17, 63, 65, 127, 129, 257):
                        logical = [
                            (
                                torch.randn(1, 2, n, d, device="cuda") * 0.5
                            ).half()
                            for _ in range(3)
                        ]
                        _, row = numerical(
                            fn,
                            inputs_for(name, logical),
                            stage,
                            f"{name}/d{d}/s{stage}/n{n}/seed{seed}",
                            reference(*logical),
                        )
                        rows.append(row)
                for shape in (
                    (0, 2, 17, d),
                    (1, 0, 17, d),
                    (1, 2, 0, d),
                    (2, 3, 65, d),
                ):
                    logical = [
                        (torch.randn(shape, device="cuda") * 0.5).half()
                        for _ in range(3)
                    ]
                    _, row = numerical(
                        fn,
                        inputs_for(name, logical),
                        stage,
                        f"{name}/s{stage}/shape{shape}",
                        reference(*logical),
                    )
                    rows.append(row)
                logical = [
                    (torch.randn(1, 1, 65, d, device="cuda") * 0.5).half()
                    for _ in range(3)
                ]
                packed = inputs_for(name, logical)
                offset = []
                for x in packed:
                    raw = torch.empty(
                        x.numel() + 1, device="cuda", dtype=x.dtype
                    )
                    t = raw[1:].view_as(x)
                    t.copy_(x)
                    offset.append(t)
                _, row = numerical(
                    fn,
                    offset,
                    stage,
                    f"{name}/d{d}/s{stage}/offset-one",
                    reference(*logical),
                    True,
                )
                rows.append(row)
        logical = [
            (torch.randn(1, 2, 65, 64, device="cuda") * 0.5).half()
            for _ in range(3)
        ]
        q, k, v = inputs_for(name, logical)
        out = torch.empty_like(q)
        if spec["argc"] == 5:
            for stage in (-3, 0, 3, 5):
                _, row = numerical(
                    fn,
                    (q, k, v),
                    stage,
                    f"{name}/alias{stage}",
                    reference(*logical),
                )
                rows.append(row)
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
        }

        def reject_shape(shape, stage):
            logical = [
                torch.empty(shape, device="cuda", dtype=torch.float16)
                for _ in range(3)
            ]
            return fn(
                *inputs_for(name, logical), torch.empty_like(logical[0]), stage
            )

        invalid["grid-limit"] = lambda: reject_shape((65536, 1, 0, 64), 1)
        invalid["index-limit"] = lambda: reject_shape((0, 1, 2147483647, 64), 1)
        invalid["head-192"] = lambda: reject_shape((1, 1, 65, 192), 1)
        if name != "flash_attn_cute":
            invalid["head-16"] = lambda: reject_shape((1, 1, 65, 16), 1)
        if transposed_v(name):
            invalid["V-layout"] = lambda: fn(q, k, logical[2], out, 1)
        for stage, ds in spec["dimensions"].items():
            for d in (96, 256, 512, 1024):
                if d not in ds:
                    try:
                        reject_shape((1, 1, 65, d), int(stage))
                    except RuntimeError:
                        passed = True
                    else:
                        passed = False
                    rows.append(
                        dict(id=f"{name}/reject-d{d}-s{stage}", passed=passed)
                    )
        for label, call in invalid.items():
            try:
                call()
            except RuntimeError:
                passed = True
            else:
                passed = False
            rows.append(dict(id=name + "/reject-" + label, passed=passed))
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
                args = inputs_for(name, (q, k, v))
                for _ in range(4):
                    fn(*args, out, 1 + i % 2)
                queued.append((stream, args, out, i))
        for stream, args, out, i in queued:
            stream.synchronize()
            rows.append(
                dict(
                    id=f"{name}/stream{i}",
                    passed=bool((out == (i + 1) * 0.125).all()),
                )
            )
        print(
            name,
            "boundaries",
            sum(not c["passed"] for c in rows),
            "failures",
            flush=True,
        )
    return rows


def storage_checks(lib):
    rows = []
    for spec in CONTRACTS:
        name = spec["function"]
        fn = configure(spec, getattr(lib, name))
        for stage, ds in spec["dimensions"].items():
            stage = int(stage)
            for d in ds:
                _, bc, state_type = geometry(name, d, stage)
                n = bc * 32
                q = torch.zeros(1, 1, n, d, device="cuda", dtype=torch.float16)
                k = torch.zeros_like(q)
                v = torch.full_like(q, 2.0**-11)
                v[:, :, :bc] = 1
                value = torch.tensor(0.0)
                opposite = torch.tensor(0.0)
                for block in range(32):
                    add = bc if block == 0 else bc * 2.0**-11
                    value = value + add
                    opposite = opposite + add
                    if state_type == "FP16":
                        value = value.half().float()
                    else:
                        opposite = opposite.half().float()
                expected = (value / n).half().item()
                wrong = (opposite / n).half().item()
                out = torch.empty_like(q)
                fn(*inputs_for(name, (q, k, v)), out, stage)
                rows.append(
                    dict(
                        id=f"{name}/storage/d{d}/s{stage}",
                        output_state=state_type,
                        expected=expected,
                        opposite_state_result=wrong,
                        passed=expected != wrong
                        and bool((out == expected).all()),
                    )
                )
    return rows


def inspect_isa(lib, destination):
    from .isa_objects import hip_isa_objects
    import yaml

    isa = hip_isa_objects(lib.__file__, destination.resolve())
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n",
        (ROOT / isa["assembly"]).read_text(),
        flags=re.M,
    )
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
        document = notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
        for entry in yaml.safe_load(document)["amdhsa.kernels"]:
            metadata[entry[".name"]] = entry
    kernels = []
    for i in range(1, len(sections), 3):
        symbol, body = sections[i + 1 : i + 3]
        shared = re.search(
            r"attention_low_sharedILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", symbol
        )
        tiled = re.search(
            r"attention_low_tiledILi(\d+)ELi(\d+)ELb([01])ELi(\d+)E", symbol
        )
        split = re.search(r"attention_split_kvILi(\d+)ELi(\d+)E", symbol)
        if shared:
            d, stage, kind, mask = map(int, shared.groups())
            br = (
                (128 if d <= 64 else 64 if d == 128 else 32)
                if kind == 4
                else 64 if kind == 0 or d == 256 else 128
            )
            bc = (
                br
                if kind == 4
                else (
                    (64 if d < 128 else 16)
                    if kind == 0
                    else 32 if kind == 1 and d >= 128 else 64
                )
            )
            q = br * d
            kv = stage * bc * d
            data = (
                max(q, kv)
                if kind >= 2
                else q + kv + (bc * d if kind == 0 else 0)
            )
            lds = (data + br * bc) * 2
            family = "shared"
            extra = dict(kind=kind, swizzle_mask=mask)
            fp32pv = kind == 4
        elif tiled:
            d, stage, tile_v, mask = map(int, tiled.groups())
            transposed = not tile_v and mask == 7
            br = (
                (32 if d == 256 else d)
                if transposed
                else 64 if d < 128 else 128
            )
            bc = d if transposed else br
            od = min(d, 128)
            data = stage * max((br + bc) * 16, bc * 16 if tile_v else 16 * od)
            lds = (data + br * bc) * 2
            family = "tiled"
            extra = dict(tile_v=bool(tile_v), swizzle_mask=mask)
            fp32pv = False
        elif split:
            d, stage = map(int, split.groups())
            br = bc = 64
            lds = (br * d + stage * bc * d + br * bc) * 2 + 2 * br * 4 * 4
            family = "split_kv"
            extra = {}
            fp32pv = False
        else:
            continue
        e = metadata[symbol]
        row = dict(
            symbol=symbol,
            family=family,
            d=d,
            stage=stage,
            query_tile=br,
            key_tile=bc,
            dynamic_lds_bytes=lds,
            fp32_wmma=body.count("v_wmma_f32_16x16x16_f16"),
            fp16_wmma=body.count("v_wmma_f16_16x16x16_f16"),
            scratch=body.count("scratch_"),
            code_sha256=hashlib.sha256(body.encode()).hexdigest(),
            **extra,
        )
        row.update(
            {
                key[1:]: e[key]
                for key in (
                    ".vgpr_count",
                    ".sgpr_count",
                    ".vgpr_spill_count",
                    ".private_segment_fixed_size",
                )
            }
        )
        row["passed"] = (
            lds <= 65536
            and row["fp32_wmma"] > 0
            and (fp32pv or row["fp16_wmma"] > 0)
        )
        kernels.append(row)
    assert len(kernels) == 210, len(kernels)
    isa["kernels"] = kernels
    isa["passed"] = all(k["passed"] for k in kernels)
    return isa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--create-fixtures", action="store_true")
    parser.add_argument("--no-performance", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.create_fixtures:
        fixtures(args.fixtures)
        return 0
    from .environment import verify
    from .loader import load, source_paths

    identity = verify()
    torch.cuda.set_device(args.device)
    lib = load("flash_attn")
    report = dict(
        environment=identity,
        device=args.device,
        fixture_sha256=sha(args.fixtures),
        sources={
            str(p.relative_to(ROOT)): sha(p) for p in source_paths("flash_attn")
        },
        harness_sha256=sha(__file__),
        contracts_sha256=sha(
            Path(__file__).with_name("flash_low_contracts.json")
        ),
        binary_sha256=sha(lib.__file__),
        checks=[],
        performance=[],
        telemetry_before=telemetry("hip"),
    )
    values = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for spec in CONTRACTS:
        name = spec["function"]
        fn = configure(spec, getattr(lib, name))
        for c in torch.load(args.fixtures, weights_only=True):
            logical = [t.cuda() for t in c["inputs"]]
            packed = inputs_for(name, logical)
            d = logical[0].shape[-1]
            for stage, ds in spec["dimensions"].items():
                if d not in ds:
                    continue
                stage = int(stage)
                identifier = f"{name}/s{stage}/{c['id']}"
                actual, row = numerical(
                    fn, packed, stage, identifier, c["reference"]
                )
                row.update(
                    function=name,
                    input_sha256=c["input_sha256"],
                    packed_input_sha256=[tensor_sha(t.cpu()) for t in packed],
                )
                report["checks"].append(row)
                values[identifier] = actual
                if not row["passed"]:
                    print("FAIL", row, flush=True)
                if c["performance"] and not args.no_performance:
                    out = torch.empty_like(logical[0])
                    fn(*packed, out, stage)
                    old = out.clone()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        fn(*packed, out, stage)
                    with torch.cuda.stream(stream):
                        out.fill_(float("nan"))
                        graph.replay()
                    stream.synchronize()
                    replay = torch.equal(out, old)
                    perf = graph_pair(
                        lambda: fn(*packed, out, stage),
                        lambda: F.scaled_dot_product_attention(*logical),
                        calls=8,
                        sample_ms=10,
                        trials=3,
                    )
                    report["performance"].append(
                        dict(
                            id=identifier,
                            function=name,
                            stage=stage,
                            shape=list(out.shape),
                            graph_replay_verified=replay,
                            accuracy_passed=row["passed"],
                            **perf,
                        )
                    )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, "shared done", flush=True)
    report["isa"] = inspect_isa(
        lib, args.output.parent / (args.output.stem + "_isa")
    )
    report["boundaries"] = boundaries(lib)
    report["storage"] = storage_checks(lib)
    report["devices"] = devices(lib)
    torch.save(values, args.output.with_suffix(".pt"))
    report["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    report["telemetry_after"] = telemetry("hip")
    report["passed"] = (
        report["isa"]["passed"]
        and all(
            row["passed"]
            for key in ("checks", "boundaries", "storage", "devices")
            for row in report[key]
        )
        and all(row["graph_replay_verified"] for row in report["performance"])
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "DONE",
        report["passed"],
        {
            key: len(report[key])
            for key in (
                "checks",
                "boundaries",
                "storage",
                "devices",
                "performance",
            )
        },
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
