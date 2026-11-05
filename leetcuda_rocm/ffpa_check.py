"""FFPA independent FP64, pipeline audit, runtime, ISA and CUDA comparison."""

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .cross_validate import CUTLASS, UPSTREAM, sha, tensor_sha
from .environment import ROOT
from .ffpa import HEAD_DIMS, VARIANTS

STAGES = ((1, 1), (2, 2), (2, 3), (3, 2), (3, 3))
ATOL = RTOL = 0.003
LSE_ATOL = LSE_RTOL = 2e-5


def reference(q, k, v):
    scores = q.double() @ k.double().transpose(-1, -2) / q.shape[-1] ** 0.5
    if k.shape[-2] == 0:
        return torch.zeros_like(q, dtype=torch.float64), torch.full(
            q.shape[:-1], -torch.inf, dtype=torch.float64
        )
    return scores.softmax(-1) @ v.double(), scores.logsumexp(-1)


def fixtures(args):
    torch.set_num_threads(8)
    cases = []

    def add(label, d, nq, nk, seed, pattern=None, b=1, h=2):
        gen = torch.Generator().manual_seed(seed)
        q, k, v = [
            torch.randn(shape, generator=gen).half()
            for shape in ((b, h, nq, d), (b, h, nk, d), (b, h, nk, d))
        ]
        if pattern == "uniform":
            q.zero_()
            k.zero_()
        if pattern == "zero_v":
            v.zero_()
        if pattern == "large":
            q.mul_(4)
            k.mul_(4)
        if pattern == "cancel":
            v[..., 1::2, :] = -v[..., ::2, :]
        if pattern == "rescale":
            q.zero_()
            q[..., 0] = 1
            k.zero_()
            for block, log2 in enumerate((0, 4, 16, 18, 32, 33)):
                k[..., block * 64 : (block + 1) * 64, 0] = (
                    log2 * d**0.5 * 0.6931471805599453
                )
        expected, lse = reference(q, k, v)
        cases.append(
            dict(
                id=label,
                q=q,
                k=k,
                v=v,
                expected=expected,
                lse=lse,
                input_sha256={
                    name: tensor_sha(x)
                    for name, x in zip(("q", "k", "v"), (q, k, v))
                },
                variants=list(VARIANTS) if nq == nk else ["non_ws"],
            )
        )

    for d in HEAD_DIMS:
        for seed in (17, 93):
            for n in (128, 256, 384):
                add(f"random/{d}/{seed}/{n}", d, n, n, seed)
            add(f"unequal/{d}/{seed}", d, 128, 193, seed)
    for d in (64, 192, 512, 1024):
        for pattern in ("uniform", "zero_v", "large", "cancel", "rescale"):
            add(f"{pattern}/{d}", d, 384, 384, 20261106, pattern)
        add(f"long/{d}", d, 128, 4096, 97, b=1, h=1)
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cases=cases, generator_sha256=sha(__file__)), args.fixtures)
    print(
        json.dumps(dict(cases=len(cases), sha256=sha(args.fixtures))),
        flush=True,
    )


def cuda_build(d, original, cutlass, tail_fix=False):
    from torch.utils.cpp_extension import load

    binding = (
        ROOT / "leetcuda_rocm/native/ffpa_cuda_reference.cpp"
    ).read_text()
    include = original / "kernels/interview"
    headers = {
        str(p.relative_to(original)): sha(p)
        for p in sorted(include.glob("*.cuh"))
    }
    header = (include / "ffpa_attn.cuh").read_text()
    if tail_fix:
        changes = {
            "      // QK GEMM: S += Q[d_chunk] @ K[d_chunk]^T": "      if (d_next >= kDChunks) { cp_async_wait<0>(); __syncthreads(); }\n      // QK GEMM: S += Q[d_chunk] @ K[d_chunk]^T",
            "      // Rescale O accumulator（与 TMA WS kernel 完全一致）": "      if (v_next >= kDChunks) { cp_async_wait<0>(); __syncthreads(); }\n      // Rescale O accumulator（与 TMA WS kernel 完全一致）",
        }
        for old, new in changes.items():
            assert header.count(old) == 1
            header = header.replace(old, new)
        binding = binding.replace(
            '#include "ffpa_attn.cuh"', '#include "ffpa_reference.cuh"'
        )
    flags = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-DNOTES_V2_ENABLE_CUTE=1",
        "-DNOTES_V2_ENABLE_TMA_MMA_WS=1",
        f"-DFFPA_HEAD_DIM={d}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    digest = hashlib.sha256(
        (
            binding
            + (header if tail_fix else "")
            + json.dumps([flags, headers, CUTLASS], sort_keys=True)
        ).encode()
    ).hexdigest()[:16]
    build = ROOT / ".cache/cuda_comparison" / f"ffpa_{d}_{digest}"
    build.mkdir(parents=True, exist_ok=True)
    if tail_fix:
        staged = build / "ffpa_reference.cuh"
        if not staged.exists() or staged.read_text() != header:
            staged.write_text(header)
    source = build / "binding.cu"
    if not source.exists() or source.read_text() != binding:
        source.write_text(binding)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    os.environ.setdefault("MAX_JOBS", "2")
    lib = load(
        name=f"leetcuda_ffpa_cuda_{d}_{digest}",
        sources=[str(source)],
        build_directory=str(build),
        extra_cuda_cflags=flags,
        extra_cflags=["-O3", "-std=c++20"],
        extra_include_paths=[str(cutlass / "include"), str(include)],
        extra_ldflags=["-lcuda"],
    )
    return lib, dict(
        source_kind=(
            "explicit_async_tail_drain_fix" if tail_fix else "upstream_original"
        ),
        reference_header_sha256=hashlib.sha256(header.encode()).hexdigest(),
        upstream_commit=UPSTREAM,
        cutlass_commit=CUTLASS,
        headers_sha256=headers,
        binding_sha256=sha(source),
        binary_sha256=sha(lib.__file__),
        compiler_flags=flags,
    )


def accurate(actual, expected, lse=False):
    actual = actual.cpu().double()
    expected = expected.double()
    passed = torch.allclose(
        actual,
        expected,
        atol=LSE_ATOL if lse else ATOL,
        rtol=LSE_RTOL if lse else RTOL,
    )
    finite = torch.isfinite(expected)
    error = (
        (actual[finite] - expected[finite]).abs().max().item()
        if finite.any()
        else 0.0
    )
    return dict(passed=passed, max_abs=error)


def isa(lib, folder):
    import yaml
    from .isa_objects import hip_isa_objects

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
        if "leetcuda_ffpa" not in symbol or "kernel" not in symbol:
            continue
        e = metadata[symbol]
        kernels.append(
            dict(
                symbol=symbol,
                wmma=body.count("v_wmma_f32_16x16x16_f16"),
                load128=body.count("global_load_b128"),
                wait_load=body.count("s_wait_loadcnt"),
                vgpr=e[".vgpr_count"],
                spill=e[".vgpr_spill_count"],
                scratch=e[".private_segment_fixed_size"],
                lds=e[".group_segment_fixed_size"],
            )
        )
    report.update(
        kernels=kernels,
        passed=len(kernels) == 27
        and all(
            k["wmma"] and k["load128"] and k["wait_load"] and k["lds"] <= 65536
            for k in kernels
        ),
    )
    return report


def extras(lib, device):
    from .ffpa import ffpa_attention as api

    checks = []
    gen = torch.Generator().manual_seed(20261106)

    def inputs(d, nq, nk, b=1, h=1, unaligned=False):
        cpu = [
            torch.randn(shape, generator=gen).half()
            for shape in ((b, h, nq, d), (b, h, nk, d), (b, h, nk, d))
        ]
        xs = []
        for x in cpu:
            if unaligned:
                base = torch.empty(x.numel() + 1, device=device, dtype=x.dtype)
                y = base[1:].view_as(x)
                y.copy_(x)
            else:
                y = x.to(device)
            xs.append(y)
        return cpu, xs

    for d in HEAD_DIMS:
        for nq, nk in (
            (1, 1),
            (17, 79),
            (63, 65),
            (129, 257),
            (0, 64),
            (16, 0),
        ):
            cpu, xs = inputs(d, nq, nk, unaligned=True)
            expected, lr = reference(*cpu)
            for kind, (sq, sv) in itertools.product(
                range(3), itertools.product((1, 2, 3), repeat=2)
            ):
                storage = torch.full(
                    (xs[0].numel() + 2,),
                    91.0,
                    device=device,
                    dtype=torch.float16,
                )
                o = storage[1:-1].view_as(xs[0])
                o.fill_(float("nan"))
                lse = torch.full(xs[0].shape[:-1], float("nan"), device=device)
                lib.forward(*xs, o, lse, None, kind, sq, sv, 0)
                a = accurate(o, expected)
                l = accurate(lse, lr, True)
                checks.append(
                    dict(
                        id=f"boundary/{d}/{nq},{nk}/{kind}/{sq},{sv}",
                        passed=a["passed"]
                        and l["passed"]
                        and storage[0].item() == 91
                        and storage[-1].item() == 91,
                        output=a,
                        lse=l,
                    )
                )
        print(
            "boundaries",
            d,
            "failed",
            sum(not x["passed"] for x in checks),
            flush=True,
        )
    # Producer/consumer counters plus delayed producer and delayed consumer.
    for delay, (sq, sv) in itertools.product(
        (0, 1, 2), itertools.product((1, 2, 3), repeat=2)
    ):
        cpu, xs = inputs(512, 16, 192)
        expected, lr = reference(*cpu)
        o = torch.empty_like(xs[0])
        lse = torch.empty(xs[0].shape[:-1], device=device)
        stats = torch.zeros((1, 1, 8), device=device, dtype=torch.int32)
        lib.forward(*xs, o, lse, stats, 1, sq, sv, delay)
        a = accurate(o, expected)
        s = stats.cpu().flatten().tolist()
        passed = (
            a["passed"]
            and s[:4] == [24] * 4
            and 0 < s[4] <= sq
            and 0 < s[5] <= sv
            and s[7] == 0
        )
        if delay == 2:
            passed &= s[4] == sq and s[5] == sv
        checks.append(
            dict(
                id=f"pipeline/{delay}/{sq},{sv}",
                passed=bool(passed),
                audit=s,
                output=a,
            )
        )
    # Explicit FA4 retained-max and rescaled-max branches.
    cpu, xs = inputs(64, 16, 384)
    cpu[0].zero_()
    cpu[0][..., 0] = 1
    cpu[1].zero_()
    for block, log2 in enumerate((0, 4, 16, 18, 32, 33)):
        cpu[1][..., block * 64 : (block + 1) * 64, 0] = (
            log2 * 8 * 0.6931471805599453
        )
    xs = [x.to(device) for x in cpu]
    expected, lr = reference(*cpu)
    o = torch.empty_like(xs[0])
    lse = torch.empty(xs[0].shape[:-1], device=device)
    stats = torch.zeros((1, 1, 8), device=device, dtype=torch.int32)
    lib.forward(*xs, o, lse, stats, 2, 2, 3, 0)
    a = accurate(o, expected)
    l = accurate(lse, lr, True)
    s = stats.cpu().flatten().tolist()
    checks.append(
        dict(
            id="conditional_rescale_both_branches",
            passed=a["passed"] and l["passed"] and s[6] == 3,
            audit=s,
            output=a,
            lse=l,
        )
    )
    # Empty batch/head, output storage, rejects and public API routing.
    for shape in ((0, 2, 17, 64), (2, 0, 17, 64)):
        xs = [
            torch.empty(shape, device=device, dtype=torch.float16)
            for _ in range(3)
        ]
        for variant in VARIANTS:
            o, l = api(*xs, variant=variant, return_lse=True)
            checks.append(
                dict(
                    id=f"empty/{shape}/{variant}",
                    passed=o.numel() == 0 and l.numel() == 0,
                )
            )
    cpu, xs = inputs(192, 17, 79)
    expected, lr = reference(*cpu)
    for variant in VARIANTS:
        o, l = api(*xs, variant=variant, return_lse=True)
        checks.append(
            dict(
                id=f"api/{variant}",
                passed=accurate(o, expected)["passed"]
                and accurate(l, lr, True)["passed"],
            )
        )
    bad = {
        "cpu": lambda: api(*cpu),
        "dtype": lambda: api(*[x.float() for x in xs]),
        "rank": lambda: api(*[x[0] for x in xs]),
        "shape": lambda: api(xs[0], xs[1], xs[2][..., :-1, :]),
        "stride": lambda: api(*[x.transpose(-1, -2) for x in xs]),
        "grad": lambda: api(xs[0].detach().requires_grad_(), xs[1], xs[2]),
        "d": lambda: api(
            *[
                torch.empty((1, 1, 16, 96), device=device, dtype=torch.float16)
                for _ in range(3)
            ]
        ),
        "variant": lambda: api(*xs, variant="tma"),
        "stage0": lambda: api(*xs, stages_qk=0),
        "stage4": lambda: api(*xs, stages_v=4),
        "stage_float": lambda: api(*xs, stages_v=2.0),
        "stage_bool": lambda: api(*xs, stages_qk=True),
        "alias": lambda: api(*xs, out=xs[0]),
        "out_dtype": lambda: api(
            *xs, out=torch.empty_like(xs[0], dtype=torch.float32)
        ),
        "lse_shape": lambda: api(*xs, lse=torch.empty(1, device=device)),
        "out_grad": lambda: api(
            *xs, out=torch.empty_like(xs[0], requires_grad=True)
        ),
    }
    for name, fn in bad.items():
        try:
            fn()
            ok = False
        except (ValueError, RuntimeError):
            ok = True
        checks.append(dict(id="reject/" + name, passed=ok))
    streams = [torch.cuda.Stream(device=device) for _ in range(6)]
    outputs = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for variant in VARIANTS:
                outputs.append(api(*xs, variant=variant, return_lse=True))
    for stream in streams:
        stream.synchronize()
    for i, (o, l) in enumerate(outputs):
        checks.append(
            dict(
                id=f"stream/{i}",
                passed=accurate(o, expected)["passed"]
                and accurate(l, lr, True)["passed"],
            )
        )
    for variant in VARIANTS:
        o = torch.empty_like(xs[0])
        l = torch.empty(xs[0].shape[:-1], device=device)
        stream = streams[0]
        with torch.cuda.stream(stream):
            api(*xs, variant=variant, out=o, lse=l)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                api(*xs, variant=variant, out=o, lse=l)
            o.fill_(float("nan"))
            l.fill_(float("nan"))
            graph.replay()
            stream.synchronize()
        checks.append(
            dict(
                id="graph/" + variant,
                passed=accurate(o, expected)["passed"]
                and accurate(l, lr, True)["passed"],
            )
        )
    if torch.cuda.device_count() > 1:
        other = 1 - device
        ys = [x.to(other) for x in cpu]
        for variant in VARIANTS:
            with torch.cuda.device(other):
                o = api(*xs, variant=variant)
                guard = torch.cuda.current_device() == other
            try:
                api(xs[0], ys[1], xs[2], variant=variant)
                reject = False
            except ValueError:
                reject = True
            checks.append(
                dict(
                    id="device/" + variant,
                    passed=guard and reject and accurate(o, expected)["passed"],
                )
            )
    return checks


def run(args):
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    fixture = torch.load(args.fixtures, map_location="cpu", weights_only=True)
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        checks=[],
        builds={},
        passed=False,
        gate=dict(atol=ATOL, rtol=RTOL, lse_atol=LSE_ATOL, lse_rtol=LSE_RTOL),
    )
    if args.backend == "hip":
        from .environment import verify
        from .ffpa import _native

        report["environment"] = verify()
        lib = _native(args.device)
    else:
        report["environment"] = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )
    outputs = {}
    stream = torch.cuda.Stream()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for d in args.dims:
        if args.backend == "cuda":
            lib, report["builds"][d] = cuda_build(
                d, args.original, args.cutlass, args.cuda_tail_fix
            )
        for c in fixture["cases"]:
            if c["q"].shape[-1] != d:
                continue
            xs = [c[key].to(args.device) for key in ("q", "k", "v")]
            stream.wait_stream(torch.cuda.current_stream())
            for variant, (sq, sv) in itertools.product(c["variants"], STAGES):
                name = f"{c['id']}/{variant}/{sq},{sv}"
                o = torch.full_like(xs[0], float("nan"))
                l = torch.full(
                    xs[0].shape[:-1], float("nan"), device=args.device
                )
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    if args.backend == "hip":
                        lib.forward(
                            *xs, o, l, None, VARIANTS[variant], sq, sv, 0
                        )
                    else:
                        lib.forward(*xs, o, l, VARIANTS[variant], sq, sv)
                stream.synchronize()
                a = accurate(o, c["expected"])
                le = (
                    accurate(l, c["lse"], True)
                    if args.backend == "hip" or variant == "non_ws"
                    else None
                )
                row = dict(
                    id=name,
                    passed=a["passed"] and (le is None or le["passed"]),
                    output=a,
                    lse=le,
                )
                report["checks"].append(row)
                outputs[name] = dict(
                    o=o.cpu(), lse=l.cpu() if le is not None else None
                )
                if not row["passed"]:
                    print("FAILED", row, flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save(outputs, args.output.with_suffix(".pt"))
        print(
            "D",
            d,
            "checks",
            len(report["checks"]),
            "failed",
            sum(not c["passed"] for c in report["checks"]),
            flush=True,
        )
    if args.backend == "hip":
        if args.extras:
            report["checks"] += extras(lib, args.device)
        report["isa"] = isa(
            lib, args.output.parent / (args.output.stem + "_isa")
        )
        report["sources_sha256"] = {
            p: sha(ROOT / p)
            for p in (
                "kernels/rocm/ffpa_attention_hip.cuh",
                "leetcuda_rocm/native/ffpa_bindings.cpp",
                "leetcuda_rocm/ffpa.py",
            )
        }
    report["passed"] = all(
        c["passed"] for c in report["checks"]
    ) and report.get("isa", {}).get("passed", True)
    report["output_sha256"] = sha(args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", report["passed"], len(report["checks"]), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


def compare(args):
    a = torch.load(args.hip, map_location="cpu", weights_only=True)
    b = torch.load(args.cuda, map_location="cpu", weights_only=True)
    checks = []
    for key, values in b.items():
        for part, x in values.items():
            if x is None:
                continue
            row = accurate(a[key][part], x, part == "lse")
            checks.append(
                dict(
                    id=key + "/" + part,
                    **row,
                    exact=torch.equal(a[key][part], x),
                )
            )
    result = dict(
        passed=all(x["passed"] for x in checks),
        checks=checks,
        hip_sha256=sha(args.hip),
        cuda_sha256=sha(args.cuda),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("COMPARE", result["passed"], len(checks))
    if not result["passed"]:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fixtures")
    f.add_argument("--fixtures", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=("hip", "cuda"), required=True)
    r.add_argument("--fixtures", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--extras", action="store_true")
    r.add_argument("--cuda-tail-fix", action="store_true")
    r.add_argument("--dims", type=int, nargs="+", default=HEAD_DIMS)
    r.add_argument("--original", type=Path, default=ROOT / "original")
    r.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    c = sub.add_parser("compare")
    c.add_argument("--hip", type=Path, required=True)
    c.add_argument("--cuda", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
