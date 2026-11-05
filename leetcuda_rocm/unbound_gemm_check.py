"""Qualify the three GEMMs omitted from upstream pybind lists.

Original CUDA TN and an explicit accumulator-only FP32 reference are separate.
WGMMA is a labelled cuBLAS mathematical reference on sm120, not a CUDA port.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import torch

from .environment import ROOT
from .cross_validate import sha, tensor_sha, UPSTREAM
from .unbound import GEMM_ENTRIES
from .benchmark import graph_pair

TN_SOURCE = "kernels/hgemm/mma/swizzle/hgemm_mma_stage_tn_swizzle.cu"
SHAPES = (
    (128, 128, 64),
    (128, 256, 192),
    (256, 128, 512),
    (128, 256, 1024),
    (128, 256, 2048),
    (128, 2304, 64),
)


def fp32_reference(source):
    """Only widen original TN MMA accumulator; preserve copy/layout/schedule."""
    start = source.index("  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];")
    end = source.index(
        "  {\n    for (int i = 0; i < WARP_TILE_M; ++i) {", start
    )
    block = source[start:end].replace("RC[", "RF32[")
    block = block.replace(
        "RF32[WARP_TILE_M][WARP_TILE_N][2];",
        "RF32[WARP_TILE_M][WARP_TILE_N][4];",
    )
    block = block.replace(
        "RF32[i][j][1] = 0;",
        "RF32[i][j][1] = 0; RF32[i][j][2] = 0; RF32[i][j][3] = 0;",
    )

    def mma(m):
        a = [x.strip() for x in m[1].split(",")]
        assert len(a) == 10
        return (
            "HMMA16816_F32("
            + ", ".join(
                a[:2]
                + ["RF32[i][j][2]", "RF32[i][j][3]"]
                + a[2:8]
                + a[8:]
                + ["RF32[i][j][2]", "RF32[i][j][3]"]
            )
            + ");"
        )

    block, count = re.subn(r"HMMA16816\((.*?)\);", mma, block, flags=re.S)
    assert count == 2
    pack = """
  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  for(int i=0;i<WARP_TILE_M;i++) for(int j=0;j<WARP_TILE_N;j++)
    for(int x=0;x<2;x++) RC[i][j][x]=uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(RF32[i][j][2*x])))) | (uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(RF32[i][j][2*x+1]))))<<16);
"""
    macro = """
#define HMMA16816_F32(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1,c0,c1,c2,c3) \\
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(c0),"r"(c1),"r"(c2),"r"(c3))
"""
    return macro + source[:start] + block + pack + source[end:]


def cuda_build(args):
    from .cuda_pending_check import build

    original = (args.original / TN_SOURCE).read_text()
    staged_root = ROOT / ".cache/unbound_reference" / args.precision
    p = staged_root / TN_SOURCE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        fp32_reference(original) if args.precision == "fp32" else original
    )
    name = list(GEMM_ENTRIES)[2]
    group = dict(
        source=TN_SOURCE,
        module="hgemm",
        functions=[
            dict(
                name=name,
                declaration=f"void {name}(torch::Tensor,torch::Tensor,torch::Tensor,int,bool,int);",
                argc=6,
            )
        ],
    )
    args.source_root = staged_root
    args.cuda_current_stream = True
    lib, meta = build(group, args)
    meta.update(
        original_source_sha256=sha(args.original / TN_SOURCE),
        staged_reference_sha256=sha(p),
        upstream_commit=UPSTREAM,
    )
    return lib, meta


def fixture(path, m, n, k, offset=0):
    raw = torch.frombuffer(
        bytearray(Path(path).read_bytes()), dtype=torch.float32
    )
    a = (
        (raw[(torch.arange(m * k) + offset) % len(raw)] * 0.5)
        .half()
        .reshape(m, k)
    )
    bt = (
        (raw[len(raw) - 1 - (torch.arange(n * k) + offset) % len(raw)] * 0.5)
        .half()
        .reshape(n, k)
    )
    return a, bt


def accurate(x, ref):
    error = (x.double() - ref.double()).abs()
    return dict(
        passed=bool(torch.isfinite(x).all() and error.max() <= 0.01),
        max_abs=float(error.max()),
        atol=0.01,
        rtol=0,
    )


def run(args):
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if args.backend == "hip":
        from .unbound import load_unbound, source_paths
        from .environment import verify

        lib = load_unbound()
        identity = verify()
        meta = dict(
            source_sha256={
                str(p.relative_to(ROOT)): sha(p) for p in source_paths()
            },
            binary=lib.__file__,
            binary_sha256=sha(lib.__file__),
        )
        funcs = {name: getattr(lib, name) for name in GEMM_ENTRIES}
    else:
        lib, meta = cuda_build(args)
        identity = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )

        def math_reference(a, b, c, s, sw, stride):
            torch.mm(a, b.view(b.size(1), b.size(0)).t(), out=c)

        funcs = {
            name: (getattr(lib, name) if kind == 5 else math_reference)
            for name, kind in GEMM_ENTRIES.items()
        }
    r = dict(
        backend=args.backend,
        precision=args.precision,
        environment=identity,
        build=meta,
        fixture_sha256=sha(args.fixtures),
        harness_sha256=sha(__file__),
        helpers_sha256={
            p: sha(ROOT / p)
            for p in (
                "leetcuda_rocm/unbound.py",
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/cuda_pending_check.py",
            )
        },
        reference_kind={
            n: (
                (
                    "original_cuda_tn"
                    if args.precision == "original"
                    else "explicit_fp32_cuda_tn"
                )
                if k == 5
                else "cublas_fp32_math_not_original_wgmma"
            )
            for n, k in GEMM_ENTRIES.items()
        },
        checks=[],
        boundaries=[],
        pipeline=[],
        state_controls=[],
        performance=[],
    )
    outputs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def checkpoint():
        args.output.write_text(json.dumps(r, indent=2) + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def execute(
        key, fn, ac, btc, ref, stage, sw, stride, misalign=0, keep=True
    ):
        m, k = ac.shape
        n = btc.shape[0]
        with torch.cuda.stream(stream):
            aa = torch.empty(
                ac.numel() + misalign, device="cuda", dtype=torch.float16
            )
            bb = torch.empty(
                btc.numel() + misalign, device="cuda", dtype=torch.float16
            )
            a = aa[misalign:].view(m, k)
            b = bb[misalign:].view(k, n)
            a.copy_(ac)
            b.copy_(btc.view(k, n))
            backing = torch.full(
                (m * n + 16,), -123, device="cuda", dtype=torch.float16
            )
            c = backing[8:-8].view(m, n)
            c.fill_(float("nan"))
            fn(a, b, c, stage, sw, stride)
            stream.synchronize()
            got = c.cpu()
            canary = bool(
                (backing[:8] == -123).all() and (backing[-8:] == -123).all()
            )
            r["checks"].append(
                dict(
                    id=key,
                    **accurate(got, ref),
                    canary=canary,
                    input_sha256=[tensor_sha(ac), tensor_sha(btc)],
                )
            )
            if keep:
                outputs[key] = got
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                fn(a, b, c, stage, sw, stride)
            c.fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            result = c.cpu()
            r["checks"].append(
                dict(
                    id=key + "/graph",
                    **accurate(result, ref),
                    canary=bool(
                        (backing[:8] == -123).all()
                        and (backing[-8:] == -123).all()
                    ),
                    same_as_eager=torch.equal(result, got),
                )
            )
            if keep:
                outputs[key + "/graph"] = result

    cases = []
    for offset in (0, 8192):
        for m, n, k in SHAPES:
            a, bt = fixture(args.fixtures, m, n, k, offset)
            ref = (a.double() @ bt.double().t()).half()
            cases.append((offset, m, n, k, a, bt, ref))
    for name, fn in funcs.items():
        for stage in (2, 3, 4, 5):
            for offset, m, n, k, a, bt, ref in cases:
                options = (
                    ((False, 2048), (True, 256), (True, 512), (True, 2048))
                    if n == 2304
                    else ((False, 2048),)
                )
                for sw, stride in options:
                    key = f"shared/{name}/s{stage}/o{offset}/{m}_{n}_{k}/sw{int(sw)}_{stride}"
                    execute(key, fn, a, bt, ref, stage, sw, stride)
        print("shared", name, flush=True)
        checkpoint()
    if args.backend == "hip":
        for name, fn in funcs.items():
            for stage in (2, 3, 4, 5):
                for m, n, k, misalign in (
                    (65, 137, 80, 0),
                    (1, 3, 17, 1),
                    (129, 257, 129, 1),
                ):
                    a, bt = fixture(args.fixtures, m, n, k)
                    ref = (a.double() @ bt.double().t()).half()
                    execute(
                        f"tail/{name}/s{stage}/{m}_{n}_{k}",
                        fn,
                        a,
                        bt,
                        ref,
                        stage,
                        True,
                        128,
                        misalign,
                        False,
                    )
            a, bt = fixture(args.fixtures, 128, 128, 192)
            ref = (a.double() @ bt.double().t()).half()
            for stage in (-1, 0, 1, 9):
                execute(
                    f"fallback/{name}/{stage}",
                    fn,
                    a,
                    bt,
                    ref,
                    stage,
                    False,
                    0,
                    keep=False,
                )
        for kind in (3, 5):
            for stage in (2, 3, 4, 5):
                for f32 in (False, True):
                    ac = torch.ones(16, 1024, dtype=torch.float16)
                    bt = torch.full((16, 1024), 2**-15, dtype=torch.float16)
                    bt[:, :16] = 0.0625
                    ref = torch.full(
                        (16, 16), 1.03125 if f32 else 1.0, dtype=torch.float16
                    )
                    fn = lambda a, b, c, s, sw, stride: lib._diagnostic(
                        a, b, c, s, sw, stride, kind, f32, None
                    )
                    before = len(r["checks"])
                    execute(
                        f"state/k{kind}/s{stage}/f32{f32}",
                        fn,
                        ac,
                        bt,
                        ref,
                        stage,
                        False,
                        0,
                        keep=False,
                    )
                    r["state_controls"].append(
                        dict(
                            kind=kind,
                            stage=stage,
                            fp32=f32,
                            passed=all(
                                x["max_abs"] == 0 for x in r["checks"][before:]
                            ),
                        )
                    )
        for stage in (2, 3, 4, 5):
            with torch.cuda.stream(stream):
                a = torch.ones((64, 384), device="cuda", dtype=torch.float16)
                b = torch.ones((384, 128), device="cuda", dtype=torch.float16)
                c = torch.empty((64, 128), device="cuda", dtype=torch.float16)
                metrics = torch.zeros(16, device="cuda", dtype=torch.int32)
                lib._diagnostic(a, b, c, stage, False, 0, 3, True, metrics)
                stream.synchronize()
                v = metrics.cpu().tolist()
                r["pipeline"].append(
                    dict(
                        stage=stage,
                        metrics=v,
                        passed=v[:5] == [12] * 5
                        and v[5] == stage
                        and v[6] == 0
                        and v[8] == 12
                        and bool((c == 384).all()),
                    )
                )
        # Direct binding checks, including storage aliases and stage/grid validation.
        a = torch.ones((128, 128), device="cuda", dtype=torch.float16)
        b = a.clone()
        c = a.clone()
        tests = {
            "dtype": (a.float(), b, c, 2, False, 0),
            "cpu": (a.cpu(), b, c, 2, False, 0),
            "shape": (a, b, c[:64], 2, False, 0),
            "rank": (a.flatten(), b, c, 2, False, 0),
            "noncontiguous": (a.t(), b, c, 2, False, 0),
            "grad": (a.clone().requires_grad_(), b, c, 2, False, 0),
            "alias": (a, b, a, 2, False, 0),
            "stride_zero": (a, b, c, 2, True, 0),
            "stride_negative": (a, b, c, 2, True, -1),
            "empty": (a[:0], b, c[:0], 2, False, 0),
        }
        if torch.cuda.device_count() > 1:
            tests["mixed_device"] = (
                a,
                b.to("cuda:1" if args.device == 0 else "cuda:0"),
                c,
                2,
                False,
                0,
            )
        for name, fn in funcs.items():
            for key, values in tests.items():
                passed = False
                try:
                    fn(*values)
                except RuntimeError:
                    passed = True
                r["boundaries"].append(dict(id=name + "/" + key, passed=passed))
        # The guard must select the tensor's device and restore the caller's.
        if torch.cuda.device_count() > 1:
            other = 1 if args.device == 0 else 0
            for name, fn in funcs.items():
                with torch.cuda.device(other):
                    fn(a, b, c, 2, False, 0)
                    restored = torch.cuda.current_device() == other
                torch.cuda.synchronize(args.device)
                r["boundaries"].append(
                    dict(
                        id=name + "/device_guard",
                        passed=restored and bool((c == 128).all()),
                    )
                )
    if args.performance:
        for name, fn in funcs.items():
            for stage in (2, 3, 4, 5):
                for m, n, k in ((256, 256, 256), (512, 512, 1024)):
                    ac, btc = fixture(args.fixtures, m, n, k)
                    ref = (ac.double() @ btc.double().t()).half()
                    a = ac.cuda()
                    bt = btc.cuda()
                    b = bt.view(k, n)
                    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
                    base = torch.empty_like(c)
                    op = lambda: fn(a, b, c, stage, False, 2048)
                    pair = graph_pair(op, lambda: torch.mm(a, bt.t(), out=base))
                    pair.update(
                        id=f"{name}/s{stage}/{m}_{n}_{k}",
                        passed=accurate(c.cpu(), ref)["passed"]
                        and accurate(base.cpu(), ref)["passed"],
                        source_kind=(
                            r["reference_kind"][name]
                            if args.backend == "cuda"
                            else "native_amd"
                        ),
                    )
                    r["performance"].append(pair)
            print("perf", name, flush=True)
            checkpoint()
    torch.save(outputs, args.output.with_suffix(".pt"))
    r["outputs_sha256"] = sha(args.output.with_suffix(".pt"))
    if args.backend == "hip":
        from .isa_objects import hip_isa_objects

        r["isa"] = hip_isa_objects(
            lib.__file__, args.output.with_suffix("").resolve() / "isa"
        )
    else:
        binary = Path(lib.__file__)
        dest = args.output.parent / "binaries" / f"{sha(binary)}.so"
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            dest.write_bytes(binary.read_bytes())
        assert sha(dest) == meta["binary_sha256"]
        r["frozen_binary"] = str(dest.resolve().relative_to(ROOT))
    r["passed"] = all(
        x["passed"] and x.get("canary", True) and x.get("same_as_eager", True)
        for group in (
            "checks",
            "boundaries",
            "pipeline",
            "state_controls",
            "performance",
        )
        for x in r[group]
    )
    r["counts"] = {
        k: len(r[k])
        for k in (
            "checks",
            "boundaries",
            "pipeline",
            "state_controls",
            "performance",
        )
    }
    checkpoint()
    print(r["counts"], "PASSED", r["passed"], flush=True)


def compare(args):
    a, b = (torch.load(p, weights_only=True) for p in (args.hip, args.cuda))
    assert a.keys() == b.keys()
    checks = [
        dict(id=k, **accurate(a[k], b[k]), exact=torch.equal(a[k], b[k]))
        for k in a
    ]
    result = dict(
        checks=checks,
        passed=all(x["passed"] for x in checks),
        count=len(checks),
        exact=sum(x["exact"] for x in checks),
        hip_sha256=sha(args.hip),
        cuda_sha256=sha(args.cuda),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("COMPARISON", result["count"], result["exact"], result["passed"])
    assert result["passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=("hip", "cuda"), required=True)
    r.add_argument("--precision", choices=("original", "fp32"), default="fp32")
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--fixtures", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--original", type=Path, default=ROOT / "original")
    r.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    r.add_argument("--performance", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--hip", type=Path, required=True)
    c.add_argument("--cuda", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args) if args.command == "run" else compare(args)
