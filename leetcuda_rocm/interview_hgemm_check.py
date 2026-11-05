"""Standalone HGEMM schedules: actual original CUDA, explicit FP32 reference, HIP."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from .environment import ROOT
from .cross_validate import UPSTREAM, CUTLASS, sha


def cuda_header(original, precision):
    source = (original / "kernels/interview/hgemm.cuh").read_text()
    stream = "<<<grid, block, kShmSize>>>"
    assert source.count(stream) == 1
    source = source.replace(
        stream, "<<<grid, block, kShmSize, leetcuda_test_stream>>>"
    )
    if precision == "fp32":
        pattern = r"uint32_t RC\[kValTileM\]\[kValTileN\]\[2\] = \{0\};"
        starts = list(re.finditer(pattern, source))
        assert len(starts) == 3
        for match in reversed(starts):
            begin = match.start()
            end = re.search(r"// Epilogue[:：]", source[begin:]).start() + begin
            block = source[begin:end]
            block = re.sub(r"\bRC\b", "RF32", block).replace(
                "RF32[kValTileM][kValTileN][2] = {0}",
                "RF32[kValTileM][kValTileN][4] = {0}",
                1,
            )

            def mma(m):
                clean = re.sub(r"//[^\n]*", "", m[1])
                args = [p.strip() for p in clean.split(",")]
                assert len(args) == 10, args
                extra = lambda x, last: x.rsplit("[", 1)[0] + f"[{last}]"
                return (
                    "HMMA16816F32("
                    + ", ".join(
                        args[:2]
                        + [extra(args[0], 2), extra(args[0], 3)]
                        + args[2:8]
                        + args[8:]
                        + [extra(args[8], 2), extra(args[8], 3)]
                    )
                    + ");"
                )

            block, count = re.subn(
                r"HMMA16816\((.*?)\);", mma, block, flags=re.S
            )
            assert count == 1
            pack = """uint32_t RC[kValTileM][kValTileN][2];
    for (int i=0;i<kValTileM;i++) for(int j=0;j<kValTileN;j++) {
      RC[i][j][0] = uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(RF32[i][j][0])))) | (uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(RF32[i][j][1])))) << 16);
      RC[i][j][1] = uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(RF32[i][j][2])))) | (uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(RF32[i][j][3])))) << 16);
    }
    """
            source = source[:begin] + block + pack + source[end:]
    return source


def run(args):
    torch.cuda.set_device(args.device)
    source = ROOT / "kernels/rocm/interview_hgemm_bench.cpp"
    files = [source, ROOT / "kernels/rocm/bench_runtime.hpp", Path(__file__)]
    staged = None
    if args.backend == "hip":
        from .environment import verify, cache_key

        identity = verify()
        directory = ROOT / ".cache/rocm10" / cache_key(identity) / "cpp"
        compiler = "/opt/rocm/bin/hipcc"
        flags = [
            "-O3",
            "-std=c++20",
            "--offload-arch=gfx1201",
            "-DLEETCUDA_ROCM=1",
        ]
        include = [ROOT / "kernels"]
        libs = []
        files += [
            ROOT / p
            for p in (
                "kernels/interview/hgemm.cuh",
                "kernels/rocm/interview_hgemm_hip.cuh",
                "kernels/rocm/interview_common_hip.cuh",
            )
        ]
    else:
        identity = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )
        directory = ROOT / ".cache/cuda_comparison"
        compiler = "/usr/local/cuda/bin/nvcc"
        flags = [
            "-O3",
            "-std=c++20",
            "-arch=sm_120a",
            "-x",
            "cu",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DNOTES_V2_ENABLE_CUTE=1",
            "-DNOTES_V2_ENABLE_TMA_MMA_WS=1",
            f'-DLEETCUDA_REFERENCE_F32={int(args.precision=="fp32")}',
        ]
        include = [
            args.original / "kernels",
            args.original / "kernels/interview",
            args.cutlass / "include",
        ]
        libs = ["-lcuda", "-lcublas"]
        files += [
            args.original / "kernels/interview" / n
            for n in ("hgemm.cuh", "base.cuh", "common.cuh")
        ]
        staged = cuda_header(args.original, args.precision)
    sources = {str(p.resolve()): sha(p) for p in files}
    digest = hashlib.sha256(
        json.dumps([sources, flags, staged, CUTLASS], sort_keys=True).encode()
    ).hexdigest()[:16]
    folder = directory / f"interview_hgemm_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    if staged is not None:
        header = folder / "interview_hgemm_reference.cuh"
        header.write_text(staged)
        include.insert(0, folder)
    binary = folder / "hgemm_bench"
    command = [
        compiler,
        *flags,
        *sum((["-I", str(p.resolve())] for p in include), []),
        str(source),
        *libs,
        "-o",
        str(binary),
    ]
    if not binary.exists():
        subprocess.run(command, check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    invocation = [
        str(binary),
        str(args.device),
        str(args.fixtures),
        str(args.output.with_suffix(".bin")),
    ]
    if args.performance:
        invocation.append("--performance")
    result = subprocess.run(
        invocation, text=True, capture_output=True, timeout=600
    )
    args.output.with_suffix(".stdout").write_text(result.stdout)
    args.output.with_suffix(".stderr").write_text(result.stderr)
    checks = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith("{")
    ]
    r = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        environment=identity,
        source_kind=(
            "native_distinct_AMD_schedules"
            if args.backend == "hip"
            else (
                "original_with_current_stream_adapter"
                if args.precision == "original"
                else "explicit_FP32_accumulator_reference_with_current_stream_adapter"
            )
        ),
        wgmma_cuda_scope="Kind3 is cuBLAS FP32 mathematical reference; original Hopper WGMMA cannot compile for sm120",
        upstream_commit=UPSTREAM,
        cutlass_commit=CUTLASS,
        sources_sha256=sources,
        harness_sha256=sha(__file__),
        fixture_sha256=sha(args.fixtures),
        binary_sha256=sha(binary),
        build_command=command,
        invocation=invocation,
        checks=checks,
        returncode=result.returncode,
        linked_libraries=subprocess.check_output(
            ["ldd", str(binary)], text=True
        ),
        passed=result.returncode == 0
        and len(checks) == (477 if args.backend == "hip" else 336)
        and all(c["passed"] for c in checks),
    )
    if staged is not None:
        r["reference_header_sha256"] = sha(header)
        r["reference_header"] = str(header)
    if args.output.with_suffix(".bin").exists():
        r["output_sha256"] = sha(args.output.with_suffix(".bin"))
    if args.backend == "hip":
        from .isa_objects import hip_isa_objects

        r["isa"] = hip_isa_objects(
            binary, (args.output.parent / (args.output.stem + "_isa")).resolve()
        )
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print(
        "DONE",
        r["passed"],
        len(checks),
        "FAILED",
        sum(not c["passed"] for c in checks),
    )
    print(result.stderr)
    return int(not r["passed"])


def compare(args):
    a, b = (json.loads(p.read_text()) for p in (args.hip, args.cuda))
    assert a["fixture_sha256"] == b["fixture_sha256"]
    x, y = (
        np.fromfile(p.with_suffix(".bin"), dtype="<f8")
        for p in (args.hip, args.cuda)
    )
    ai = {c["id"]: c for c in a["checks"] if c["id"].startswith("shared/")}
    bi = {c["id"]: c for c in b["checks"]}
    assert ai.keys() == bi.keys() and len(ai) == 336
    checks = []
    for key, c in ai.items():
        d = bi[key]
        assert all(c[k] == d[k] for k in ("count", "atol", "rtol"))
        actual = x[c["offset"] : c["offset"] + c["count"]]
        expected = y[d["offset"] : d["offset"] + d["count"]]
        checks.append(
            dict(
                id=key,
                passed=bool(
                    np.allclose(
                        actual, expected, atol=c["atol"], rtol=c["rtol"]
                    )
                ),
                exact=bool(np.array_equal(actual, expected)),
                max_abs=float(np.max(np.abs(actual - expected))),
            )
        )
    r = dict(
        passed=a["passed"] and b["passed"] and all(c["passed"] for c in checks),
        checks=checks,
        fixture_sha256=a["fixture_sha256"],
        reports_sha256={str(p): sha(p) for p in (args.hip, args.cuda)},
        scope="four CUDA source schedules; Kind3 uses explicitly identified cuBLAS mathematical reference",
    )
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print("COMPARE", r["passed"], len(checks))
    return int(not r["passed"])


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=("hip", "cuda"), required=True)
    r.add_argument(
        "--precision", choices=("original", "fp32"), default="original"
    )
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--fixtures", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--performance", action="store_true")
    r.add_argument("--original", type=Path, default=ROOT / "original")
    r.add_argument("--cutlass", type=Path, default=ROOT / ".cache/cutlass")
    c = sub.add_parser("compare")
    for n in ("hip", "cuda", "output"):
        c.add_argument("--" + n, type=Path, required=True)
    args = p.parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
