"""Build, execute and compare original/ported standalone interview examples."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .cross_validate import UPSTREAM, sha
from .environment import ROOT


def run(args):
    torch.cuda.set_device(args.device)
    if args.backend == "hip":
        from .environment import verify, cache_key

        identity = verify()
        include = ROOT / "kernels"
        compiler = "/opt/rocm/bin/hipcc"
        flags = [
            "-O3",
            "-std=c++20",
            "--offload-arch=gfx1201",
            "-DLEETCUDA_ROCM=1",
        ]
        directory = ROOT / ".cache/rocm10" / cache_key(identity) / "cpp"
    else:
        identity = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            image=os.environ.get("LEETCUDA_CUDA_IMAGE"),
        )
        include = args.original / "kernels"
        compiler = "/usr/local/cuda/bin/nvcc"
        flags = ["-O3", "-std=c++20", "-arch=sm_120a", "-x", "cu"]
        directory = ROOT / ".cache/cuda_comparison"
    source = ROOT / f"kernels/rocm/interview_{args.suite}_bench.cpp"
    files = [
        source,
        ROOT / "kernels/rocm/bench_runtime.hpp",
        include
        / (
            "interview/sgemv.cuh"
            if args.suite == "base"
            else "interview/sgemm.cuh"
        ),
        include / "interview/base.cuh",
        include / "interview/common.cuh",
    ]
    if args.backend == "hip":
        files.append(ROOT / "kernels/rocm/interview_common_hip.cuh")
    if args.backend == "hip" and args.suite == "sgemm":
        files.append(ROOT / "kernels/rocm/interview_sgemm_hip.cuh")
    sources = {str(p): sha(p) for p in files}
    digest = hashlib.sha256(
        json.dumps([sources, flags], sort_keys=True).encode()
    ).hexdigest()[:16]
    folder = directory / f"interview_{args.suite}_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / f"{args.suite}_bench"
    command = [
        compiler,
        *flags,
        "-I",
        str(include),
        str(source),
        "-o",
        str(binary),
    ]
    if not binary.exists():
        subprocess.run(command, check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = args.output.with_suffix(".bin")
    invocation = [
        str(binary),
        str(args.device),
        str(args.fixtures),
        str(outputs),
    ]
    if args.performance:
        invocation.append("--performance")
    result = subprocess.run(invocation, text=True, capture_output=True)
    args.output.with_suffix(".stdout").write_text(result.stdout)
    args.output.with_suffix(".stderr").write_text(result.stderr)
    checks = [json.loads(line) for line in result.stdout.splitlines()]
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        suite=args.suite,
        environment=identity,
        source_kind=(
            "upstream_original"
            if args.backend == "cuda"
            else (
                "native_port_of_original_SIMT"
                if args.suite == "base"
                else "SIMT_and_BF16_WMMA_single_TF32"
            )
        ),
        upstream_commit=UPSTREAM,
        sources_sha256=sources,
        harness_sha256=sha(__file__),
        fixture_sha256=sha(args.fixtures),
        output_sha256=sha(outputs) if outputs.exists() else None,
        binary_sha256=sha(binary),
        build_command=command,
        invocation=invocation,
        checks=checks,
        returncode=result.returncode,
        linked_libraries=subprocess.check_output(
            ["ldd", str(binary)], text=True
        ),
        timing_method=(
            "5 warmup then 100 event-timed calls, non-default stream, includes submission gaps; no speedup claim"
            if args.performance
            else None
        ),
        passed=result.returncode == 0
        and len(checks) == {"base": 370, "sgemm": 234}[args.suite]
        and all(c["passed"] for c in checks),
    )
    if args.backend == "hip":
        from .isa import hip_isa

        report["isa"] = hip_isa(
            binary, (args.output.parent / (args.output.stem + "_isa")).resolve()
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", report["passed"], len(checks), flush=True)
    if not report["passed"]:
        print(result.stderr)
        raise SystemExit(1)


def compare(args):
    a = json.loads(args.hip.read_text())
    b = json.loads(args.cuda.read_text())
    assert a["fixture_sha256"] == b["fixture_sha256"]
    x = np.fromfile(args.hip.with_suffix(".bin"), dtype="<f8")
    y = np.fromfile(args.cuda.with_suffix(".bin"), dtype="<f8")
    checks = []
    assert len(a["checks"]) == len(b["checks"])
    for c, d in zip(a["checks"], b["checks"]):
        assert all(
            c[k] == d[k] for k in ("id", "count", "offset", "atol", "rtol")
        )
        start = c["offset"]
        end = start + c["count"]
        expected = y[start:end]
        actual = x[start:end]
        checks.append(
            dict(
                id=c["id"],
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
        hip_sha256=sha(args.hip),
        cuda_sha256=sha(args.cuda),
        fixture_sha256=a["fixture_sha256"],
    )
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print("COMPARE", r["passed"], len(checks))
    if not r["passed"]:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", choices=("base", "sgemm"), default="base")
    r.add_argument("--backend", choices=("hip", "cuda"), required=True)
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--fixtures", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--performance", action="store_true")
    r.add_argument("--original", type=Path, default=ROOT / "original")
    c = sub.add_parser("compare")
    c.add_argument("--hip", type=Path, required=True)
    c.add_argument("--cuda", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
