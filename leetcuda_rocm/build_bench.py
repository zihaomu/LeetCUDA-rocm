"""Build/run qualified standalone HIP benchmarks in the locked foundation."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .environment import ROOT, cache_key, verify
from .isa import hip_isa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=("fp8", "fp4"))
    parser.add_argument(
        "--fp8-format", choices=("e4m3", "e5m2"), default="e4m3"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.module != "fp8" and args.fp8_format != "e4m3":
        parser.error("--fp8-format only applies to fp8")
    output_dir = (
        ROOT
        / "artifacts/rocm"
        / ("fp8_native" if args.module == "fp8" else "fp4")
    )
    if args.module == "fp8" and args.fp8_format == "e5m2":
        output_dir = output_dir / "e5m2"
    if args.output is None:
        args.output = output_dir / "cpp.json"
    if args.performance:
        args.run = True
    torch.cuda.set_device(args.device)
    identity = verify()
    sources = [
        ROOT / s
        for s in (
            f"kernels/rocm/{args.module}_bench.cpp",
            f"kernels/rocm/{args.module}_gemm_hip.cuh",
            f"kernels/interview/{args.module}_gemm.cuh",
            "leetcuda_rocm/build_bench.py",
        )
    ]
    digest = hashlib.sha256(
        b"".join(p.read_bytes() for p in sources) + args.fp8_format.encode()
    ).hexdigest()[:16]
    directory = (
        ROOT
        / ".cache/rocm10"
        / cache_key(identity)
        / "cpp"
        / f"{args.module}_{digest}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / f"{args.module}_bench"
    command = [
        "/opt/rocm/bin/hipcc",
        "-O3",
        "-std=c++20",
        "--offload-arch=gfx1201",
        "-DLEETCUDA_ROCM=1",
        *(["-DLEETCUDA_FP8_E5M2=1"] if args.fp8_format == "e5m2" else []),
        "-I",
        str(ROOT / "kernels"),
        str(sources[0]),
        "-lhipblas",
        "-o",
        str(binary),
    ]
    if not binary.exists():
        subprocess.run(command, check=True)
    libraries = {}
    for line in subprocess.check_output(
        ["ldd", str(binary)], text=True
    ).splitlines():
        match = re.search(r"=> (/\S+)", line)
        if match and any(
            name in line
            for name in (
                "amdhip64",
                "hipblas",
                "rocblas",
                "hsa-runtime",
                "amd_comgr",
            )
        ):
            path = Path(match[1])
            libraries[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": identity,
        "logical_device": args.device,
        "build_command": command,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "linked_runtime_sha256": libraries,
        "isa": hip_isa(binary, output_dir / "cpp_isa"),
        "module": args.module,
        "fp8_format": args.fp8_format if args.module == "fp8" else None,
        "checks": [],
        "performance": [],
        "executed": args.run,
        "benchmark_method": (
            "HIP graphs of 32 calls, "
            + "adaptive replays targeting 40ms"
            + ", five alternating samples; hipBLAS BF16 FP32-accumulation baseline; allocations and offline B quantization excluded"
        ),
    }
    if args.run:
        proc = subprocess.run(
            [str(binary), str(args.device)]
            + (["--performance"] if args.performance else []),
            text=True,
            capture_output=True,
        )
        for line in proc.stdout.splitlines():
            row = json.loads(line)
            report[
                "performance" if row["kind"] == "performance" else "checks"
            ].append(row)
        report["returncode"] = proc.returncode
        report["stderr"] = proc.stderr
        report["passed"] = (
            proc.returncode == 0
            and bool(report["checks"])
            and all(r["passed"] for r in report["checks"])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "binary": str(binary),
                "checks": len(report["checks"]),
                "performance": len(report["performance"]),
                "passed": report.get("passed"),
            }
        )
    )
    if args.run and not report["passed"]:
        print(report["stderr"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
