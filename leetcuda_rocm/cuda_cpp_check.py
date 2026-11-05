"""Capture standalone CUDA benchmark execution; exit zero is not qualification.

Run inside scripts/cuda/run_cpp.sh. Raw metrics have different references
(including pre-quantization FP64), so this records them without inventing a
single numerical threshold for the entire benchmark.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    linkage = subprocess.check_output(["ldd", str(binary)], text=True)
    libraries = re.findall(r"=> (/[^ ]+)", linkage)
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        image=os.environ["LEETCUDA_CUDA_IMAGE"],
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        physical_gpu=os.environ["LEETCUDA_CUDA_GPU"],
        nvcc=subprocess.check_output(
            ["/usr/local/cuda/bin/nvcc", "--version"], text=True
        ),
        binary_sha256=sha(binary),
        harness_sha256=sha(__file__),
        source_sha256={
            str(p.relative_to(args.source_root)): sha(p)
            for p in sorted(args.source_root.rglob("*"))
            if p.suffix in (".cu", ".cuh", ".h", ".sh")
        },
        runtime_linkage=linkage,
        runtime_library_sha256={p: sha(p) for p in libraries},
        qualification_passed=False,
        scope="Execution and printed-metric audit only; no shared CUDA/HIP fixtures or unified numerical gate for C++ yet",
        hardware_excluded=[
            "Hopper WGMMA: sm_90a-only path is not built for sm_120a"
        ],
        cases=[],
    )
    cases = [
        ("base_256", ["256", "256", "256"]),
        *[
            (flag, ["--" + flag])
            for flag in (
                "fp8-gemm",
                "fp4-gemm",
                "fa3-cute-tma-smoke",
                "fa3-cute",
                "fa2-cute",
                "fa2-cute-cpasync",
                "pd-cute",
                "sdnw-cute",
            )
        ],
        (
            "performance_512",
            [
                "--bench-all",
                "--mnk",
                "512,512,512",
                "--bhnd",
                "1,2,256,64",
                "--warmup",
                "3",
                "--repeat",
                "10",
            ],
        ),
    ]
    metric = re.compile(
        r"\|\s*([^|\n]+?)\s*\|\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)\s*\|"
    )
    for name, flags in cases:
        log = args.output / (name + ".log")
        with log.open("w") as stream:
            try:
                proc = subprocess.run(
                    [str(binary), *flags],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=240,
                )
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        contents = log.read_text(errors="replace")
        metrics = [
            dict(label=label.strip(), value=value)
            for label, value in metric.findall(contents)
        ]
        row = dict(
            name=name,
            arguments=flags,
            returncode=code,
            execution_completed=code == 0,
            log_sha256=sha(log),
            metrics=metrics,
            numerical_qualification="not_established",
        )
        report["cases"].append(row)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                dict(name=name, returncode=code, printed_metrics=len(metrics))
            ),
            flush=True,
        )
    return int(any(not c["execution_completed"] for c in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
