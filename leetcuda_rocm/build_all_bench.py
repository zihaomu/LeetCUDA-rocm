"""Build and execute every qualified standalone HIP suite, recording each result."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from .environment import ROOT, verify, cache_key
from .cross_validate import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/rocm/all_cpp"
    )
    args = p.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    identity = verify()
    report = dict(
        environment=identity,
        logical_device=0,
        suites=[],
        passed=False,
        harness_sha256=sha(__file__),
    )
    fixture = folder / "fixtures.bin"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/rocm/make_standalone_fixtures.py"),
            "--output",
            str(fixture),
        ],
        check=True,
    )

    def record(name, command, path):
        log = folder / (name + ".log")
        with log.open("w") as f:
            proc = subprocess.run(
                command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
            )
        result = json.loads(path.read_text()) if path.exists() else {}
        row = dict(
            name=name,
            command=command,
            returncode=proc.returncode,
            report=str(path),
            report_sha256=sha(path) if path.exists() else None,
            log_sha256=sha(log),
            passed=proc.returncode == 0 and result.get("passed", False),
            checks=len(result.get("checks", [])),
        )
        report["suites"].append(row)
        (folder / "manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(name, row["passed"], row["checks"], flush=True)
        if not row["passed"]:
            raise RuntimeError(f"{name} failed: {log}")

    for suite in ("base", "sgemm"):
        path = folder / (suite + ".json")
        record(
            suite,
            [
                sys.executable,
                "-m",
                "leetcuda_rocm.standalone_check",
                "run",
                "--backend",
                "hip",
                "--suite",
                suite,
                "--fixtures",
                str(fixture),
                "--output",
                str(path),
            ],
            path,
        )
    path = folder / "hgemm.json"
    record(
        "hgemm",
        [
            sys.executable,
            "-m",
            "leetcuda_rocm.interview_hgemm_check",
            "run",
            "--backend",
            "hip",
            "--fixtures",
            str(fixture),
            "--output",
            str(path),
        ],
        path,
    )
    for name, extra in (
        ("fp8_e4m3", []),
        ("fp8_e5m2", ["--fp8-format", "e5m2"]),
        ("fp4", []),
        ("ffpa", []),
    ):
        module = "fp8" if name.startswith("fp8") else name
        path = folder / (name + ".json")
        record(
            name,
            [
                sys.executable,
                "-m",
                "leetcuda_rocm.build_bench",
                module,
                "--run",
                "--output",
                str(path),
                *extra,
            ],
            path,
        )
    # New raw C++ suites have no Torch headers or link dependencies.
    for name in ("interview_attention", "supplemental"):
        if name == "supplemental":
            binary = folder / "supplemental"
            cmd = [
                sys.executable,
                str(ROOT / "scripts/rocm/build_supplemental.py"),
                "--output",
                str(binary),
            ]
        else:
            files = [
                ROOT / "kernels/rocm" / x
                for x in (
                    "interview_attention_bench.cpp",
                    "interview_flash_attention_hip.cuh",
                    "interview_common_hip.cuh",
                    "bench_runtime.hpp",
                )
            ] + [ROOT / "kernels/interview/flash_attn.cuh"]
            digest = hashlib.sha256(
                b"".join(p.read_bytes() for p in files)
            ).hexdigest()[:16]
            directory = (
                ROOT
                / ".cache/rocm10"
                / cache_key(identity)
                / "cpp"
                / ("interview_attention_" + digest)
            )
            directory.mkdir(parents=True, exist_ok=True)
            binary = directory / "bench"
            cmd = [
                "/opt/rocm/bin/hipcc",
                "-O3",
                "-std=c++20",
                "--offload-arch=gfx1201",
                "-DLEETCUDA_ROCM=1",
                "-I" + str(ROOT / "kernels"),
                "-I" + str(ROOT / "kernels/rocm"),
                str(files[0]),
                "-o",
                str(binary),
            ]
            if binary.exists():
                cmd = None
        if cmd:
            with (folder / (name + "_build.log")).open("w") as log:
                subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        proc = subprocess.run(
            [str(binary), str(fixture), str(folder / (name + ".bin"))],
            text=True,
            capture_output=True,
        )
        (folder / (name + ".log")).write_text(proc.stdout + proc.stderr)
        checks = [
            json.loads(line)
            for line in proc.stdout.splitlines()
            if line.startswith('{"id"')
        ]
        expected = 646 if name == "interview_attention" else 100
        linked = subprocess.check_output(["ldd", str(binary)], text=True)
        row = dict(
            name=name,
            checks=len(checks),
            returncode=proc.returncode,
            binary=str(binary),
            binary_sha256=sha(binary),
            fixture_sha256=sha(fixture),
            linked_libraries=linked,
            passed=proc.returncode == 0
            and len(checks) == expected
            and all(x["passed"] for x in checks)
            and "libtorch" not in linked,
        )
        (folder / (name + ".json")).write_text(
            json.dumps(dict(**row, records=checks), indent=2) + "\n"
        )
        report["suites"].append(row)
        print(name, row["passed"], row["checks"], flush=True)
        if not row["passed"]:
            raise RuntimeError(name + " failed")
    report["passed"] = all(x["passed"] for x in report["suites"])
    report["total_checks"] = sum(x["checks"] for x in report["suites"])
    (folder / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                suites=len(report["suites"]),
                checks=report["total_checks"],
            )
        )
    )


if __name__ == "__main__":
    main()
