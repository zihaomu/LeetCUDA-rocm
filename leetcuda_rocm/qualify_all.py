"""Run the complete AMD functional acceptance in the locked project environment.

Cross-platform and performance qualifications remain separate frozen reports.
This entry point generates missing fixtures and preserves every suite's output.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .cross_validate import sha
from .environment import ROOT, verify


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/rocm/full_acceptance"
    )
    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Only run the Python operator suites",
    )
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(
        environment=verify(),
        harness_sha256=sha(__file__),
        suites=[],
        passed=False,
        scope="AMD functional regression; performance and CUDA source comparisons are separately qualified",
    )
    if len(report["environment"]["visible_device_pci_bus_ids"]) < 2:
        raise RuntimeError(
            "Full qualification requires two permitted visible GPUs for device guards"
        )
    manifest = folder / "manifest.json"

    def run(name, module, arguments, result=None):
        command = [
            sys.executable,
            "-m",
            "leetcuda_rocm." + module,
            *map(str, arguments),
        ]
        log = folder / (name + ".log")
        print(name, "running", flush=True)
        with log.open("w") as stream:
            proc = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
        data = (
            json.loads(result.read_text()) if result and result.exists() else {}
        )
        passed = proc.returncode == 0 and (
            data.get("passed", False) if result else True
        )
        row = dict(
            name=name,
            command=command,
            returncode=proc.returncode,
            log=str(log),
            log_sha256=sha(log),
            passed=passed,
        )
        if result:
            row.update(
                report=str(result),
                report_sha256=sha(result) if result.exists() else None,
                counts=data.get("counts"),
            )
        report["suites"].append(row)
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(name, "PASS" if passed else "FAIL", flush=True)
        if not passed:
            raise RuntimeError(f"{name} failed; see {log}")

    def fixture(name, module):
        path = folder / (name + "_fixtures.pt")
        if not path.exists():
            run(name + "_fixtures", module, ["fixtures", "--fixtures", path])
        return path

    def check(name, module, arguments=()):
        output = folder / (name + ".json")
        run(name, module, [*arguments, "--output", output], output)

    check("original229", "smoke")
    check("matrix_attention", "matrix_attention_check", ["--correctness-only"])
    for fmt in ("e4m3", "e5m2"):
        check(
            "fp8_" + fmt,
            "fp8_check",
            ["--fp8-format", fmt, "--guard-device", 1],
        )
    check("fp4", "fp4_check", ["--guard-device", 1])
    for name, module, extra in (
        ("triton", "triton_examples_check", ["--guard-device", 1]),
        (
            "training",
            "attention_training_check",
            ["--backend", "hip", "--guard-device", 1],
        ),
        ("ffpa", "ffpa_check", ["--backend", "hip", "--extras"]),
        (
            "interview_attention",
            "interview_attention_check",
            ["--backend", "hip"],
        ),
        ("supplemental", "supplemental_check", ["--backend", "hip"]),
        ("cute_dsl", "cute_dsl_check", ["--backend", "hip"]),
        ("swizzle", "swizzle_examples_check", ["--backend", "hip"]),
    ):
        path = fixture(name, module)
        check(name, module, ["run", "--fixtures", path, *extra])
    path = fixture("tiling", "flash_tiled_check")
    check(
        "unbound_attention",
        "unbound_attention_check",
        ["run", "--backend", "hip", "--fixtures", path],
    )
    binary_fixture = folder / "standalone_fixtures.bin"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/rocm/make_standalone_fixtures.py"),
            "--output",
            str(binary_fixture),
        ],
        check=True,
    )
    check(
        "unbound_gemm",
        "unbound_gemm_check",
        ["run", "--backend", "hip", "--fixtures", binary_fixture],
    )
    if not args.skip_cpp:
        cpp_folder = folder / "cpp"
        run(
            "standalone_cpp",
            "build_all_bench",
            ["--output", cpp_folder],
            cpp_folder / "manifest.json",
        )
    report["passed"] = all(x["passed"] for x in report["suites"])
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print("Complete functional acceptance:", report["passed"], flush=True)


if __name__ == "__main__":
    main()
