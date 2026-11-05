"""Probe direct-to-LDS builtin availability in the locked gfx1201 compiler."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .environment import ROOT, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": verify(),
        "probes": [],
    }
    folder = ROOT / "artifacts/rocm/async_copy/probes"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = """#include <hip/hip_runtime.h>
using V4 = int __attribute__((ext_vector_type(4)));
using Global = void __attribute__((address_space(1)));
using Local = void __attribute__((address_space(3)));
using Global4 = V4 __attribute__((address_space(1)));
using Local4 = V4 __attribute__((address_space(3)));
__global__ void probe(V4 *src, V4 *out) {
  __shared__ V4 tile[32];
"""
    for builtin, call in (
        (
            "global_load_lds",
            "__builtin_amdgcn_global_load_lds((Global*)(src+threadIdx.x), (Local*)tile, 4, 0, 0);",
        ),
        (
            "global_load_async_lds",
            "__builtin_amdgcn_global_load_async_lds((Global*)(src+threadIdx.x), (Local*)tile, 4, 0, 0);",
        ),
        (
            "global_load_async_to_lds_b128",
            "__builtin_amdgcn_global_load_async_to_lds_b128((Global4*)(src+threadIdx.x), (Local4*)(tile+threadIdx.x), 0, 0);",
        ),
    ):
        source = folder / f"{builtin}.cpp"
        source.write_text(
            prefix
            + call
            + "\n  __syncthreads();\n  out[threadIdx.x] = tile[threadIdx.x];\n}\n"
        )
        command = [
            "/opt/rocm/bin/hipcc",
            "-O3",
            "--offload-arch=gfx1201",
            "-c",
            str(source),
            "-o",
            str(source.with_suffix(".o")),
        ]
        proc = subprocess.run(command, text=True, capture_output=True)
        report["probes"].append(
            {
                "builtin": "__builtin_amdgcn_" + builtin,
                "source": str(source.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "command": command,
                "returncode": proc.returncode,
                "stderr": proc.stderr,
                "stdout": proc.stdout,
            }
        )
        print(builtin, proc.returncode, proc.stderr, flush=True)
    report["script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
