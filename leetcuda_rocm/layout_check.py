"""Qualify and benchmark the eleven AMD implementations of CuTe transposes."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .environment import ROOT, verify
from .loader import load, source_paths
from .matrix_attention_check import timed_pair
from .transpose_checks import transpose_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261103)
    torch.set_grad_enabled(False)
    lib = load("mat_transpose")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": verify(),
        "logical_device": args.device,
        "seed": 20261103,
        "checks": transpose_checks(lib),
        "performance": [],
        "method": "HIP graphs, reused buffers, 5 alternating 40ms samples; median. Effective bytes are one read plus one write, not measured HBM traffic.",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                *source_paths("mat_transpose"),
                *sorted((ROOT / "leetcuda_rocm").glob("*.py")),
            )
        },
        "passed": False,
    }
    for m, n in ((256, 512), (1024, 1024), (4096, 4096)):
        x = torch.randn(m, n, device="cuda")
        out = torch.empty(n, m, device="cuda")
        ref = torch.empty_like(out)
        for name in sorted(
            n for n in dir(lib) if n.startswith("mat_transpose_cute")
        ):
            ours = lambda: getattr(lib, name)(x, out)
            baseline = lambda: ref.copy_(x.T)
            ours()
            baseline()
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
            timing = timed_pair(ours, baseline)
            record = {
                "function": name,
                "shape": [m, n],
                "baseline": "torch.Tensor.copy_(x.T), preallocated output",
                "correctness_passed": True,
                **timing,
                "effective_GBps": 8 * m * n / (timing["ours_ms"] * 1e6),
            }
            report["performance"].append(record)
            print(json.dumps(record), flush=True)
    report["passed"] = all(row["passed"] for row in report["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
