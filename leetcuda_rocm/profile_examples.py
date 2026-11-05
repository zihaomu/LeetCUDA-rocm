"""Small, validated workload for rocprofv3; all twelve original profiling kernels."""

import argparse
import json
from pathlib import Path
import torch
from .supplemental import build, NAMES
from .environment import verify
from .cross_validate import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--size", type=int, default=1048576)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeat < 1 or args.size < 1 or args.size % 8:
        raise ValueError("positive repeat and size divisible by eight required")
    torch.set_num_threads(4)
    lib, meta = build("profile")
    records = []
    for kind, name in enumerate(NAMES["profile"]):
        dtype = torch.float32 if kind % 6 < 2 else torch.float16
        a = (
            torch.arange(args.size, device="cuda", dtype=torch.float32)
            .remainder_(17)
            .sub_(8)
            .to(dtype)
        )
        b = torch.ones_like(a)
        o = torch.empty_like(a)
        for _ in range(args.repeat):
            lib.forward(a, b, o, kind)
        torch.cuda.synchronize()
        ref = a + b if kind < 6 else a.clamp(min=0)
        records.append(
            dict(name=name, passed=torch.equal(o, ref), launches=args.repeat)
        )
    report = dict(
        environment=verify(),
        build=meta,
        checks=records,
        passed=all(x["passed"] for x in records),
        harness_sha256=sha(__file__),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"]
    print(
        json.dumps(
            dict(
                passed=True,
                kernels=len(records),
                launches=args.repeat * len(records),
            )
        )
    )


if __name__ == "__main__":
    main()
