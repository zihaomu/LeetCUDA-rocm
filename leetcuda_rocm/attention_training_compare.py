"""Compare frozen attention forward/gradient outputs with unchanged gates."""

import argparse
import json
from pathlib import Path

import torch

from .attention_training_check import GATES, check
from .cross_validate import sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("hip", "cuda", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = (args.hip, args.cuda)
    reports = [json.loads(p.read_text()) for p in paths]
    assert reports[0]["fixture_sha256"] == reports[1]["fixture_sha256"]
    assert (
        reports[0]["implementation_sha256"]
        == reports[1]["implementation_sha256"]
    )
    assert all(r["passed"] for r in reports)
    for p, r in zip(paths, reports):
        assert sha(p.with_suffix(".pt")) == r["outputs_sha256"]
    outputs = [
        torch.load(p.with_suffix(".pt"), weights_only=True) for p in paths
    ]
    assert outputs[0].keys() == outputs[1].keys()
    checks = []
    for key in outputs[0]:
        a, b = (o[key] for o in outputs)
        check(a, b, a.dtype, key, checks)
        checks[-1]["exact"] = torch.equal(a, b)
    report = dict(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        fixture_sha256=reports[0]["fixture_sha256"],
        report_sha256={str(p): sha(p) for p in paths},
        source_kind="same Triton attention algorithm on two backends, both independently checked against CPU FP64 and Philox; not an upstream CUDA training implementation",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                checks=len(checks),
                exact=sum(c["exact"] for c in checks),
            )
        )
    )
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
