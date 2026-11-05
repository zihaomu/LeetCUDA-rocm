"""Compare the shared-input Triton reports without conflating parity/accuracy."""

import argparse
import json
from pathlib import Path

import torch

from .triton_examples_check import sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--hip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in (args.cuda, args.hip)]
    for key in ("fixture_sha256", "harness_sha256", "source_sha256"):
        assert reports[0][key] == reports[1][key], key
    values = []
    for path, report in zip((args.cuda, args.hip), reports):
        assert report["outputs_sha256"] == sha(path.with_suffix(".pt"))
        values.append(torch.load(path.with_suffix(".pt"), weights_only=True))
    assert values[0].keys() == values[1].keys()
    gates = {r["id"]: r for r in reports[0]["checks"] if "atol" in r}
    rows = []
    for key, left in values[0].items():
        right = values[1][key]
        gate = gates[key]
        rows.append(
            dict(
                id=key,
                bitwise_equal=torch.equal(left, right),
                parity_passed=torch.allclose(
                    left, right, atol=gate["atol"], rtol=gate["rtol"]
                ),
            )
        )
    result = dict(
        fixture_sha256=reports[0]["fixture_sha256"],
        report_sha256={"cuda": sha(args.cuda), "hip": sha(args.hip)},
        independent_accuracy_passed=all(r["passed"] for r in reports),
        checks=rows,
        summary=dict(
            total=len(rows),
            bitwise_equal=sum(r["bitwise_equal"] for r in rows),
            parity_passed=sum(r["parity_passed"] for r in rows),
        ),
    )
    result["passed"] = result["independent_accuracy_passed"] and all(
        r["parity_passed"] for r in rows
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
