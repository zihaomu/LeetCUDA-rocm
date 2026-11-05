"""Compare original CUDA FP16 state against the explicitly FP32 AMD WMMA port."""

import argparse
import json
from pathlib import Path

import torch

from .cross_validate import sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--hip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = (args.cuda, args.hip)
    reports = [json.loads(p.read_text()) for p in paths]
    assert reports[0]["backend"] == "cuda" and reports[1]["backend"] == "hip"
    assert reports[0]["fixture_sha256"] == reports[1]["fixture_sha256"]
    assert reports[0]["harness_sha256"] == reports[1]["harness_sha256"]
    assert reports[1]["passed"], "Do not publish an unqualified AMD candidate"
    values = []
    for p, report in zip(paths, reports):
        assert sha(p.with_suffix(".pt")) == report["outputs_sha256"]
        values.append(torch.load(p.with_suffix(".pt"), weights_only=True))
    checks = [{r["id"]: r for r in report["checks"]} for report in reports]
    assert (
        values[0].keys()
        == values[1].keys()
        == checks[0].keys()
        == checks[1].keys()
    )
    result = dict(
        fixture_sha256=reports[0]["fixture_sha256"],
        report_sha256={
            backend: sha(path) for backend, path in zip(("cuda", "hip"), paths)
        },
        note="FP16 I/O, CUDA FP16 state versus AMD FP32 state. Parity and FP64 accuracy are separate gates.",
        checks=[],
        performance=[],
    )
    for key, cuda in values[0].items():
        hip = values[1][key]
        cc, hc = checks[0][key], checks[1][key]
        assert cc["input_sha256"] == hc["input_sha256"]
        assert (cc["atol"], cc["rtol"]) == (hc["atol"], hc["rtol"])
        result["checks"].append(
            dict(
                id=key,
                cuda_accuracy=cc["passed"],
                hip_accuracy=hc["passed"],
                bitwise_equal=torch.equal(cuda, hip),
                parity_within_gate=torch.allclose(
                    cuda, hip, atol=hc["atol"], rtol=hc["rtol"]
                ),
                max_abs_cuda_hip=(cuda.float() - hip.float())
                .abs()
                .max()
                .item(),
            )
        )
    perf = {
        json.dumps([p["function"], p["shape_mnk"], p["config"]]): p
        for p in reports[0]["performance"]
    }
    for hp in reports[1]["performance"]:
        cp = perf[json.dumps([hp["function"], hp["shape_mnk"], hp["config"]])]
        assert hp["graph_replay_verified"] and cp["graph_replay_verified"]
        result["performance"].append(
            dict(
                function=hp["function"],
                shape_mnk=hp["shape_mnk"],
                config=hp["config"],
                cuda_ms=cp["ours_ms"],
                hip_ms=hp["ours_ms"],
                cuda_time_div_hip_time=cp["ours_ms"] / hp["ours_ms"],
                cuda_accuracy=cp["accuracy_passed"],
                hip_accuracy=hp["accuracy_passed"],
                hip_torch_speed_ratio=hp["baseline_over_ours"],
            )
        )
    result["summary"] = dict(
        shared_cases=len(result["checks"]),
        **{
            key: sum(c[key] for c in result["checks"])
            for key in (
                "cuda_accuracy",
                "hip_accuracy",
                "bitwise_equal",
                "parity_within_gate",
            )
        },
        performance_pairs=len(result["performance"]),
        amd_qualification_passed=reports[1]["passed"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
