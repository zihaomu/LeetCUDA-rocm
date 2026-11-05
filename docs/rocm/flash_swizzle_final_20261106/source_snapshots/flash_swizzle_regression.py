"""Compare all four existing Attention exports with the pre-swizzle binary."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from .benchmark import graph_pair
from .cross_validate import ROOT, sha, tensor_sha
from .environment import verify
from .flash_swizzle_check import telemetry
from .loader import load, source_paths


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    identity = verify()
    baseline = json.loads(args.baseline.read_text())
    binary = ROOT / baseline["isa"]["binary"]
    assert sha(binary) == baseline["isa"]["binary_sha256"]
    spec = importlib.util.spec_from_file_location(
        Path(baseline["isa"]["build_cache_binary"]).stem, binary
    )
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    current = load("flash_attn")
    checks, cases = [], []
    fixture_sha256 = {}
    for kind in ("shared", "tiling"):
        suffixes = ("kv", "qkv") if kind == "shared" else ("qk", "qkv")
        fixture_path = (
            ROOT
            / f"artifacts/rocm/flash_{'tiled' if kind == 'tiling' else 'shared'}/fixtures.pt"
        )
        fixture = torch.load(fixture_path, weights_only=True)
        fixture_sha256[kind] = sha(fixture_path)
        for suffix in suffixes:
            name = f"flash_attn_mma_stages_split_q_{kind}_{suffix}_acc_f32"
            for case in fixture:
                assert case["input_sha256"] == [
                    tensor_sha(t) for t in case["inputs"]
                ]
                inputs = [t.cuda() for t in case["inputs"]]
                for stage in (1, 2):
                    if (
                        kind == "shared"
                        and inputs[0].shape[-1] == 256
                        and stage == 2
                    ):
                        continue
                    old, new = torch.empty_like(inputs[0]), torch.empty_like(
                        inputs[0]
                    )
                    getattr(frozen, name)(*inputs, old, stage)
                    getattr(current, name)(*inputs, new, stage)
                    equal = torch.equal(old, new)
                    accuracy = torch.allclose(
                        new.cpu(),
                        case["reference"].half(),
                        atol=0.003,
                        rtol=0.003,
                    )
                    checks.append(
                        dict(
                            id=f"{name}/{case['id']}/s{stage}",
                            passed=equal and accuracy,
                            bitwise_equal=equal,
                            accuracy_passed=accuracy,
                        )
                    )
                    if case["performance"]:
                        cases.append((name, case, stage))
    report = dict(
        environment=identity,
        harness_sha256=sha(__file__),
        fixture_sha256=fixture_sha256,
        helper_sha256={
            path: sha(ROOT / path)
            for path in (
                "leetcuda_rocm/benchmark.py",
                "leetcuda_rocm/flash_swizzle_check.py",
            )
        },
        source_sha256={
            str(p.relative_to(ROOT)): sha(p) for p in source_paths("flash_attn")
        },
        baseline_report_sha256=sha(args.baseline),
        baseline_binary_sha256=sha(binary),
        current_binary_sha256=sha(current.__file__),
        checks=checks,
        performance=[],
        telemetry=[],
    )
    for round_id in range(3):
        before = telemetry("hip")
        ordered = cases[round_id:] + cases[:round_id]
        for name, case, stage in ordered:
            inputs = [t.cuda() for t in case["inputs"]]
            old, new = torch.empty_like(inputs[0]), torch.empty_like(inputs[0])
            pair = graph_pair(
                lambda: getattr(current, name)(*inputs, new, stage),
                lambda: getattr(frozen, name)(*inputs, old, stage),
                calls=8,
                sample_ms=40,
                trials=5,
            )
            report["performance"].append(
                dict(
                    id=f"{name}/{case['id']}/s{stage}/r{round_id}",
                    function=name,
                    shape=list(new.shape),
                    stage=stage,
                    round=round_id,
                    input_sha256=case["input_sha256"],
                    within_ten_percent=pair["ours_ms"]
                    <= pair["baseline_ms"] * 1.10,
                    **pair,
                )
            )
        report["telemetry"].append(
            dict(round=round_id, before=before, after=telemetry("hip"))
        )
        print(
            f"existing export regression round {round_id + 1}/3 complete",
            flush=True,
        )
    report["passed"] = all(c["passed"] for c in checks)
    report["performance_no_regression"] = all(
        p["within_ten_percent"] for p in report["performance"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                checks=len(checks),
                performance=len(report["performance"]),
                performance_no_regression=report["performance_no_regression"],
            )
        )
    )
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
