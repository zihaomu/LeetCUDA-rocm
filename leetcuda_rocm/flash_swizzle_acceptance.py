"""Judge the frozen swizzle plan's correctness and three-round timing gates."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def performance(rows, kernels):
    groups = defaultdict(list)
    for row in rows:
        groups[row["function"]].append(row)
    output = []
    for name, selected in groups.items():
        configs = defaultdict(list)
        for row in selected:
            configs[(tuple(row["shape"]), row["stage"])].append(row)
        rounds = []
        for index in range(3):
            samples = [r for r in selected if r["round"] == index]
            assert len(samples) == 18
            gain = math.exp(
                statistics.mean(
                    math.log(r["parent_ms"] / r["ours_ms"]) for r in samples
                )
            )
            worst = max(r["ours_ms"] / r["parent_ms"] for r in samples)
            rounds.append(
                dict(
                    round=index,
                    geometric_mean_speedup=gain,
                    worst_slowdown=worst,
                    passed=gain >= 1.05 and worst <= 1.10,
                )
            )
        stability = []
        for (shape, stage), samples in configs.items():
            assert len(samples) == 3
            variation = {}
            for field in ("ours_ms", "parent_ms"):
                values = [r[field] for r in samples]
                variation[field] = (
                    max(values) - min(values)
                ) / statistics.median(values)
            stability.append(
                dict(
                    shape=shape,
                    stage=stage,
                    relative_range=variation,
                    stable=max(variation.values()) <= 0.05,
                )
            )
        mask = {"q": 1, "qk": 3, "qkv": 7}[name.rsplit("_", 1)[1]]
        resource = [k for k in kernels if k["swizzle_mask"] == mask]
        assert len(resource) == 14
        resource_ok = all(
            k["scratch"] == 0
            and k["vgpr_spill_count"] == 0
            and k["dynamic_lds_bytes"] <= 49152
            for k in resource
        )
        stable = all(r["stable"] for r in stability)
        output.append(
            dict(
                function=name,
                configurations=len(configs),
                rounds=rounds,
                stability=stability,
                stable=stable,
                resources_passed=resource_ok,
                passed=stable
                and resource_ok
                and all(r["passed"] for r in rounds),
            )
        )
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--hip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.hip.read_text())
    result = dict(
        input_report_sha256=hashlib.sha256(args.hip.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        correctness_passed=report["passed"],
        performance=performance(
            report["performance"], report["isa"]["kernels"]
        ),
    )
    result["performance_passed"] = all(
        r["passed"] for r in result["performance"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                r["function"]: dict(
                    passed=r["passed"],
                    stable=r["stable"],
                    gains=[x["geometric_mean_speedup"] for x in r["rounds"]],
                )
                for r in result["performance"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
