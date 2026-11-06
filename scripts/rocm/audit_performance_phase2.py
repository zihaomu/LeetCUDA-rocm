"""Audit the frozen second-phase evidence and the four current source files."""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/rocm/performance_phase2"
SOURCES = {
    "kernels/rocm/interview_flash_attention_hip.cuh",
    "kernels/rocm/fp8_gemm_hip.cuh",
    "kernels/rocm/ffpa_attention_hip.cuh",
    "leetcuda_rocm/attention_training.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def performance(report, count, key, target):
    assert report["passed"]
    assert all(row["passed"] for row in report["checks"])
    rows = report["performance"]
    assert len(rows) == 3 * count
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
        for side in ("ours", "baseline"):
            samples = row["samples_ms"][side]
            assert len(samples) == 5
            assert all(math.isfinite(x) and x > 0 for x in samples)
            assert math.isclose(
                row[side + "_ms"], statistics.median(samples), rel_tol=1e-10
            )
        assert math.isclose(
            row["baseline_over_ours"],
            row["baseline_ms"] / row["ours_ms"],
            rel_tol=1e-10,
        )
        assert row["ours_ms"] <= 1.1 * row["baseline_ms"], key(row)
    assert len(groups) == count
    maximum_spread = 0
    for identity, group in groups.items():
        assert sorted(row["round"] for row in group) == [0, 1, 2]
        for side in ("ours_ms", "baseline_ms"):
            values = [row[side] for row in group]
            spread = (max(values) - min(values)) / statistics.median(values)
            assert spread <= 0.05, (identity, side, spread)
            maximum_spread = max(maximum_spread, spread)
    selected = [row for row in rows if target(row)]
    gain = math.exp(
        statistics.mean(math.log(row["baseline_over_ours"]) for row in selected)
    )
    assert gain >= 1.05, gain
    return dict(
        configurations=count,
        target_configurations=len(selected) // 3,
        target_geomean_speedup=gain,
        worst_slowdown=max(row["ours_ms"] / row["baseline_ms"] for row in rows),
        maximum_round_spread=maximum_spread,
    )


def main(quiet=False):
    manifest = read(EVIDENCE / "manifest.json")
    assert set(manifest["current_sources"]) == SOURCES
    assert set(manifest["baseline_sources"]) == SOURCES
    for name, digest in manifest["current_sources"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in manifest["baseline_sources"].items():
        assert sha(EVIDENCE / "baseline" / name) == digest, name
    for name, digest in manifest["evidence_sha256"].items():
        assert sha(EVIDENCE / name) == digest, name
    for name, expected in manifest["correctness"].items():
        report = read(EVIDENCE / name)
        assert report["passed"], name
        fields = expected["fields"]
        assert sum(len(report[field]) for field in fields) == expected["count"]
        assert all(row["passed"] for field in fields for row in report[field])
        if "source" in expected:
            digest = report
            for component in expected["digest_path"]:
                digest = digest[component]
            assert (
                digest == manifest["current_sources"][expected["source"]]
            ), name

    results = {}
    for label, count, key, target, source in (
        (
            "fa",
            32,
            lambda r: (r["variant"], r["d"], r["n"]),
            lambda r: True,
            "kernels/rocm/interview_flash_attention_hip.cuh",
        ),
        (
            "fp8",
            108,
            lambda r: (r["format"], *r["shape"], r["mode"], r["ws"], r["path"]),
            lambda r: r["path"] == "full",
            "kernels/rocm/fp8_gemm_hip.cuh",
        ),
        (
            "ffpa",
            135,
            lambda r: (r["variant"], r["d"], r["sq"], r["sv"]),
            lambda r: r["variant"] == "ws",
            "kernels/rocm/ffpa_attention_hip.cuh",
        ),
        (
            "training",
            12,
            lambda r: (r["dtype"], r["n"], r["d"], r["path"]),
            lambda r: r["d"] == 256 and r["path"] == "forward_backward",
            "leetcuda_rocm/attention_training.py",
        ),
    ):
        report = read(EVIDENCE / label / "performance.json")
        hashes = (
            report["build"]["source_sha256"]
            if label == "fa"
            else report["source_sha256"]
        )
        assert manifest["current_sources"][source] in hashes.values(), source
        results[label] = performance(report, count, key, target)

    kernels = read(EVIDENCE / "fa/resources.json")["kernels"]
    production = [k for k in kernels if not k["audit"]]
    assert len(production) == 16
    assert sum(k["spill"] > 0 for k in production) == 13
    assert max(k["scratch"] for k in production) == 752
    assert all(k["lds"] <= 65536 for k in kernels)
    ffpa = read(EVIDENCE / "ffpa/primary.json")
    assert len(ffpa["isa"]["kernels"]) == 27
    assert all(k["spill"] == k["scratch"] == 0 for k in ffpa["isa"]["kernels"])
    dispatch = read(EVIDENCE / "fp8/dispatch_sweep.json")
    assert dispatch["passed"] and not dispatch["regressions_gt_10_percent"]
    assert len(dispatch["performance"]) == 80
    assert (
        dispatch["source_sha256"]
        == manifest["current_sources"]["kernels/rocm/fp8_gemm_hip.cuh"]
    )
    for name in ("fa", "fp8", "ffpa", "training"):
        assert read(EVIDENCE / name / "cross.json")["passed"]
    manifest["audited_performance"] = results
    if not quiet:
        print(json.dumps(dict(passed=True, performance=results)))
    return manifest


if __name__ == "__main__":
    main()
