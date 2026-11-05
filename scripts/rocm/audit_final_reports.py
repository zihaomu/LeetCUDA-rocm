"""Audit the final checked-in qualification against reports and current sources."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/rocm"
FULL = DOCS / "unified_acceptance_20261106/full"
B1 = DOCS / "flash_swizzle_final_20261106"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(FULL / "manifest.json")
    assert manifest["passed"] and len(manifest["suites"]) == 23
    for row in manifest["suites"]:
        assert row["passed"] and row["returncode"] == 0
        if "report" in row:
            path = FULL / (
                "cpp/manifest.json"
                if row["name"] == "standalone_cpp"
                else row["name"] + ".json"
            )
            assert read(path)["passed"]
            assert sha(path) == row["report_sha256"], path

    original = read(FULL / "original229.json")
    counts = {
        "original229": sum(
            len(x["checks"]) for x in original["modules"].values()
        ),
        "matrix_attention": len(
            read(FULL / "matrix_attention.json")["correctness"]
        ),
    }
    for name in ("fp8_e4m3", "fp8_e5m2", "fp4", "triton", "training", "ffpa"):
        counts[name] = len(read(FULL / (name + ".json"))["checks"])
    for name, fields in {
        "interview_attention": (
            "checks",
            "pipeline",
            "boundaries",
            "layout",
            "memory",
        ),
        "supplemental": ("checks", "boundaries"),
        "cute_dsl": ("checks", "boundaries", "pipeline"),
        "swizzle": ("checks", "boundaries", "layout"),
        "unbound_attention": ("checks", "storage", "boundaries"),
        "unbound_gemm": ("checks", "boundaries", "pipeline"),
    }.items():
        counts[name] = sum(
            read(FULL / (name + ".json"))["counts"][f] for f in fields
        )
    cpp = read(FULL / "cpp/manifest.json")
    for row in cpp["suites"]:
        assert row["passed"] and row["returncode"] == 0
        if "report" in row:
            assert (
                sha(FULL / "cpp" / Path(row["report"]).name)
                == row["report_sha256"]
            )
        else:
            detail = read(FULL / "cpp" / (row["name"] + ".json"))
            assert all(detail[k] == v for k, v in row.items())
            assert "libtorch" not in row["linked_libraries"]
    counts["standalone_cpp"] = sum(x["checks"] for x in cpp["suites"])
    assert counts == read(FULL / "summary.json")["checks_by_group"]
    assert sum(counts.values()) == 15163
    for field in ("source_files_sha256", "adapter_sha256"):
        for path, digest in original[field].items():
            assert sha(ROOT / path) == digest, path

    cross = read(FULL / "qualified229_final_comparison.json")
    assert not cross["missing_cuda"] and not cross["missing_hip"]
    assert not any(cross["errors"])
    for row in cross["summary"].values():
        assert (
            row["cuda_passed"]
            == row["hip_passed"]
            == row["parity_within_gate"]
            == row["cases"]
        )
    assert sum(x["cases"] for x in cross["summary"].values()) == 462

    acceptance = read(B1 / "acceptance.json")
    assert acceptance["correctness_passed"] and acceptance["performance_passed"]
    assert len(acceptance["performance"]) == 3
    assert acceptance["input_report_sha256"] == sha(B1 / "hip.json")
    report = read(B1 / "hip.json")
    for item in report["performance_datasets"].values():
        if isinstance(item, dict):
            assert sha(B1 / item["report"]) == item["sha256"]
    for name in (
        "frozen_final.json",
        "frozen_qk_retest.json",
        "hip_second.json",
    ):
        data = read(B1 / name)
        assert data["passed"]
        assert (
            data["builds"]["hip"]["binary_sha256"]
            == data["isa"]["binary_sha256"]
        )
        assert "/frozen/" in data["isa"]["build_cache_binary"]
        for path, digest in data["builds"]["hip"]["source_sha256"].items():
            assert sha(ROOT / path) == digest, path
    regression = read(B1 / "existing_regression.json")
    assert regression["passed"] and regression["performance_no_regression"]
    assert regression["current_binary_sha256"] == report["isa"]["binary_sha256"]
    assert read(B1 / "cross.json")["passed"]
    audit = read(FULL / "immutable_loader_audit.json")
    assert audit["passed"] and audit["dispatcher_in_frozen_binary"]
    for path, digest in audit["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    inventory = read(DOCS / "leetcuda_rocm10_operator_inventory.json")
    assert inventory["summary"]["qualified"] == 229
    assert len(inventory["additional_qualified_amd_interfaces"]) == 9
    assert inventory["evidence_sha256"] == sha(FULL / "original229.json")
    print(
        json.dumps(
            dict(
                passed=True,
                functional_checks=sum(counts.values()),
                cross_cases=462,
                original_interfaces=229,
                b1_performance_entries=3,
                source_hashes="matched",
            )
        )
    )


if __name__ == "__main__":
    main()
