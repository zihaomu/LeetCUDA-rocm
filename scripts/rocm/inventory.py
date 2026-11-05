"""Rebuild the original-interface inventory from a complete passing report."""

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
workspace = root.parent
sys.path.insert(0, str(root))
from leetcuda_rocm.loader import KERNELS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--qualification", type=Path, required=True)
parser.add_argument(
    "--output",
    type=Path,
    default=workspace / "doc/leetcuda_rocm10_operator_inventory.json",
)
args = parser.parse_args()
report_path = args.qualification.resolve()
report = json.loads(report_path.read_text())
assert set(report["modules"]) == set(
    KERNELS
), "Use a complete qualification, not a selected module report"
assert report["passed"], "Qualification must pass before publishing support"
assert all(
    result["passed"] and all(c["passed"] for c in result["checks"])
    for result in report["modules"].values()
)
qualified = {
    (module, check["function"])
    for module, result in report["modules"].items()
    for check in result["checks"]
    if check["passed"]
}
rejected = {
    (module, name)
    for module, entries in report["excluded_functions"].items()
    for name in entries
}
research_path = root / "docs/rocm/hgemm_research_20261103.json"
research = json.loads(research_path.read_text())
rejected.update(("hgemm", name) for name in research["failed_functions"])
gemv_research_path = root / "docs/rocm/hgemv_research_20261103.json"
gemv_research = json.loads(gemv_research_path.read_text())
rejected.update(("hgemv", name) for name in gemv_research["failed_functions"])
matrix_gemv_key = ("hgemv", "hgemv_tensor_core_cute")
matrix_gemv_path = root / "docs/rocm/hgemv_matrix_20261105.json"
if matrix_gemv_key in qualified:
    matrix_gemv = json.loads(matrix_gemv_path.read_text())
    assert matrix_gemv["passed"] and all(
        c["passed"] for c in matrix_gemv["checks"]
    )
    assert matrix_gemv["original_accumulator"] == "FP16"
    assert matrix_gemv["rocm_accumulator"] == "FP32"
    rejected.discard(matrix_gemv_key)
fp8_report_path = root / "docs/rocm/fp8_pipeline_20261103.json"
fp8_report = json.loads(fp8_report_path.read_text())
assert fp8_report["passed"] and all(c["passed"] for c in fp8_report["checks"])
fp4_report_path = root / "docs/rocm/fp4_pipeline_20261103.json"
fp4_report = json.loads(fp4_report_path.read_text())
assert fp4_report["passed"] and all(c["passed"] for c in fp4_report["checks"])
async_report_path = root / "docs/rocm/async_sgemm_20261103.json"
async_report = json.loads(async_report_path.read_text())
assert async_report["passed"] and all(
    c["passed"] for c in async_report["checks"]
)
lexical = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/'
)


def uncomment(source):
    return lexical.sub(
        lambda match: (
            re.sub(r"[^\n]", " ", match[0])
            if match[0].startswith(("//", "/*"))
            else match[0]
        ),
        source,
    )


def reason(module, name):
    if (module, name) in qualified:
        return "qualified", "validated"
    if (module, name) in rejected:
        return (
            "numerical_rejected",
            {
                "hgemm": "fp16_accumulation_error",
                "hgemv": "fp16_gemv_accuracy",
                "reduce": "bf16_accumulation_error",
            }[module],
        )
    if "cublas" in name:
        return "pending", "blas_backend_port"
    if module in ("flash_attn", "ws_hgemm") or any(
        word in name for word in ("cute", "wmma", "_mma_")
    ):
        return "pending", "matrix_isa_or_cute_rewrite"
    if name.endswith("_async"):
        return "pending", "async_copy_rewrite"
    if module in ("hgemm", "sgemm", "merge_attn_states"):
        return "pending", "ordinary_hip_candidate"
    raise AssertionError((module, name))


modules = []
entries = []
helpers = []
hashes = {}
for source in sorted((root / "kernels").rglob("*")):
    if source.suffix not in (".cu", ".cc", ".cpp", ".cuh", ".h", ".hpp"):
        continue
    content = uncomment(source.read_text())
    start = content.find("PYBIND11_MODULE(")
    if start < 0:
        continue
    opening = content.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (content[end] == "{") - (content[end] == "}")
        end += 1
    body = content[opening:end]
    assert set(re.findall(r"\b(TORCH_BINDING_\w+)\s*\(", body)) <= {
        "TORCH_BINDING_COMMON_EXTENSION"
    }
    matches = list(
        re.finditer(r"TORCH_BINDING_COMMON_EXTENSION\s*\(\s*(\w+)\s*\)", body)
    )
    matches += list(re.finditer(r'\bm\.def\(\s*"([^"]+)"', body))
    module = source.relative_to(root / "kernels").parts[0].replace("-", "_")
    if module == "openai_triton":
        module = "merge_attn_states"
    rel = str(source.relative_to(root))
    hashes[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
    seen = set()
    for match in matches:
        name = match[1]
        if name in seen:
            continue
        seen.add(name)
        line = content.count("\n", 0, opening + match.start()) + 1
        if name in ("init_cublas_handle", "destroy_cublas_handle"):
            helpers.append({"source": rel, "line": line, "function": name})
            continue
        status, category = reason(module, name)
        entries.append(
            {
                "module": module,
                "function": name,
                "source": rel,
                "line": line,
                "status": status,
                "reason": category,
            }
        )
        if (
            module == "sgemm"
            and name.endswith("_async")
            and status == "qualified"
        ):
            entries[-1]["implementation"] = {
                "mechanism": "AMD VGPR prefetch, explicit load-counter wait, LDS double buffering",
                "original_mechanism": "NVIDIA cp.async direct-to-shared copy",
                "numeric_semantics": "Original per-thread output layout and FP32 FMA K order",
                "evidence": str(async_report_path.relative_to(root)),
            }
        if (module, name) == matrix_gemv_key and status == "qualified":
            entries[-1]["implementation"] = {
                "mechanism": "AMD FP16-input WMMA with FP32 accumulation",
                "original_mechanism": "NVIDIA CuTe FP16 MMA accumulation",
                "original_accumulator": "FP16",
                "rocm_accumulator": "FP32",
                "precision_changed": True,
                "numeric_semantics": "Same FP16 I/O and GEMV operation; strengthened accumulator, not bitwise FP16 MMA emulation",
                "evidence": str(matrix_gemv_path.relative_to(root)),
                "evidence_sha256": hashlib.sha256(
                    matrix_gemv_path.read_bytes()
                ).hexdigest(),
            }
    current = [entry for entry in entries if entry["source"] == rel]
    counts = collections.Counter(entry["status"] for entry in current)
    modules.append(
        {
            "module": module,
            "source": rel,
            "total": len(current),
            "qualified": counts["qualified"],
            "pending": counts["pending"],
            "numerical_rejected": counts["numerical_rejected"],
            "qualified_percent": round(
                100 * counts["qualified"] / len(current), 2
            ),
        }
    )

observed = {
    (entry["module"], entry["function"])
    for entry in entries
    if entry["status"] == "qualified"
}
assert observed == qualified
assert {
    (entry["module"], entry["function"])
    for entry in entries
    if entry["status"] == "numerical_rejected"
} == rejected
assert len(entries) == len(
    {(entry["module"], entry["function"]) for entry in entries}
)

# CUDA coverage is an independent axis. Successful NVIDIA execution must never
# promote an AMD-pending or numerically rejected interface to qualified.
cross_dir = root / "docs/rocm/cross_5090_20261105"
cuda_coverage = {}
for filename, field, source_kind in (
    ("qualified157_cuda.json", "passed", "ported_worktree"),
    ("cuda_v2.json", "fp64_accuracy_passed", "upstream_original"),
    ("patched_hgemm.json", "fp64_accuracy_passed", "ported_worktree"),
):
    path = cross_dir / filename
    cross = json.loads(path.read_text())
    by_function = collections.defaultdict(list)
    for check in cross["checks"]:
        key = (check["module"], check["function"])
        # The matrix HGEMV's public FP32 implementation is covered by the
        # qualified report. Its old FP16 experiment remains historical.
        if filename == "cuda_v2.json" and key in qualified:
            continue
        by_function[key].append(check)
    for key, checks in by_function.items():
        cuda_coverage[key] = {
            "source_kind": source_kind,
            "cases": len(checks),
            "accuracy_passed_cases": sum(c[field] for c in checks),
            "all_cases_passed": all(c[field] for c in checks),
            "evidence": str(path.relative_to(root)),
            "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
pending_path = cross_dir / "pending56_cuda.json"
pending_cuda = json.loads(pending_path.read_text())
for group in pending_cuda["groups"]:
    assert group["phase"] == "finished" and not group["error"]
    for fn in group["group"]["functions"]:
        key = (group["group"]["module"], fn["name"])
        assert key not in cuda_coverage
        checks = [c for c in group["checks"] if c["function"] == fn["name"]]
        assert len(checks) == 3
        cuda_coverage[key] = {
            "source_kind": "upstream_original",
            "cases": len(checks),
            "accuracy_passed_cases": sum(c["passed"] for c in checks),
            "all_cases_passed": all(c["passed"] for c in checks),
            "evidence": str(pending_path.relative_to(root)),
            "evidence_sha256": hashlib.sha256(
                pending_path.read_bytes()
            ).hexdigest(),
        }
assert set(cuda_coverage) == {(e["module"], e["function"]) for e in entries}
for entry in entries:
    entry["cuda_comparison"] = cuda_coverage[
        (entry["module"], entry["function"])
    ]
counts = collections.Counter(entry["status"] for entry in entries)
reason_counts = collections.Counter(entry["reason"] for entry in entries)
fully_qualified = sum(row["qualified"] == row["total"] for row in modules)
partially_qualified = sum(
    0 < row["qualified"] < row["total"] for row in modules
)
unqualified = sum(row["qualified"] == 0 for row in modules)
snapshot = {
    "inventory_date": report["timestamp_utc"][:10],
    "scope": "Original compute entrypoints declared in PYBIND11_MODULE blocks under kernels/",
    "counting_rules": [
        "Strip comments; count TORCH_BINDING_COMMON_EXTENSION and literal m.def declarations.",
        "Deduplicate within each extension module; include all optional preprocessor branches.",
        "Exclude init_cublas_handle and destroy_cublas_handle; do not expand template/shape/dtype cases.",
        "Exclude unbound CUDA kernels, standalone C++ benchmarks, Triton/CuTe DSL entrypoints and third-party dependencies.",
        "Additional AMD compute interfaces and their quantization helpers are recorded separately, outside the original-export denominator.",
        "Pending portability categories are source-review estimates, not successful compilation or execution results.",
    ],
    "git_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "worktree_has_changes": bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        )
    ),
    "evidence": str(report_path.relative_to(root)),
    "evidence_timestamp_utc": report["timestamp_utc"],
    "evidence_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "research_evidence": str(research_path.relative_to(root)),
    "research_evidence_sha256": hashlib.sha256(
        research_path.read_bytes()
    ).hexdigest(),
    "gemv_research_evidence": str(gemv_research_path.relative_to(root)),
    "gemv_research_evidence_sha256": hashlib.sha256(
        gemv_research_path.read_bytes()
    ).hexdigest(),
    "fp8_evidence": str(fp8_report_path.relative_to(root)),
    "fp8_evidence_sha256": hashlib.sha256(
        fp8_report_path.read_bytes()
    ).hexdigest(),
    "fp4_evidence": str(fp4_report_path.relative_to(root)),
    "fp4_evidence_sha256": hashlib.sha256(
        fp4_report_path.read_bytes()
    ).hexdigest(),
    "async_sgemm_evidence": str(async_report_path.relative_to(root)),
    "async_sgemm_evidence_sha256": hashlib.sha256(
        async_report_path.read_bytes()
    ).hexdigest(),
    "source_sha256": hashes,
    "summary": {
        "total_modules": len(modules),
        "total_original_compute_entrypoints": len(entries),
        **counts,
        "qualified_percent": round(100 * counts["qualified"] / len(entries), 2),
        "fully_qualified_modules": fully_qualified,
        "partially_qualified_modules": partially_qualified,
        "no_qualified_exports_modules": unqualified,
        "reason_counts": dict(reason_counts),
        "cuda_tested_interfaces": len(cuda_coverage),
        "cuda_all_selected_cases_passed_interfaces": sum(
            c["all_cases_passed"] for c in cuda_coverage.values()
        ),
        "cuda_scope": "Selected cases with explicit original/patched source per entry; not whole-project qualification",
    },
    "excluded_helpers": helpers,
    "additional_qualified_amd_interfaces": [
        "leetcuda_rocm.gemm",
        "leetcuda_rocm.flash_attention",
        "leetcuda_rocm.fp8_gemm",
        "leetcuda_rocm.fp8_gemm_prepacked",
        "leetcuda_rocm.fp4_gemm",
        "leetcuda_rocm.fp4_gemm_prepacked",
    ],
    "additional_qualified_helpers": [
        "leetcuda_rocm.quantize_fp8_weight",
        "leetcuda_rocm.quantize_fp4_weight",
    ],
    "modules": modules,
    "entries": entries,
}
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(snapshot["summary"], ensure_ascii=False, indent=2))
print(output)
