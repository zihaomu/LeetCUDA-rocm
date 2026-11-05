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
wmma_path = root / "docs/rocm/wmma_hgemm_20261105.json"
wmma_keys = {
    key
    for key in qualified
    if key[0] == "hgemm" and key[1].startswith("hgemm_wmma_")
}
if wmma_keys:
    wmma_report = json.loads(wmma_path.read_text())
    assert wmma_report["passed"]
    assert all(
        c["passed"] for c in wmma_report["checks"] + wmma_report["boundaries"]
    )
    assert all(
        p["graph_replay_verified"] and p["accuracy_passed"]
        for p in wmma_report["performance"]
    )
    assert wmma_keys == {("hgemm", n) for n in wmma_report["variants"]}
    assert wmma_report["original_accumulator"] == "FP16"
    assert wmma_report["rocm_accumulator"] == "FP32"
mma_path = root / "docs/rocm/mma_hgemm_20261105/hip.json"
mma_keys = {
    key
    for key in qualified
    if key[0] == "hgemm" and key[1].startswith("hgemm_mma_")
}
if mma_keys:
    mma_report = json.loads(mma_path.read_text())
    assert mma_report["passed"] and all(
        c["passed"] for c in mma_report["checks"] + mma_report["boundaries"]
    )
    assert mma_keys == {("hgemm", n) for n in mma_report["variants"]}
    assert (
        mma_report["original_accumulator"] == "FP16"
        and mma_report["rocm_accumulator"] == "FP32"
    )
    assert len(mma_report["isa"]["kernels"]) == 36
    assert all(
        k["wmma"] and k["lds_bytes"] <= 65536
        for k in mma_report["isa"]["kernels"]
    )
    assert all(
        p["accuracy_passed"] and p["graph_replay_verified"]
        for p in mma_report["performance"]
    )
fp8_report_path = root / "docs/rocm/fp8_pipeline_20261103.json"
flash_keys = {
    key
    for key in qualified
    if key[0] == "flash_attn"
    and "_shared_" in key[1]
    and "_acc_f32" in key[1]
    and not key[1].endswith("_rr")
}
rr_keys = {
    key
    for key in qualified
    if key[0] == "flash_attn" and key[1].endswith("_acc_f32_rr")
}
rr_path = root / "docs/rocm/flash_rr_20261105/hip.json"
if rr_keys:
    rr_report = json.loads(rr_path.read_text())
    assert rr_report["passed"]
    assert rr_keys == {
        ("flash_attn", c["function"]) for c in rr_report["checks"]
    }
    for field in ("checks", "boundaries", "storage_checks", "device_checks"):
        assert rr_report[field] and all(c["passed"] for c in rr_report[field])
    assert len(rr_report["storage_checks"]) == 19
    rr_kernels = [k for k in rr_report["isa"]["kernels"] if k["rr"]]
    assert len(rr_kernels) == 19 and all(
        k["wmma"] and k["dynamic_lds_bytes"] <= 65536 for k in rr_kernels
    )
    assert len(rr_report["performance"]) == 69
    assert all(
        p["accuracy_passed"] and p["graph_replay_verified"]
        for p in rr_report["performance"]
    )
tiled_keys = {
    key
    for key in qualified
    if key[0] == "flash_attn"
    and "_tiling_" in key[1]
    and "_acc_f32" in key[1]
    and "_swizzle_" not in key[1]
}
swizzle_keys = {
    key
    for key in qualified
    if key[0] == "flash_attn" and "_acc_f32_swizzle_" in key[1]
}
low_keys = {
    key
    for key in qualified
    if key[0] == "flash_attn" and "_acc_f32" not in key[1]
}
low_path = root / "docs/rocm/flash_low_20261105/hip.json"
if low_keys:
    low_report = json.loads(low_path.read_text())
    assert low_report["passed"] and low_report["isa"]["passed"]
    assert low_keys == {
        ("flash_attn", c["function"]) for c in low_report["checks"]
    }
    for field in ("checks", "boundaries", "storage", "devices"):
        assert low_report[field] and all(c["passed"] for c in low_report[field])
    assert len(low_report["isa"]["kernels"]) == 210
    assert all(
        p["accuracy_passed"] and p["graph_replay_verified"]
        for p in low_report["performance"]
    )
    low_contracts = {
        c["function"]: c
        for c in json.loads(
            (root / "leetcuda_rocm/flash_low_contracts.json").read_text()
        )
    }

swizzle_path = root / "docs/rocm/flash_swizzle_20261105/hip.json"
swizzle_performance_path = (
    root / "docs/rocm/flash_swizzle_20261105/acceptance.json"
)
if swizzle_keys:
    swizzle_report = json.loads(swizzle_path.read_text())
    assert swizzle_report["passed"]
    assert swizzle_keys == {
        ("flash_attn", c["function"]) for c in swizzle_report["checks"]
    }
    assert len(swizzle_report["checks"]) == 372
    assert len(swizzle_report["storage_checks"]) == 42
    assert len(swizzle_report["isa"]["kernels"]) == 42
    for field in (
        "checks",
        "boundaries",
        "storage_checks",
        "device_checks",
        "layout_checks",
    ):
        assert swizzle_report[field] and all(
            c["passed"] for c in swizzle_report[field]
        )
    assert all(k["wmma"] > 0 for k in swizzle_report["isa"]["kernels"])
    swizzle_performance = json.loads(swizzle_performance_path.read_text())
    assert (
        swizzle_performance["input_report_sha256"]
        == hashlib.sha256(swizzle_path.read_bytes()).hexdigest()
    )
    swizzle_performance_by_name = {
        row["function"]: row for row in swizzle_performance["performance"]
    }
flash_path = root / "docs/rocm/flash_shared_20261105/hip.json"
if flash_keys:
    flash_report = json.loads(flash_path.read_text())
    assert flash_report["passed"]
    assert all(
        c["passed"] for c in flash_report["checks"] + flash_report["boundaries"]
    )
    assert flash_keys == {
        ("flash_attn", c["function"]) for c in flash_report["checks"]
    }
    assert len(flash_report["isa"]["kernels"]) == 18
    assert all(k["wmma"] > 0 for k in flash_report["isa"]["kernels"])
tiled_path = root / "docs/rocm/flash_tiled_20261105/hip.json"
if tiled_keys:
    tiled_report = json.loads(tiled_path.read_text())
    assert tiled_report["passed"]
    assert all(
        c["passed"]
        for c in tiled_report["checks"]
        + tiled_report["boundaries"]
        + tiled_report["storage_checks"]
    )
    assert tiled_keys == {
        ("flash_attn", c["function"]) for c in tiled_report["checks"]
    }
    assert len(tiled_report["storage_checks"]) == 28
    assert all(
        c["distinguishes_fp16_state"]
        for c in tiled_report["storage_checks"]
        if c["output_state"] == "FP16"
    )
    assert len(tiled_report["isa"]["kernels"]) == 28
    assert all(
        k["wmma"] > 0 and k["scratch"] == 0
        for k in tiled_report["isa"]["kernels"]
    )
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
        if (module, name) in wmma_keys:
            entries[-1]["implementation"] = {
                "mechanism": "Native gfx1201 WMMA; variant-specific block/warp tiles, LDS ring and grid swizzle",
                "original_accumulator": "FP16",
                "rocm_accumulator": "FP32",
                "precision_changed": True,
                "variant_geometry": wmma_report["variants"][name],
                "cp_async_replacement": "VGPR prefetch, explicit load wait, LDS publication",
                "numeric_semantics": "FP16 I/O, FP32 state; not bitwise FP16 accumulation emulation",
                "evidence": str(wmma_path.relative_to(root)),
                "evidence_sha256": hashlib.sha256(
                    wmma_path.read_bytes()
                ).hexdigest(),
            }
        if (module, name) in mma_keys:
            entries[-1]["implementation"] = {
                "mechanism": "Native AMD WMMA with variant-specific K16/K32 staging, static/dynamic LDS, paired register fragments, wide LDS reads, rr lifetimes, XOR layouts and packed TN",
                "original_accumulator": "FP16",
                "rocm_accumulator": "FP32",
                "precision_changed": True,
                "numeric_semantics": "FP16 I/O with explicitly upgraded FP32 state; no claim of FP16 MMA bitwise equivalence",
                "variant_mode": mma_report["variants"][name],
                "architecture_adaptation": "K32 stage5 uses64x128; CuTe stage2 uses128x256, stage3/4 uses128x128; full requested stage depth retained within64KiB",
                "evidence": str(mma_path.relative_to(root)),
                "evidence_sha256": hashlib.sha256(
                    mma_path.read_bytes()
                ).hexdigest(),
                "performance_status": "measured_pending_optimization",
            }
        if (module, name) in low_keys:
            cute = name == "flash_attn_cute"
            vt = name.endswith("_swizzle_qkv") and "tiling_qkv_" not in name
            entries[-1]["implementation"] = {
                "mechanism": "Native gfx1201 WMMA; distinct split-Q/split-KV, shared LDS lifetimes, Q/K/V XOR layouts, tiling, Os2g LDS output and CuTe geometry",
                "original_accumulator": "FP32 QK/PV" if cute else "FP16 QK/PV",
                "rocm_accumulator": (
                    "FP32 QK/PV" if cute else "FP32 QK; FP16 PV"
                ),
                "precision_changed": not cute,
                "numeric_semantics": "Explicit QK-only precision upgrade for 19 MMA interfaces; preserve FP16 PV and original FP16/FP32 output-state type; CuTe unchanged FP32",
                "v_storage_layout": (
                    "contiguous [B,H,D,N]" if vt else "contiguous [B,H,N,D]"
                ),
                "head_dimensions_by_stage": low_contracts[name]["dimensions"],
                "architecture_adaptation": "D256 shared BR64; transposed-V tiling-QK keeps original BC=D with BR32 at D256; LDS P exchange and guarded tails",
                "performance_status": "measured_pending_optimization",
                "evidence": str(low_path.relative_to(root)),
                "evidence_sha256": hashlib.sha256(
                    low_path.read_bytes()
                ).hexdigest(),
            }
        if (module, name) in flash_keys:
            entries[-1]["implementation"] = {
                "mechanism": "Native gfx1201 split-Q WMMA attention; distinct shared-KV or shared-QKV LDS lifetime",
                "original_accumulator": "FP32",
                "rocm_accumulator": "FP32",
                "precision_changed": False,
                "numeric_semantics": "FP16 Q/K/V/P/O; FP32 QK/PV, online max/sum and output state; no bitwise guarantee",
                "stages": "Same upstream dispatch: <=1 maps to 1, >1 maps to 2; stage 2 prefetches V during QK",
                "head_dimensions": "32,64,96,128 at stages 1/2; 256 only at stage 1",
                "architecture_adaptation": "D=256 query tile reduced from 128 to 64 rows to fit 64 KiB LDS; FP16 P is exchanged through LDS",
                "evidence": str(flash_path.relative_to(root)),
                "evidence_sha256": hashlib.sha256(
                    flash_path.read_bytes()
                ).hexdigest(),
            }
        if (module, name) in rr_keys:
            entries[-1]["implementation"] = {
                "mechanism": "Native gfx1201 WMMA; two cooperating output warp groups at D>64 share Q/K/V loads, halving live output state per thread",
                "original_accumulator": "FP32",
                "rocm_accumulator": "FP32",
                "precision_changed": False,
                "numeric_semantics": "FP16 Q/K/V/P/O; FP32 QK/PV, online max/sum and output state",
                "head_dimensions": "32,64,96,128,256; stages1/2 except KV D256 only stage1",
                "architecture_adaptation": "D256 BR64; QKV D256/stage2 BC32 to fit 64KiB LDS; D<=64 retains single group; QK is repeated per output group",
                "performance_status": "measured_pending_optimization",
                "evidence": str(rr_path.relative_to(root)),
                "evidence_sha256": hashlib.sha256(
                    rr_path.read_bytes()
                ).hexdigest(),
            }
        if (module, name) in tiled_keys:
            entries[-1]["implementation"] = {
                "mechanism": "Native gfx1201 split-Q WMMA attention; distinct reduction-striped V (tiling-QK) or output-striped V (tiling-QKV)",
                "original_accumulator": "FP32 QK/PV; output state FP32 at D<256, FP16 at D>=256",
                "rocm_accumulator": "FP32 QK/PV; output state FP32 at D<256, FP16 at D>=256",
                "precision_changed": False,
                "numeric_semantics": "FP16 Q/K/V/P/O; FP32 online max/sum; running output rounded to FP16 after every key block at D>=256; no bitwise guarantee",
                "stages": "Same upstream dispatch: <=1 maps to 1, >1 maps to 2; stage 2 uses VGPR prefetch and two LDS slots",
                "head_dimensions": "32,64,96,128,256,512,1024 at stages 1/2",
                "architecture_adaptation": "Preserve original query/key tiles (64 below D=128, otherwise 128); partition output into up to 128 columns per CTA, recomputing QK for each output partition; at most 48 KiB LDS",
                "evidence": str(tiled_path.relative_to(root)),
                "evidence_sha256": hashlib.sha256(
                    tiled_path.read_bytes()
                ).hexdigest(),
            }
        if (module, name) in swizzle_keys:
            entries[-1]["implementation"] = {
                "mechanism": "Native gfx1201 tiled WMMA attention with distinct Q, QK or QKV XOR LDS permutations",
                "original_accumulator": "FP32 QK/PV; output state FP32 at D<256, FP16 at D>=256",
                "rocm_accumulator": "FP32 QK/PV; output state FP32 at D<256, FP16 at D>=256",
                "precision_changed": False,
                "lds_mapping": "row*Cols + (col XOR (((row >> 2) & 1)<<3)); V mapping applied after transposition",
                "swizzled_operands": name.rsplit("_", 1)[1].upper(),
                "explicit_lds_vector_bits": 128,
                "head_dimensions": "32,64,96,128,256,512,1024 at stages 1/2",
                "architecture_adaptation": "128-column output partitions above D=128 repeat QK; at most 48 KiB LDS; original state rounding frequency preserved",
                "evidence": str(swizzle_path.relative_to(root)),
                "evidence_sha256": hashlib.sha256(
                    swizzle_path.read_bytes()
                ).hexdigest(),
                "performance_status": (
                    "qualified"
                    if swizzle_performance_by_name[name]["passed"]
                    else "pending_optimization"
                ),
                "performance_evidence": str(
                    swizzle_performance_path.relative_to(root)
                ),
                "performance_evidence_sha256": hashlib.sha256(
                    swizzle_performance_path.read_bytes()
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
if wmma_keys:
    qualified_cuda_path = root / (
        "docs/rocm/flash_tiled_20261105/qualified170_cuda.json"
        if tiled_keys
        else (
            "docs/rocm/flash_shared_20261105/qualified168_cuda.json"
            if flash_keys
            else "docs/rocm/wmma_hgemm_20261105/qualified166_cuda.json"
        )
    )
    if swizzle_keys:
        qualified_cuda_path = (
            root / "docs/rocm/flash_swizzle_20261105/qualified173_cuda.json"
        )
    if rr_keys:
        qualified_cuda_path = (
            root / "docs/rocm/flash_rr_20261105/qualified175_cuda.json"
        )
    if mma_keys:
        qualified_cuda_path = (
            root / "docs/rocm/mma_hgemm_20261105/qualified186_cuda.json"
        )
    if low_keys:
        qualified_cuda_path = (
            root / "docs/rocm/flash_low_20261105/qualified206_cuda.json"
        )
    qualified_cuda = json.loads(qualified_cuda_path.read_text())
    assert qualified_cuda["passed"]
    by_function = collections.defaultdict(list)
    for check in qualified_cuda["checks"]:
        by_function[(check["module"], check["function"])].append(check)
    assert set(by_function) <= qualified
    for key, checks in by_function.items():
        cuda_coverage[key] = {
            "source_kind": "ported_worktree",
            "cases": len(checks),
            "accuracy_passed_cases": sum(c["passed"] for c in checks),
            "all_cases_passed": all(c["passed"] for c in checks),
            "evidence": str(qualified_cuda_path.relative_to(root)),
            "evidence_sha256": hashlib.sha256(
                qualified_cuda_path.read_bytes()
            ).hexdigest(),
        }
    # Preserve the stronger long-K findings instead of replacing them with
    # the smaller cases in the all-interface regression.
    wmma_cuda_path = root / "docs/rocm/wmma_hgemm_20261105/cuda_fixed.json"
    wmma_cuda = json.loads(wmma_cuda_path.read_text())
    for key in wmma_keys:
        checks = [
            c for c in wmma_cuda["checks"] if c["id"].startswith(key[1] + "/")
        ]
        assert checks
        cuda_coverage[key] = {
            "source_kind": "ported_worktree_with_current_stream_launches",
            "cases": len(checks),
            "accuracy_passed_cases": sum(c["passed"] for c in checks),
            "all_cases_passed": all(c["passed"] for c in checks),
            "scope": "Expanded K=4096 coverage reveals original FP16 accumulation error; AMD explicitly uses FP32 state",
            "evidence": str(wmma_cuda_path.relative_to(root)),
            "evidence_sha256": hashlib.sha256(
                wmma_cuda_path.read_bytes()
            ).hexdigest(),
        }
if mma_keys:
    mma_cuda_path = root / "docs/rocm/mma_hgemm_20261105/cuda_original.json"
    mma_cuda = json.loads(mma_cuda_path.read_text())
    for key in mma_keys:
        checks = [
            c for c in mma_cuda["checks"] if c["id"].startswith(key[1] + "/")
        ]
        assert checks
        cuda_coverage[key] = {
            "source_kind": "upstream_original_with_current_stream_launches",
            "cases": len(checks),
            "accuracy_passed_cases": sum(c["passed"] for c in checks),
            "resource_rejected_cases": sum(
                not c.get("executed", True) for c in checks
            ),
            "all_cases_passed": all(c["passed"] for c in checks),
            "scope": "Expanded K4096 and stage5 reveal original FP16 numerical limitations and oversized LDS allocations; AMD explicitly upgrades precision and adapts geometry",
            "evidence": str(mma_cuda_path.relative_to(root)),
            "evidence_sha256": hashlib.sha256(
                mma_cuda_path.read_bytes()
            ).hexdigest(),
        }
for keys, flash_cuda_path in (
    (flash_keys, root / "docs/rocm/flash_shared_20261105/cuda.json"),
    (tiled_keys, root / "docs/rocm/flash_tiled_20261105/cuda.json"),
    (swizzle_keys, root / "docs/rocm/flash_swizzle_20261105/cuda.json"),
    (rr_keys, root / "docs/rocm/flash_rr_20261105/cuda.json"),
):
    if not keys:
        continue
    flash_cuda = json.loads(flash_cuda_path.read_text())
    assert flash_cuda["passed"]
    for key in keys:
        checks = [c for c in flash_cuda["checks"] if c["function"] == key[1]]
        assert checks
        cuda_coverage[key] = {
            "source_kind": "upstream_original_with_current_stream_launches",
            "cases": len(checks),
            "accuracy_passed_cases": sum(c["passed"] for c in checks),
            "all_cases_passed": all(c["passed"] for c in checks),
            "evidence": str(flash_cuda_path.relative_to(root)),
            "evidence_sha256": hashlib.sha256(
                flash_cuda_path.read_bytes()
            ).hexdigest(),
        }
if low_keys:
    low_cuda_path = root / "docs/rocm/flash_low_20261105/cuda_original.json"
    low_cuda = json.loads(low_cuda_path.read_text())
    for key in low_keys:
        checks = [c for c in low_cuda["checks"] if c["function"] == key[1]]
        excluded = [
            c for c in low_cuda["not_executed"] if c["function"] == key[1]
        ]
        assert checks
        cuda_coverage[key] = {
            "source_kind": "upstream_original_with_current_stream_launches",
            "cases": len(checks),
            "accuracy_passed_cases": sum(c["passed"] for c in checks),
            "all_cases_passed": all(c["passed"] for c in checks)
            and not excluded,
            "not_executed_cases": len(excluded),
            "scope": "Preserve large-logit FP16 QK failures; two shared swizzle-QKV D96 indexing defects; tiling-QK swizzle-QKV requires N%D==0. Transposed V contract honored.",
            "evidence": str(low_cuda_path.relative_to(root)),
            "evidence_sha256": hashlib.sha256(
                low_cuda_path.read_bytes()
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
        "qualified_with_explicit_accumulator_upgrade": sum(
            e.get("implementation", {}).get("precision_changed", False)
            for e in entries
        ),
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
