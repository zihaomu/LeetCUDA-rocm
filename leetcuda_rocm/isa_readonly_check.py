"""Check that ISA extraction preserves real binaries and earlier device code."""

import hashlib
import json
from datetime import datetime, timezone

import torch

from .environment import ROOT, verify
from .fp4 import _native as fp4_native
from .fp8 import _native as fp8_native
from .isa import hip_isa
from .isa_objects import hip_isa_objects


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.cuda.set_device(0)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": verify(),
        "checks": [],
        "passed": False,
        "source_sha256": {
            s: digest(ROOT / s)
            for s in (
                "leetcuda_rocm/isa.py",
                "leetcuda_rocm/isa_objects.py",
                "leetcuda_rocm/isa_readonly_check.py",
            )
        },
        "legacy_isa_source": {
            "path": "artifacts/rocm/isa/before_readonly/isa.py",
            "sha256": digest(
                ROOT / "artifacts/rocm/isa/before_readonly/isa.py"
            ),
        },
    }
    single = []
    for name, native in (("fp4", fp4_native), ("fp8", fp8_native)):
        prior_path = ROOT / f"docs/rocm/{name}_pipeline_20261103.json"
        prior = json.loads(prior_path.read_text())
        single.append(
            (name + "_torch", ROOT / native(0).__file__, prior_path, prior)
        )
        cpp_path = ROOT / f"docs/rocm/{name}_cpp_20261103.json"
        cpp = json.loads(cpp_path.read_text())
        single.append(
            (name + "_cpp", ROOT / cpp["build_command"][-1], cpp_path, cpp)
        )
    try:
        for name, binary, evidence, prior in single:
            before = digest(binary)
            isa = hip_isa(binary, ROOT / "artifacts/rocm/isa/readonly" / name)
            after = digest(binary)
            assert before == after, f"ISA inspection modified {name}"
            assert isa["hsaco_sha256"] == prior["isa"]["hsaco_sha256"], name
            report["checks"].append(
                {
                    "name": name,
                    "passed": True,
                    "binary_sha256_before": before,
                    "binary_sha256_after": after,
                    "isa": isa,
                    "prior_evidence": str(evidence.relative_to(ROOT)),
                    "prior_evidence_sha256": digest(evidence),
                    "device_code_unchanged": True,
                }
            )
        evidence = ROOT / "docs/rocm/async_sgemm_20261103.json"
        prior = json.loads(evidence.read_text())
        binary = ROOT / prior["isa"]["build_cache_binary"]
        before = digest(binary)
        isa = hip_isa_objects(
            binary, ROOT / "artifacts/rocm/isa/readonly/sgemm"
        )
        after = digest(binary)
        assert before == after
        assert [o["hsaco_sha256"] for o in isa["objects"]] == [
            o["hsaco_sha256"] for o in prior["isa"]["objects"]
        ]
        assert digest(ROOT / isa["binary"]) == isa["binary_sha256"] == before
        report["checks"].append(
            {
                "name": "sgemm_multi_object",
                "passed": True,
                "binary_sha256_before": before,
                "binary_sha256_after": after,
                "isa": isa,
                "prior_evidence": str(evidence.relative_to(ROOT)),
                "prior_evidence_sha256": digest(evidence),
                "device_code_unchanged": True,
            }
        )
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    output = ROOT / "docs/rocm/isa_readonly_20261103.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(report["checks"]),
                "error": report.get("error"),
            }
        )
    )
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
