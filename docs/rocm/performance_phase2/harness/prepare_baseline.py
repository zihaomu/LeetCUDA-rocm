"""Restore the frozen comparison sources without overwriting other evidence."""

import hashlib
import json
import shutil
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
TARGET = ROOT / "artifacts/rocm/performance_phase2/baseline"
manifest = json.loads((EVIDENCE / "manifest.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, digest in manifest["baseline_sources"].items():
    source, target = EVIDENCE / "baseline" / name, TARGET / name
    assert sha(source) == digest, source
    if target.exists():
        assert sha(target) == digest, target
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
receipt = TARGET / "manifest.json"
if not receipt.exists():
    receipt.write_text(
        json.dumps(
            dict(
                commit=manifest["baseline_commit"],
                files=manifest["baseline_sources"],
            ),
            indent=2,
        ) + "\n"
    )
print("Frozen baseline sources verified:", len(manifest["baseline_sources"]))
