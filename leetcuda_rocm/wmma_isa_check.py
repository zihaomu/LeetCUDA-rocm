"""Read-only audit of each compiled AMD WMMA tile/stage specialization."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .environment import ROOT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    expected = set()
    for name, (bm, bn, wm, wn, rm, rn, mode) in report["variants"].items():
        if mode == "global":
            expected.add("global")
            continue
        stages = (
            range(2, 5 if (wm, wn) == (4, 4) else 6)
            if "ring" in mode
            else [2 if mode == "prefetch2" else 1]
        )
        expected.update(
            (wm, wn, rm, rn, s, int(mode == "dynamic-ring")) for s in stages
        )
    path = ROOT / report["isa"]["assembly"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == report["isa"]["assembly_sha256"]
    sections = re.split(
        r"^([0-9a-f]+) <([^>]+)>:\n", path.read_text(), flags=re.M
    )
    rows, observed = [], set()
    pattern = r"wmma_tiledILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb([01])EEE"
    for index in range(1, len(sections), 3):
        symbol, body = sections[index + 1 : index + 3]
        if "wmma_" not in symbol:
            continue
        match = re.search(pattern, symbol)
        key = tuple(map(int, match.groups())) if match else "global"
        assert key in expected and key not in observed, symbol
        observed.add(key)
        overlaps = 0
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if "global_load_b128" not in line:
                continue
            for later in lines[i + 1 :]:
                if "s_wait_loadcnt" in later:
                    break
                if "v_wmma_" in later:
                    overlaps += 1
                    break
        needs_prefetch = key != "global" and key[4] > 1
        row = dict(
            symbol=symbol,
            specialization=key,
            wmma_instructions=body.count("v_wmma_f32_16x16x16_f16"),
            scratch_instructions=body.count("scratch_"),
            load_before_wmma_before_wait_sites=overlaps,
        )
        row["passed"] = (
            row["wmma_instructions"] > 0
            and row["scratch_instructions"] == 0
            and (not needs_prefetch or overlaps > 0)
        )
        rows.append(row)
    assert observed == expected
    result = dict(
        binary_sha256=report["isa"]["binary_sha256"],
        assembly_sha256=digest,
        source_sha256=report["builds"]["hip"]["source_sha256"],
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Static linear instruction order; not a cycle-level overlap measurement. Stage-2 static ring and fixed double buffer intentionally share a specialization.",
        kernels=rows,
        passed=all(r["passed"] for r in rows),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], specializations=len(rows))))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
