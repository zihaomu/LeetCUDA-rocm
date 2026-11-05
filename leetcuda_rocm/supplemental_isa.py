"""Freeze code objects and report actual resources for supplemental examples."""

import argparse
import json
import re
import subprocess
from pathlib import Path
import yaml
from .environment import ROOT, verify
from .cross_validate import sha
from .isa_objects import hip_isa_objects


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite",
        choices=("supplemental", "cute_dsl", "swizzle"),
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    entries = []
    if args.suite == "supplemental":
        from .supplemental import build

        for group in ("profile", "vector", "transpose"):
            lib, meta = build(group)
            entries.append((group, lib, meta))
    elif args.suite == "cute_dsl":
        from .cute_dsl import _native, source_paths

        entries = [
            (
                "cute_dsl",
                _native(0),
                dict(
                    source_sha256={
                        str(p.relative_to(ROOT)): sha(p) for p in source_paths()
                    }
                ),
            )
        ]
    else:
        from .swizzle_examples import native, source_paths

        entries = [
            (
                "swizzle",
                native(),
                dict(
                    source_sha256={
                        str(p.relative_to(ROOT)): sha(p) for p in source_paths()
                    }
                ),
            )
        ]
    report = dict(
        environment=verify(), harness_sha256=sha(__file__), modules=[]
    )
    for name, lib, meta in entries:
        isa = hip_isa_objects(
            lib.__file__, (args.output.parent / (name + "_isa")).resolve()
        )
        resources = {}
        for obj in isa["objects"]:
            notes = subprocess.check_output(
                [
                    "/opt/rocm/llvm/bin/llvm-readobj",
                    "--notes",
                    str(ROOT / obj["hsaco"]),
                ],
                text=True,
            )
            for e in yaml.safe_load(
                notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0]
            )["amdhsa.kernels"]:
                resources[e[".name"]] = e
        chunks = re.split(
            r"^([0-9a-f]+) <([^>]+)>:\n",
            (ROOT / isa["assembly"]).read_text(),
            flags=re.M,
        )
        kernels = []
        for i in range(1, len(chunks), 3):
            symbol, body = chunks[i + 1 : i + 3]
            if symbol not in resources:
                continue
            e = resources[symbol]
            kernels.append(
                dict(
                    symbol=symbol,
                    vgpr=e[".vgpr_count"],
                    vgpr_spill=e[".vgpr_spill_count"],
                    sgpr_spill=e.get(".sgpr_spill_count"),
                    scratch=e[".private_segment_fixed_size"],
                    lds=e[".group_segment_fixed_size"],
                    wmma=body.count("v_wmma"),
                    fma32=body.count("v_fma_f32")
                    + body.count("v_fmac_f32")
                    + body.count("v_dual_fmac_f32"),
                    packed_fma16=body.count("v_pk_fma_f16"),
                    load128=body.count("global_load_b128"),
                    store128=body.count("global_store_b128"),
                )
            )
        report["modules"].append(
            dict(name=name, build=meta, isa=isa, kernels=kernels)
        )
        print(
            name,
            [(x["vgpr"], x["scratch"], x["vgpr_spill"]) for x in kernels],
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
