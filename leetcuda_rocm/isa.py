"""Extract the actual gfx1201 code object embedded in a HIP extension."""

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

from .environment import ROOT


def hip_isa(binary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    llvm = Path("/opt/rocm/llvm/bin")
    fatbin, code, assembly = (
        output / n for n in ("fatbin", "kernel.hsaco", "kernel.amdgcn")
    )
    binary_hash = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    # Omitting objcopy's output filename rewrites its input in place, even when
    # only dumping a section. Direct the rewritten ELF to a temporary file.
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        subprocess.run(
            [
                str(llvm / "llvm-objcopy"),
                "--dump-section",
                f".hip_fatbin={fatbin}",
                str(binary),
                str(Path(scratch) / "discard.elf"),
            ],
            check=True,
        )
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest() == binary_hash
    targets = subprocess.check_output(
        [
            str(llvm / "clang-offload-bundler"),
            "--list",
            "--type=o",
            f"--input={fatbin}",
        ],
        text=True,
    ).splitlines()
    (target,) = [
        t for t in targets if t.startswith("hip-") and t.endswith("--gfx1201")
    ]
    subprocess.run(
        [
            str(llvm / "clang-offload-bundler"),
            "--unbundle",
            "--type=o",
            f"--input={fatbin}",
            f"--output={code}",
            f"--targets={target}",
        ],
        check=True,
    )
    text = subprocess.check_output(
        [str(llvm / "llvm-objdump"), "-d", str(code)], text=True
    )
    assembly.write_text(text)
    return {
        "target": target,
        "instructions": sorted(set(re.findall(r"v_wmma_[a-z0-9_]+", text))),
        "assembly": str(assembly.relative_to(ROOT)),
        "assembly_sha256": hashlib.sha256(assembly.read_bytes()).hexdigest(),
        "hsaco_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
    }
