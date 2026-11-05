"""Inspect every gfx1201 code object in a linked, multi-source HIP extension."""

import hashlib
import re
import struct
import subprocess
import tempfile
from pathlib import Path

from .environment import ROOT


def hip_isa_objects(binary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Torch can relink the same cache path in a later process. Keep the exact
    # inspected binary independently of that mutable path.
    binary_bytes = Path(binary).read_bytes()
    binary_hash = hashlib.sha256(binary_bytes).hexdigest()
    saved_binary = output.parent / "binaries" / f"{binary_hash}.so"
    saved_binary.parent.mkdir(parents=True, exist_ok=True)
    if not saved_binary.exists():
        saved_binary.write_bytes(binary_bytes)
    assert hashlib.sha256(saved_binary.read_bytes()).hexdigest() == binary_hash
    llvm = Path("/opt/rocm/llvm/bin")
    fatbin = output / "fatbin"
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        subprocess.run(
            [
                str(llvm / "llvm-objcopy"),
                "--dump-section",
                f".hip_fatbin={fatbin}",
                str(saved_binary),
                str(Path(scratch) / "discard.elf"),
            ],
            check=True,
        )
    assert hashlib.sha256(saved_binary.read_bytes()).hexdigest() == binary_hash
    data, cursor = fatbin.read_bytes(), 0
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    objects = []
    # https://clang.llvm.org/docs/ClangOffloadBundler.html#bundled-binary-file-layout
    # Each translation unit contributes a bundle; the linker aligns and
    # concatenates them. Offsets in a bundle header are relative to its start.
    while cursor < len(data):
        start = data.find(magic, cursor)
        if start < 0:
            assert not any(data[cursor:]), "Unexpected non-padding fatbin data"
            break
        assert not any(data[cursor:start]), "Unsupported bundle encoding"
        pos = start + len(magic)
        (count,) = struct.unpack_from("<Q", data, pos)
        assert 0 < count < 10000
        pos += 8
        end = pos
        for _ in range(count):
            offset, size, length = struct.unpack_from("<QQQ", data, pos)
            pos += 24
            assert 0 < length < 4096 and pos + length <= len(data)
            target = data[pos : pos + length].decode("ascii")
            pos += length
            assert start + offset + size <= len(data)
            end = max(end, start + offset + size)
            if not (target.startswith("hip-") and target.endswith("--gfx1201")):
                continue
            blob = data[start + offset : start + offset + size]
            assert blob.startswith(
                b"\x7fELF"
            ), "Expected an uncompressed AMDGPU ELF"
            code = output / f"object{len(objects)}.hsaco"
            assembly = code.with_suffix(".amdgcn")
            code.write_bytes(blob)
            text = subprocess.check_output(
                [str(llvm / "llvm-objdump"), "-d", str(code)], text=True
            )
            assembly.write_text(text)
            objects.append(
                {
                    "target": target,
                    "bundle_offset": start,
                    "object_offset": offset,
                    "hsaco": str(code.relative_to(ROOT)),
                    "hsaco_sha256": hashlib.sha256(blob).hexdigest(),
                    "assembly": str(assembly.relative_to(ROOT)),
                    "assembly_sha256": hashlib.sha256(
                        assembly.read_bytes()
                    ).hexdigest(),
                }
            )
        assert pos <= end and end > cursor
        cursor = end
    assert objects, "No gfx1201 code objects found"
    assembly = output / "kernel.amdgcn"
    assembly.write_text(
        "\n".join((ROOT / item["assembly"]).read_text() for item in objects)
    )
    return {
        "binary": str(saved_binary.relative_to(ROOT)),
        "binary_sha256": binary_hash,
        "build_cache_binary": str(Path(binary).relative_to(ROOT)),
        "objects": objects,
        "assembly": str(assembly.relative_to(ROOT)),
        "assembly_sha256": hashlib.sha256(assembly.read_bytes()).hexdigest(),
        "instructions": sorted(
            set(re.findall(r"v_wmma_[a-z0-9_]+", assembly.read_text()))
        ),
    }
