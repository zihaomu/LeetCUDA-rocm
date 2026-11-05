"""Compiler worker: publish an immutable DSO before the parent imports it.

This process never runs numerical or performance checks. Torch's mutable build
cache is allowed to relink; the parent only imports the content-addressed copy.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main():
    from torch.utils.cpp_extension import load

    request = json.loads(Path(sys.argv[1]).read_text())
    module = load(**request["build"])
    mutable = Path(module.__file__)
    data = mutable.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    frozen = Path(request["frozen_root"]) / digest / mutable.name
    if not frozen.exists():
        atomic_write(frozen, data)
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Immutable native extension hash mismatch")
    receipt = dict(
        recipe_sha256=request["recipe_sha256"],
        binary=str(frozen),
        binary_sha256=digest,
        module_name=module.__name__,
        mutable_build_path=str(mutable),
    )
    atomic_write(
        Path(request["receipt"]),
        (json.dumps(receipt, indent=2) + "\n").encode(),
    )


if __name__ == "__main__":
    main()
