"""Create the frozen, host-only FP32 inputs for standalone HIP/CUDA checks."""

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generator = random.Random(20261106)
    data = b"".join(
        struct.pack("<f", (generator.getrandbits(16) - 32768) / 32768)
        for _ in range(262144)
    )
    digest = hashlib.sha256(data).hexdigest()
    if args.output.exists() and args.output.read_bytes() != data:
        raise ValueError("Existing fixture differs; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    manifest = {
        "count": 262144,
        "dtype": "little-endian float32",
        "seed": 20261106,
        "sha256": digest,
        "generator": "(getrandbits(16)-32768)/32768; exact binary fractions",
        "generator_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(digest)


if __name__ == "__main__":
    main()
