"""Project qualification is separate from the shared foundation verifier."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "scripts/rocm/environment.lock.json"


def verify():
    import torch

    lock = json.loads(LOCK.read_text())
    actual = {
        "image_id": os.environ.get("DEEPINFER_IMAGE_ID"),
        "rocm_release": os.environ.get("DEEPINFER_ROCM_RELEASE"),
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "triton": importlib.metadata.version("triton"),
        "hip_build": torch.version.hip,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
    }
    expected = {
        "image_id": lock["image"]["id"],
        "rocm_release": lock["rocm_release"],
        "python": lock["python"],
        **lock["dependencies"],
        "hip_build": lock["hip_build"],
        "cxx11_abi": lock["cxx11_abi"],
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise RuntimeError(
                f"ROCm project lock mismatch: {key}={actual[key]!r}, expected {value!r}; use scripts/rocm/run.sh"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable")
    # HIP ordinals are not rocm-smi physical indices on this host. Check the
    # PCI identity before executing a kernel, rather than trusting the mask.
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    buses = []
    for index in range(torch.cuda.device_count()):
        bus = ctypes.create_string_buffer(64)
        status = hip.hipDeviceGetPCIBusId(bus, len(bus), index)
        if status:
            raise RuntimeError(f"Cannot verify GPU PCI identity: HIP {status}")
        buses.append(bus.value.decode().lower())
    if "0000:63:00.0" in buses:
        raise RuntimeError(
            "Physical GPU3 (PCI 0000:63:00.0) is excluded from compute; select by verified HIP-to-PCI mapping"
        )
    actual["visible_device_pci_bus_ids"] = buses
    actual["selected_device_pci_bus_id"] = buses[torch.cuda.current_device()]
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    actual["gpu_architecture"] = props.gcnArchName.split(":")[0]
    if actual["gpu_architecture"] != lock["gpu_architecture"]:
        raise RuntimeError(
            "This profile is qualified only for gfx1201 (wave32)"
        )
    actual["gpu_name"] = props.name
    actual["visible_devices"] = os.environ.get("HIP_VISIBLE_DEVICES")
    if not actual["visible_devices"] or "3" in actual["visible_devices"].split(
        ","
    ):
        raise RuntimeError(
            "Explicit HIP_VISIBLE_DEVICES is required; physical GPU3 is excluded"
        )
    actual["foundation_report"] = os.environ.get("DEEPINFER_ENVIRONMENT_REPORT")
    return actual


def cache_key(identity):
    return hashlib.sha256(
        json.dumps(
            {
                k: v
                for k, v in identity.items()
                if k
                not in (
                    "foundation_report",
                    "visible_devices",
                    "visible_device_pci_bus_ids",
                    "selected_device_pci_bus_id",
                )
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-key", action="store_true")
    args = parser.parse_args()
    identity = verify()
    print(
        cache_key(identity)
        if args.cache_key
        else json.dumps(identity, indent=2)
    )
