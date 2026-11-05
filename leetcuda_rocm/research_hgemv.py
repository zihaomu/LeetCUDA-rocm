"""Research AMD CuTe GEMV replacements without publishing unqualified exports."""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import torch

from .environment import ROOT, cache_key, verify
from .isa import hip_isa
from .loader import _load, _prepare


def arithmetic_references(a, b):
    # Independently model half FMA, tile-local half2 sums, and XOR reduction.
    m, k = a.shape
    padded_k = (k + 255) // 256 * 256
    products = torch.zeros(m, padded_k, device=a.device, dtype=torch.float64)
    products[:, :k] = a.double() * b.T.double()
    refs = {}
    for name, block in (("hgemv_f16_cute", 32), ("hgemv_f16x8_cute", 256)):
        lanes = torch.zeros(m, 32, device=a.device, dtype=torch.float16)
        for base in range(0, padded_k, block):
            tile = products[:, base : base + block]
            if block == 32:
                lanes = (lanes.double() + tile).half()
            else:
                tile = tile.view(m, 32, 8)
                pair = torch.zeros(
                    m, 32, 2, device=a.device, dtype=torch.float16
                )
                for index in range(0, 8, 2):
                    pair = (
                        pair.double() + tile[:, :, index : index + 2]
                    ).half()
                local = (pair[:, :, 0].double() + pair[:, :, 1].double()).half()
                lanes = (lanes.double() + local.double()).half()
        index = torch.arange(32, device=a.device)
        for shift in (16, 8, 4, 2, 1):
            lanes = (lanes.double() + lanes[:, index ^ shift].double()).half()
        refs[name] = lanes[:, :1]
    acc = torch.zeros(m, 1, device=a.device, dtype=torch.float16)
    acc_rtz = torch.zeros_like(acc)
    for base in range(0, padded_k, 16):
        partial = products[:, base : base + 16].sum(1, keepdim=True)
        acc = (acc.double() + partial).half()
        exact = acc_rtz.double() + partial
        nearest = exact.half()
        acc_rtz = torch.where(
            nearest.double().abs() > exact.abs(),
            (nearest.view(torch.int16) - 1).view(torch.float16),
            nearest,
        )
    refs["hgemv_tensor_core_cute"] = acc
    refs["wmma_round_toward_zero_model"] = acc_rtz
    host_products = products.cpu().numpy()
    for label, order in (
        ("sequential", tuple(range(16))),
        (
            "interleaved4",
            (0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15),
        ),
    ):
        accumulator = np.zeros(m, dtype=np.float16)
        for base in range(0, padded_k, 16):
            for column in order:
                accumulator = (
                    accumulator.astype(np.float64)
                    + host_products[:, base + column]
                ).astype(np.float16)
        refs[f"wmma_{label}_fma_model"] = torch.from_numpy(
            accumulator[:, None]
        ).to(a.device)
    return refs


def main():
    identity = verify()
    path = ROOT / "kernels/rocm/hgemv_layouts.cu"
    names = ("hgemv_f16_cute", "hgemv_f16x8_cute", "hgemv_tensor_core_cute")
    bindings = "#include <torch/extension.h>\n"
    for name in names:
        bindings += (
            f"void {name}(torch::Tensor, torch::Tensor, torch::Tensor);\n"
        )
    bindings += "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
    for name in names:
        bindings += f'm.def("{name}", &{name});\n'
    bindings += "}\n"
    lib = _load(
        "hgemv_research",
        (_prepare(bindings), _prepare(path.read_text())),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )
    checks = []
    for seed in (20261103, 20261104, 20261105):
        torch.manual_seed(seed)
        for m, k in (
            (1, 1),
            (3, 8),
            (17, 19),
            (64, 128),
            (65, 256),
            (129, 257),
            (64, 1024),
            (64, 8192),
            (3, 0),
            (0, 8),
        ):
            a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.25
            b = torch.randn(k, 1, device="cuda", dtype=torch.float16) * 0.25
            reference = a.double() @ b.double()
            arithmetic = arithmetic_references(a, b)
            for name in names:
                if name == "hgemv_f16x8_cute" and k % 8:
                    continue
                out = torch.full(
                    (m, 1), float("nan"), device="cuda", dtype=torch.float16
                )
                getattr(lib, name)(a, b, out)
                passed = torch.allclose(
                    out, reference.half(), atol=0.003, rtol=0.003
                )
                row = {
                    "function": name,
                    "seed": seed,
                    "shape_mk": [m, k],
                    "passed": passed,
                    "max_abs_error": (
                        (out.double() - reference).abs().max().item()
                        if out.numel()
                        else 0
                    ),
                    "atol": 0.003,
                    "rtol": 0.003,
                    "matches_half_arithmetic_model": torch.equal(
                        out, arithmetic[name]
                    ),
                    "max_abs_error_half_model": (
                        (out.float() - arithmetic[name].float())
                        .abs()
                        .max()
                        .item()
                        if out.numel()
                        else 0
                    ),
                }
                if name == "hgemv_tensor_core_cute":
                    row["matches_wmma_rtz_hypothesis"] = torch.equal(
                        out, arithmetic["wmma_round_toward_zero_model"]
                    )
                    for label in ("sequential", "interleaved4"):
                        row[f"matches_wmma_{label}_fma_hypothesis"] = (
                            torch.equal(
                                out, arithmetic[f"wmma_{label}_fma_model"]
                            )
                        )
                checks.append(row)
                print(json.dumps(row), flush=True)
    isa = hip_isa(lib.__file__, ROOT / "artifacts/rocm/hgemv/isa")
    assert "v_wmma_f16_16x16x16_f16" in isa["instructions"]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": identity,
        "qualified": False,
        "failed_functions": sorted(
            {c["function"] for c in checks if not c["passed"]}
        ),
        "checks": checks,
        "isa": isa,
        "source_files_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                path,
                ROOT / "leetcuda_rocm/research_hgemv.py",
                ROOT / "leetcuda_rocm/loader.py",
                ROOT / "leetcuda_rocm/isa.py",
            )
        },
        "oracle_note": "Scalar/packed half-FMA models are independent; matrix rounding/order alternatives are hypotheses, not asserted hardware semantics.",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "binary": lib.__file__,
        "binary_sha256": hashlib.sha256(
            open(lib.__file__, "rb").read()
        ).hexdigest(),
    }
    output = ROOT / "artifacts/rocm/hgemv/research.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return int(not all(c["passed"] for c in checks))


if __name__ == "__main__":
    raise SystemExit(main())
