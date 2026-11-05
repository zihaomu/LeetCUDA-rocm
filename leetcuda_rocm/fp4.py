"""NVFP4 storage/quantization with exact expansion into AMD FP16 WMMA.

This is not native FP4 matrix arithmetic. Forward only, on gfx1201.
Input must be finite; nonzero level-1 scales and their reciprocals must be
normal finite FP32, with finite scaled accumulation. Values are not scanned.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

from .environment import ROOT, cache_key, verify
from .fp8 import _input, _out
from .loader import _load, _prepare

MODES = {"single": 0, "a": 1, "b": 2, "two_level": 3}


@lru_cache(maxsize=None)
def _native(device):
    with torch.cuda.device(device):
        identity = verify()
        header = (
            (ROOT / "kernels/rocm/fp4_gemm_hip.cuh")
            .read_text()
            .removeprefix("#pragma once\n")
        )
        bindings = (ROOT / "leetcuda_rocm/native/fp4_bindings.cpp").read_text()
        return _load(
            "fp4_pipeline",
            (header + "\n" + _prepare(bindings),),
            cache_key(identity),
            identity["gpu_architecture"],
            False,
        )


def _wide(m, n, k, tile):
    if tile not in ("auto", "default", "wide"):
        raise ValueError("tile must be auto, default or wide")
    return tile == "wide" or (tile == "auto" and min(m, n, k) >= 4096)


def _workspace(a, n, workspace, offline=False):
    lib = _native(a.device.index)
    m, k = a.shape
    count = (
        lib.activation_size(m, k) if offline else lib.workspace_size(m, n, k)
    )
    return (
        torch.empty(count, device=a.device, dtype=torch.uint8)
        if workspace is None
        else workspace
    )


@dataclass(frozen=True)
class FP4Weight:
    """Packed B[N,K/2], canonical padded block scales, level-1 scales/flag."""

    data: torch.Tensor
    block_scales: torch.Tensor
    scales: torch.Tensor
    level1: bool


def quantize_fp4_weight(b, *, level1=True):
    _input(b)
    k, n = b.shape
    if k <= 0 or n <= 0 or k % 64 or n % 8:
        raise ValueError("FP4 requires positive K%64=0 and N%8=0")
    lib = _native(b.device.index)
    # Validate padding and index limits before allocating the output buffers.
    lib.workspace_size(1, n, k)
    data = torch.empty((n, k // 2), device=b.device, dtype=torch.uint8)
    sf = torch.empty(lib.sf_size(n, k), device=b.device, dtype=torch.uint8)
    scales = torch.ones(n, device=b.device, dtype=torch.float32)
    lib.quantize_b(b, data, sf, scales, level1)
    return FP4Weight(data, sf, scales, level1)


def fp4_gemm(
    a,
    b,
    *,
    mode="two_level",
    use_ws=False,
    tile="auto",
    workspace=None,
    out=None,
):
    """BF16 -> two-level NVFP4 quantization -> FP16 WMMA/FP32 sum -> BF16.

    Contiguous, inference-only, finite BF16. Positive M,N,K; K%64=0,N%8=0.
    Nonzero level-1 scales and inverses must remain normal finite FP32.
    """
    _input(a)
    _input(b)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}")
    if a.device != b.device or a.shape[1] != b.shape[0]:
        raise ValueError("GEMM device or K mismatch")
    m, k = a.shape
    n = b.shape[1]
    wide = _wide(m, n, k, tile)
    workspace = _workspace(a, n, workspace)
    out = _out(a, n, out)
    _native(a.device.index).forward(
        a, b, out, workspace, MODES[mode], use_ws, wide
    )
    return out


def fp4_gemm_prepacked(
    a,
    weight,
    *,
    level1_a=True,
    use_ws=False,
    tile="auto",
    workspace=None,
    out=None,
):
    _input(a)
    if (
        not isinstance(weight, FP4Weight)
        or not all(
            isinstance(t, torch.Tensor)
            for t in (weight.data, weight.block_scales, weight.scales)
        )
        or weight.data.ndim != 2
    ):
        raise ValueError("expected FP4Weight from quantize_fp4_weight")
    n, half_k = weight.data.shape
    if a.shape[1] != half_k * 2 or a.device != weight.data.device:
        raise ValueError("packed weight device or K mismatch")
    wide = _wide(a.shape[0], n, half_k * 2, tile)
    workspace = _workspace(a, n, workspace, True)
    out = _out(a, n, out)
    _native(a.device.index).offline(
        a,
        weight.data,
        weight.block_scales,
        weight.scales,
        out,
        workspace,
        level1_a,
        weight.level1,
        use_ws,
        wide,
    )
    return out
