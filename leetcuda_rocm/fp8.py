"""BF16 -> dynamically scaled E4M3 -> native AMD WMMA -> BF16.

Port of interview/fp8_gemm.cuh, including 128-row/column scale groups,
producer/consumer scheduling and offline B. Forward only, on gfx1201.
As in the original algorithm, nonzero group amax/448 and 448/amax must be
normal finite FP32 values. Value-domain preconditions are not scanned at runtime.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

from .environment import ROOT, cache_key, verify
from .loader import _load, _prepare

MODES = {"row_col": 0, "row_block": 1, "block_col": 2, "block_block": 3}


@lru_cache(maxsize=None)
def _native(device):
    with torch.cuda.device(device):
        identity = verify()
        header = (ROOT / "kernels/rocm/fp8_gemm_hip.cuh").read_text()
        header = header.removeprefix("#pragma once\n")
        bindings = (ROOT / "leetcuda_rocm/native/fp8_bindings.cpp").read_text()
        # Native header already supplies explicit HIP streams; adapt only the
        # PyTorch binding before the ordinary HIPify pass.
        return _load(
            "fp8_pipeline",
            (header + "\n" + _prepare(bindings),),
            cache_key(identity),
            identity["gpu_architecture"],
            False,
        )


def _input(x):
    if (
        not isinstance(x, torch.Tensor)
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
    ):
        raise ValueError("expected a BF16 matrix on the ROCm GPU")
    if not x.is_contiguous() or x.requires_grad:
        raise ValueError("expected contiguous, inference-only input")


def _workspace(a, n, workspace, offline=False):
    native = _native(a.device.index)
    m, k = a.shape
    count = (
        native.activation_size(m, k)
        if offline
        else native.workspace_size(m, n, k)
    )
    if workspace is None:
        workspace = torch.empty(count, device=a.device, dtype=torch.uint8)
    return workspace


def _out(a, n, out):
    if out is None:
        out = torch.empty(
            (a.shape[0], n), device=a.device, dtype=torch.bfloat16
        )
    return out


@dataclass(frozen=True)
class FP8Weight:
    """Offline B: E4M3 bytes in [N,K], scales, and their original granularity."""

    data: torch.Tensor
    scales: torch.Tensor
    per_col: bool


def quantize_fp8_weight(b, *, per_col=True):
    """Quantize B[K,N] once; per_col=False shares one scale per 128 columns."""
    _input(b)
    k, n = b.shape
    if k <= 0 or n <= 0 or k % 16 or n % 8:
        raise ValueError("FP8 weight requires positive K%16=0 and N%8=0")
    data = torch.empty((n, k), device=b.device, dtype=torch.uint8)
    scales = torch.empty(
        n if per_col else (n + 127) // 128, device=b.device, dtype=torch.float32
    )
    _native(b.device.index).quantize_b(b, data, scales, per_col)
    return FP8Weight(data, scales, per_col)


def fp8_gemm(a, b, *, mode="row_col", use_ws=False, workspace=None, out=None):
    """Four original scale modes; optional reusable workspace and BF16 output.

    A[M,K], B[K,N] contiguous BF16; M>0, K%16=0, N%8=0.
    Finite input, FP32 scale/inverse in normal range, finite scaled accumulation.
    use_ws=True uses distinct producer waves feeding a shared-memory ring.
    """
    _input(a)
    _input(b)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}")
    if a.device != b.device or a.shape[1] != b.shape[0]:
        raise ValueError("GEMM input device or K mismatch")
    n = b.shape[1]
    workspace = _workspace(a, n, workspace)
    out = _out(a, n, out)
    _native(a.device.index).forward(a, b, out, workspace, MODES[mode], use_ws)
    return out


def fp8_gemm_prepacked(
    a, weight, *, per_row=True, use_ws=False, workspace=None, out=None
):
    """Quantize A online and multiply by a previously quantized FP8Weight."""
    _input(a)
    if (
        not isinstance(weight, FP8Weight)
        or not isinstance(weight.data, torch.Tensor)
        or not isinstance(weight.scales, torch.Tensor)
        or weight.data.ndim != 2
    ):
        raise ValueError("expected an FP8Weight from quantize_fp8_weight")
    n, k = weight.data.shape
    if a.shape[1] != k or weight.data.device != a.device:
        raise ValueError("packed weight device or K mismatch")
    workspace = _workspace(a, n, workspace, offline=True)
    out = _out(a, n, out)
    _native(a.device.index).offline(
        a,
        weight.data,
        weight.scales,
        out,
        workspace,
        per_row,
        weight.per_col,
        use_ws,
    )
    return out
