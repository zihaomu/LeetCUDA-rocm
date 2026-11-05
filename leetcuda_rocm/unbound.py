"""Additional upstream compute definitions outside the original 229 bindings.

TN B retains the upstream logical [K,N] shape with physical contiguous [N,K]
storage. All three GEMMs use FP32 accumulation and FP16 output. The two
original FP16 accumulators are explicitly upgraded; the two WGMMA entry names
therefore share the same qualified AMD producer/consumer algorithm. Hardware
WGMMA/TMA is replaced by WMMA, vector loads and acquire/release ring slots.
"""

from functools import lru_cache

from .environment import ROOT, cache_key, verify
from .loader import _load, _prepare

GEMM_ENTRIES = {
    "hgemm_wgmma_m64n128k16_stages_tma_ws_tn": 3,
    "hgemm_wgmma_m64n128k16_f16acc_stages_tma_ws_tn": 3,
    "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_swizzle": 5,
}


def source_paths():
    return tuple(
        ROOT / p
        for p in (
            "kernels/rocm/interview_common_hip.cuh",
            "kernels/rocm/interview_hgemm_hip.cuh",
            "leetcuda_rocm/native/unbound_gemm_bindings.cpp",
        )
    )


@lru_cache(maxsize=None)
def load_unbound(name="hgemm"):
    """Load additional GEMMs; all support stages 2..5 and current-device streams.

    An out-of-range stage retains upstream fallback: 3 for WGMMA, 2 for TN.
    Swizzle stride must be positive when enabled. Inputs are finite FP16,
    contiguous inference tensors, C has separate storage. AMD supports tails
    and 2-byte input alignment; original CUDA full-tile contracts are narrower.
    Build before graph capture. gfx1201 only; no backward.
    """
    if name != "hgemm":
        raise ValueError("Currently qualified unbound module: hgemm")
    identity = verify()
    common, header, binding = (p.read_text() for p in source_paths())
    source = (
        common.replace("#pragma once", "")
        + "\n"
        + header.replace('#include "interview_common_hip.cuh"', "").replace(
            "#pragma once", ""
        )
        + "\n"
        + _prepare(binding)
    )
    return _load(
        "unbound_hgemm",
        (source,),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )
