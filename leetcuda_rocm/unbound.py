"""Additional upstream compute definitions outside the original 229 bindings.

TN B retains the upstream logical [K,N] shape with physical contiguous [N,K]
storage. All three GEMMs use FP32 accumulation and FP16 output. The two
original FP16 accumulators are explicitly upgraded; the two WGMMA entry names
therefore share the same qualified AMD producer/consumer algorithm. Hardware
WGMMA/TMA is replaced by WMMA, vector loads and acquire/release ring slots.
"""

from functools import lru_cache
import hashlib
import re
from types import ModuleType

from .environment import ROOT, cache_key, verify
from .loader import _load, _prepare

GEMM_ENTRIES = {
    "hgemm_wgmma_m64n128k16_stages_tma_ws_tn": 3,
    "hgemm_wgmma_m64n128k16_f16acc_stages_tma_ws_tn": 3,
    "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_swizzle": 5,
}
ATTENTION_ENTRY = "flash_attn_mma_stages_split_q_tiling_qk_acc_f32_rr"
ATTENTION_TARGET = "flash_attn_mma_stages_split_q_tiling_qkv_acc_f32"
ATTENTION_SOURCES = (
    "kernels/flash-attn/mma/others/flash_attn_mma_tiling_qk_F32F16F16F32_rr.cu",
    "kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qkv_F32F16F16F32.cu",
)


def attention_equivalence():
    """Check complete upstream files after name/comment/whitespace normalization."""
    originals = [(ROOT / p).read_text() for p in ATTENTION_SOURCES]
    normalized = []
    for source in originals:
        source = source.replace(ATTENTION_ENTRY, ATTENTION_TARGET)
        source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
        normalized.append(re.sub(r"\s+", "", source))
    return dict(
        equal=normalized[0] == normalized[1],
        source_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in ATTENTION_SOURCES
        },
        normalized_sha256=[
            hashlib.sha256(s.encode()).hexdigest() for s in normalized
        ],
        entry=ATTENTION_ENTRY,
        target=ATTENTION_TARGET,
    )


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
    if name == "flash_attn":
        # This is a proven source-equivalent entry, not an unrelated fallback.
        # Keep the original FP16 running output state at D >= 256.
        from .loader import load

        if not attention_equivalence()["equal"]:
            raise RuntimeError(
                "Upstream attention sources changed; requalify equivalence"
            )
        native = load("flash_attn")
        module = ModuleType("leetcuda_rocm_unbound_flash_attn")
        module.__file__ = native.__file__
        setattr(module, ATTENTION_ENTRY, getattr(native, ATTENTION_TARGET))
        return module
    if name != "hgemm":
        raise ValueError("Qualified unbound modules: hgemm, flash_attn")
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
