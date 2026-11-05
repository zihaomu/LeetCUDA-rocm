"""Independent interview FlashAttention schedules for gfx1201.

These do not alias FFPA. Only FP16 inference on gfx1201. Manual FP16-accumulator
modes are explicitly upgraded to FP32 QK/PV/online state in the qualified API.
"""

from functools import lru_cache
import math
import torch
from .environment import ROOT, verify, cache_key
from .loader import _load

VARIANTS = {
    "mma": 0,
    "tma_ws": 1,
    "fa3": 2,
    "cute": 3,
    "cute_ws": 4,
    "cute_fa3": 5,
    "persist_d": 7,
}


def source_paths():
    return tuple(
        ROOT / p
        for p in (
            "kernels/rocm/interview_common_hip.cuh",
            "kernels/rocm/interview_flash_attention_hip.cuh",
            "leetcuda_rocm/native/interview_attention_bindings.cpp",
        )
    )


@lru_cache(maxsize=None)
def _native(device):
    with torch.cuda.device(device):
        identity = verify()
        common, header, binding = (p.read_text() for p in source_paths())
        source = (
            common.replace("#pragma once", "")
            + "\n"
            + header.replace("#pragma once", "").replace(
                '#include "interview_common_hip.cuh"', ""
            )
            + "\n"
            + binding
        )
        return _load(
            "interview_attention",
            (source,),
            cache_key(identity),
            identity["gpu_architecture"],
            False,
        )


def interview_attention(
    q,
    k,
    v,
    *,
    variant="mma",
    stages_k=2,
    stages_v=1,
    pad_mask=None,
    causal=False,
    scale=None,
    persistent_ctas=0,
    out=None,
    lse=None,
    return_lse=False,
):
    """FP16 BHND forward; persist_d alone supports GQA and distinct NQ/NKV.

    K slots 1..4, V slots 1..2 for WS (1 for cooperative/FA3); persist_d uses
    1..2 slots for both. The fixed LDS capacity remains reserved at all depths.
    pad_mask bits Q/K/V select row padding 8 in mma, otherwise compact XOR;
    default mma mask 7. Other variants use their own GMMA/CuTe layouts.
    persist_d preserves half(scale*log2e*Q), bottom-right causal alignment,
    conditional rescale threshold 8 and persistent query-tile scheduling.
    No mask/dropout/autograd. Finite input/logits and positive lengths required.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {tuple(VARIANTS)}")
    for t in (q, k, v):
        if (
            not isinstance(t, torch.Tensor)
            or not t.is_cuda
            or t.ndim != 4
            or t.dtype != torch.float16
        ):
            raise ValueError(
                "interview attention expects FP16 rank-4 GPU tensors"
            )
    if any(type(x) is not int for x in (stages_k, stages_v, persistent_ctas)):
        raise ValueError("stage and CTA counts must be integers")
    if type(causal) is not bool:
        raise ValueError("causal must be boolean")
    if pad_mask is None:
        pad_mask = 7 if variant == "mma" else 0
    if type(pad_mask) is not int:
        raise ValueError("pad_mask must be an integer")
    if any(n <= 0 for n in q.shape):
        raise ValueError("all dimensions must be positive")
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    _native(q.device.index).forward(
        q,
        k,
        v,
        out,
        lse,
        VARIANTS[variant],
        stages_k,
        stages_v,
        pad_mask,
        causal,
        float(scale),
        persistent_ctas,
        None,
        0,
    )
    return (out, lse) if return_lse else out
