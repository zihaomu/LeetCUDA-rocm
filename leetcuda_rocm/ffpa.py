"""Split-D FFPA: cooperative prefetch, producer/consumer, and FA4 non-WS.

FP16 input/output with FP32 QK/PV and softmax state. gfx1201 only.
AMD geometry is 16 query rows x 64 keys. No quadratic probability buffer.
"""

from functools import lru_cache

import torch

from .environment import ROOT, cache_key, verify
from .loader import _load, _prepare

VARIANTS = {"async": 0, "ws": 1, "non_ws": 2}
HEAD_DIMS = (64, 128, 192, 256, 320, 384, 448, 512, 1024)


@lru_cache(maxsize=None)
def _native(device):
    with torch.cuda.device(device):
        identity = verify()
        header = (ROOT / "kernels/rocm/ffpa_attention_hip.cuh").read_text()
        binding = (ROOT / "leetcuda_rocm/native/ffpa_bindings.cpp").read_text()
        return _load(
            "ffpa_split_d",
            (header.removeprefix("#pragma once\n") + "\n" + _prepare(binding),),
            cache_key(identity),
            identity["gpu_architecture"],
            False,
        )


def ffpa_attention(
    q,
    k,
    v,
    *,
    variant="async",
    stages_qk=2,
    stages_v=2,
    out=None,
    lse=None,
    return_lse=False,
):
    """Inference FFPA for contiguous Q[B,H,NQ,D], K/V[B,H,NK,D].

    D is a multiple of 64 through 512, or 1024. Independent QK/V ring depths
    are integers 1..3. All variants reserve three slots, while the requested
    depth controls active slots and prefetch distance. Inputs must be finite
    and produce finite FP32 logits. No causal/mask/GQA/dropout/backward.

    async: QK64/PV64 with VGPR prefetch into LDS.
    ws: independent producer wave feeding eight consumer waves.
    non_ws: QK32/PV64, FA4 conditional rescale and LDS output staging.
    Optional natural-log LSE is an AMD extension for the first two variants.
    """
    for x in (q, k, v):
        if (
            not isinstance(x, torch.Tensor)
            or not x.is_cuda
            or x.dtype != torch.float16
            or x.ndim != 4
        ):
            raise ValueError("FFPA expects FP16 GPU tensors with rank 4")
        if not x.is_contiguous() or x.requires_grad:
            raise ValueError("FFPA expects contiguous inference tensors")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {tuple(VARIANTS)}")
    if any(
        type(x) is not int or x not in (1, 2, 3) for x in (stages_qk, stages_v)
    ):
        raise ValueError("FFPA stages must be integers 1..3")
    if q.shape[-1] not in HEAD_DIMS:
        raise ValueError(f"FFPA D must be one of {HEAD_DIMS}")
    if (
        k.shape != v.shape
        or q.shape[:2] != k.shape[:2]
        or q.shape[-1] != k.shape[-1]
        or q.device != k.device
        or q.device != v.device
    ):
        raise ValueError("FFPA input shape/device mismatch")
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    _native(q.device.index).forward(
        q, k, v, out, lse, None, VARIANTS[variant], stages_qk, stages_v, 0
    )
    return (out, lse) if return_lse else out
