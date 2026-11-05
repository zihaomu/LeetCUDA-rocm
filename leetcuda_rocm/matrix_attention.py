"""AMD matrix-unit GEMM and tiled online-softmax attention (forward only).

The algorithms correspond to LeetCUDA's tiled GEMM and split-Q attention.
Triton lowers tl.dot to RDNA4 WMMA; qualification checks the generated ISA.
These are independent AMD implementations, not wrappers around torch matmul
or SDPA. FP32 accumulators are used for both FP16 and BF16 inputs.
"""

import math
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from .environment import ROOT, cache_key, verify


@triton.jit
def _gemm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
):
    row = tl.program_id(0) * TM + tl.arange(0, TM)
    col = tl.program_id(1) * TN + tl.arange(0, TN)
    red = tl.arange(0, TK)
    acc = tl.full((TM, TN), 0, tl.float32)
    for base in range(tl.cdiv(K, TK)):
        kk = base * TK + red
        lhs = tl.load(
            A + row[:, None] * AM + kk[None, :] * AK,
            (row[:, None] < M) & (kk[None, :] < K),
            0,
        )
        rhs = tl.load(
            B + kk[:, None] * BK + col[None, :] * BN,
            (kk[:, None] < K) & (col[None, :] < N),
            0,
        )
        acc = tl.dot(lhs, rhs, acc)
    tl.store(
        C + row[:, None] * N + col[None, :],
        acc,
        (row[:, None] < M) & (col[None, :] < N),
    )


@triton.jit
def _attention_kernel(
    Q,
    K,
    V,
    O,
    H: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QS: tl.constexpr,
    QD: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KS: tl.constexpr,
    KD: tl.constexpr,
    VB: tl.constexpr,
    VH: tl.constexpr,
    VS: tl.constexpr,
    VD: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    TQ: tl.constexpr,
    TV: tl.constexpr,
    TD: tl.constexpr,
):
    # One program owns a block of query rows for one (batch, head).
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    qr = tl.program_id(0) * TQ + tl.arange(0, TQ)
    dim = tl.arange(0, TD)
    q = tl.load(
        Q + b * QB + h * QH + qr[:, None] * QS + dim[None, :] * QD,
        (qr[:, None] < NQ) & (dim[None, :] < D),
        0,
    )
    running_max = tl.full((TQ,), -float("inf"), tl.float32)
    denominator = tl.full((TQ,), 0, tl.float32)
    numerator = tl.full((TQ, TD), 0, tl.float32)
    stop = NK
    if CAUSAL:
        stop = tl.minimum(NK, (tl.program_id(0) + 1) * TQ)
    for base in range(tl.cdiv(stop, TV)):
        kr = base * TV + tl.arange(0, TV)
        k = tl.load(
            K + b * KB + h * KH + dim[:, None] * KD + kr[None, :] * KS,
            (dim[:, None] < D) & (kr[None, :] < NK),
            0,
        )
        scores = tl.dot(q, k).to(tl.float32) * (SCALE * 1.4426950408889634)
        valid = kr[None, :] < NK
        if CAUSAL:
            valid = valid & (qr[:, None] >= kr[None, :])
        scores = tl.where(valid, scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        correction = tl.exp2(running_max - next_max)
        probabilities = tl.exp2(scores - next_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        numerator = numerator * correction[:, None]
        v = tl.load(
            V + b * VB + h * VH + kr[:, None] * VS + dim[None, :] * VD,
            (kr[:, None] < NK) & (dim[None, :] < D),
            0,
        )
        numerator = tl.dot(probabilities.to(v.dtype), v, numerator)
        running_max = next_max
    result = numerator / denominator[:, None]
    tl.store(
        O + (bh * NQ + qr[:, None]) * D + dim[None, :],
        result,
        (qr[:, None] < NQ) & (dim[None, :] < D),
    )


@lru_cache(maxsize=None)
def _check_environment(device):
    with torch.cuda.device(device):
        identity = verify()
        cache = ROOT / ".cache/rocm10" / cache_key(identity) / "triton"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache)
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or target.arch != "gfx1201":
            raise RuntimeError(
                "This implementation is qualified only for ROCm gfx1201"
            )


def _check_inputs(*tensors):
    first = tensors[0]
    for tensor in tensors:
        if not tensor.is_cuda or tensor.device != first.device:
            raise ValueError("all inputs must be on the same ROCm GPU")
        if (
            tensor.dtype not in (torch.float16, torch.bfloat16)
            or tensor.dtype != first.dtype
        ):
            raise ValueError("all inputs must have the same FP16 or BF16 dtype")
        if tensor.requires_grad:
            raise ValueError(
                "forward-only operator; detach inputs explicitly for inference"
            )
        if any(size == 0 for size in tensor.shape):
            raise ValueError("empty dimensions are not supported")
        if (
            sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride())
            )
            >= 2**31
        ):
            raise ValueError(
                "tensor offsets must fit in signed 32-bit indexing"
            )
    _check_environment(first.device.index)


def _output(shape, inputs, out):
    if math.prod(shape) >= 2**31:
        raise ValueError("output must fit in signed 32-bit indexing")
    if out is None:
        return torch.empty(
            shape, dtype=inputs[0].dtype, device=inputs[0].device
        )
    if (
        tuple(out.shape) != tuple(shape)
        or out.dtype != inputs[0].dtype
        or out.device != inputs[0].device
        or not out.is_contiguous()
        or out.requires_grad
    ):
        raise ValueError(
            "out must have the expected shape, dtype, device and contiguous layout without gradients"
        )
    # Conservatively reject shared storage, including partially overlapping views.
    if any(
        out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        for x in inputs
    ):
        raise ValueError("out must not share storage with an input")
    return out


def _launch_gemm(a, b, out):
    m, k = a.shape
    n = b.shape[1]
    tm, tn, tk, warps = (
        (128, 128, 64, 8) if min(m, n) >= 512 else (64, 64, 32, 4)
    )
    return _gemm_kernel[(triton.cdiv(m, tm), triton.cdiv(n, tn))](
        a,
        b,
        out,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        TM=tm,
        TN=tn,
        TK=tk,
        num_warps=warps,
        num_stages=2,
    )


def gemm(a, b, *, out=None):
    """C[M,N] = A[M,K] @ B[K,N], FP32 accumulation, input-dtype output.

    Positive dimensions, strided inputs and tail tiles are supported. This is
    an inference-only API; optional out is contiguous and cannot alias inputs.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    _check_inputs(a, b)
    out = _output((a.shape[0], b.shape[1]), (a, b), out)
    with torch.cuda.device(a.device):
        _launch_gemm(a, b, out)
    return out


def _launch_attention(q, k, v, out, causal, scale):
    b, h, nq, d = q.shape
    nk = k.shape[2]
    return _attention_kernel[(triton.cdiv(nq, 64), b * h)](
        q,
        k,
        v,
        out,
        h,
        nq,
        nk,
        d,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        SCALE=scale,
        CAUSAL=causal,
        TQ=64,
        TV=64,
        TD=triton.next_power_of_2(d),
        num_warps=4,
        num_stages=1,
    )


def flash_attention(q, k, v, *, causal=False, scale=None, out=None):
    """Tiled softmax(QK^T * scale)V, without materializing an NQ x NK matrix.

    Q[B,H,NQ,D], K/V[B,H,NK,D], D in {16,32,64,128}. Strided inputs are
    supported. Causal mode requires NQ==NK and includes the diagonal. No
    dropout, arbitrary mask, GQA or backward pass is provided by this API.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected four-dimensional [B,H,N,D] tensors")
    if (
        k.shape != v.shape
        or q.shape[:2] != k.shape[:2]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError(
            "batch, heads, head dimension and K/V shapes must match"
        )
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("head dimension must be 16, 32, 64 or 128")
    if not isinstance(causal, bool) or (causal and q.shape[2] != k.shape[2]):
        raise ValueError(
            "causal must be bool; causal attention requires NQ==NK"
        )
    _check_inputs(q, k, v)
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    out = _output(q.shape, (q, k, v), out)
    with torch.cuda.device(q.device):
        _launch_attention(q, k, v, out, causal, scale)
    return out
