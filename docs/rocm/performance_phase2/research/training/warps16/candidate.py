"""Tiled GQA attention with masks, explicit Philox dropout and first gradients.

Independent Triton implementation. Forward and backward use matrix operations
and O(B*H*N*D) storage; no dense attention probability matrix is allocated.
"""

import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .matrix_attention import _check_environment


@triton.jit
def _scores(
    q,
    kt,
    Mask,
    b,
    h,
    qr,
    kr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    MB: tl.constexpr,
    MH: tl.constexpr,
    MQ: tl.constexpr,
    MK: tl.constexpr,
    MASK_KIND: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
):
    scores = tl.dot(q, kt).to(tl.float32) * SCALE
    valid = (qr[:, None] < NQ) & (kr[None, :] < NK)
    if CAUSAL:
        valid = valid & (qr[:, None] >= kr[None, :])
    if MASK_KIND != 0:
        mask = tl.load(
            Mask + b * MB + h * MH + qr[:, None] * MQ + kr[None, :] * MK,
            (qr[:, None] < NQ) & (kr[None, :] < NK),
            0,
        )
        if MASK_KIND == 1:
            valid = valid & mask
        else:
            scores += mask.to(tl.float32)
    return tl.where(valid, scores * 1.4426950408889634, -float("inf"))


@triton.jit
def _dropout(
    bh,
    qr,
    kr,
    Seed,
    Offset,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    P: tl.constexpr,
):
    if P > 0:
        counter = (
            (bh * NQ + qr[:, None]).to(tl.uint32) * NK
            + kr[None, :].to(tl.uint32)
            + Offset.to(tl.uint32)
        )
        keep = tl.rand(Seed, counter, n_rounds=10) >= P
        return keep.to(tl.float32) / (1.0 - P)
    else:
        return tl.full((qr.shape[0], kr.shape[0]), 1, tl.float32)


@triton.jit
def _forward(
    Q,
    K,
    V,
    Mask,
    O,
    LSE,
    Seed,
    Offset,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    MB: tl.constexpr,
    MH: tl.constexpr,
    MQ: tl.constexpr,
    MK: tl.constexpr,
    MASK_KIND: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    P: tl.constexpr,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    TD: tl.constexpr,
):
    bh = tl.program_id(1)
    b, h = bh // HQ, bh % HQ
    kh = h // (HQ // HK)
    qr = tl.program_id(0) * TQ + tl.arange(0, TQ)
    dim = tl.arange(0, TD)
    q = tl.load(
        Q + (bh * NQ + qr[:, None]) * D + dim[None, :],
        (qr[:, None] < NQ) & (dim[None, :] < D),
        0,
    )
    maximum = tl.full((TQ,), -float("inf"), tl.float32)
    denom = tl.full((TQ,), 0, tl.float32)
    acc = tl.full((TQ, TD), 0, tl.float32)
    stop = NK
    if CAUSAL:
        stop = tl.minimum(stop, (tl.program_id(0) + 1) * TQ)
    for base in range(tl.cdiv(stop, TK)):
        kr = base * TK + tl.arange(0, TK)
        kt = tl.load(
            K + ((b * HK + kh) * NK + kr[None, :]) * D + dim[:, None],
            (kr[None, :] < NK) & (dim[:, None] < D),
            0,
        )
        scores = _scores(
            q,
            kt,
            Mask,
            b,
            h,
            qr,
            kr,
            NQ,
            NK,
            MB,
            MH,
            MQ,
            MK,
            MASK_KIND,
            CAUSAL,
            SCALE,
        )
        newmax = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(newmax == -float("inf"), 0, newmax)
        alpha = tl.exp2(maximum - safe)
        prob = tl.exp2(scores - safe[:, None])
        denom = denom * alpha + tl.sum(prob, 1)
        pv = prob * _dropout(bh, qr, kr, Seed, Offset, NQ, NK, P)
        v = tl.load(
            V + ((b * HK + kh) * NK + kr[:, None]) * D + dim[None, :],
            (kr[:, None] < NK) & (dim[None, :] < D),
            0,
        )
        acc = tl.dot(pv.to(v.dtype), v, acc * alpha[:, None])
        maximum = newmax
    inv = tl.where(denom > 0, 1.0 / denom, 0)
    tl.store(
        O + (bh * NQ + qr[:, None]) * D + dim[None, :],
        acc * inv[:, None],
        (qr[:, None] < NQ) & (dim[None, :] < D),
    )
    tl.store(
        LSE + bh * NQ + qr,
        tl.where(denom > 0, maximum + tl.log2(denom), -float("inf")),
        qr < NQ,
    )


@triton.jit
def _delta(
    Q,
    K,
    V,
    Mask,
    DO,
    LSE,
    Delta,
    Seed,
    Offset,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    MB: tl.constexpr,
    MH: tl.constexpr,
    MQ: tl.constexpr,
    MK: tl.constexpr,
    MASK_KIND: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    P: tl.constexpr,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    TD: tl.constexpr,
):
    bh = tl.program_id(1)
    b, h = bh // HQ, bh % HQ
    kh = h // (HQ // HK)
    qr = tl.program_id(0) * TQ + tl.arange(0, TQ)
    dim = tl.arange(0, TD)
    valid = (qr[:, None] < NQ) & (dim[None, :] < D)
    q = tl.load(Q + (bh * NQ + qr[:, None]) * D + dim[None, :], valid, 0)
    dout = tl.load(DO + (bh * NQ + qr[:, None]) * D + dim[None, :], valid, 0)
    lse = tl.load(LSE + bh * NQ + qr, qr < NQ, other=-float("inf"))
    total = tl.full((TQ,), 0, tl.float32)
    for base in range(tl.cdiv(NK, TK)):
        kr = base * TK + tl.arange(0, TK)
        address = ((b * HK + kh) * NK + kr[:, None]) * D + dim[None, :]
        kv = (kr[:, None] < NK) & (dim[None, :] < D)
        k = tl.load(K + address, kv, 0)
        v = tl.load(V + address, kv, 0)
        scores = _scores(
            q,
            tl.trans(k),
            Mask,
            b,
            h,
            qr,
            kr,
            NQ,
            NK,
            MB,
            MH,
            MQ,
            MK,
            MASK_KIND,
            CAUSAL,
            SCALE,
        )
        safe = tl.where(lse == -float("inf"), 0, lse)
        prob = tl.exp2(scores - safe[:, None])
        dp = tl.dot(dout, tl.trans(v)).to(tl.float32) * _dropout(
            bh, qr, kr, Seed, Offset, NQ, NK, P
        )
        total += tl.sum(prob * dp, 1)
    tl.store(Delta + bh * NQ + qr, total, qr < NQ)


@triton.jit
def _backward_q(
    Q,
    K,
    V,
    Mask,
    DO,
    LSE,
    Delta,
    DQ,
    Seed,
    Offset,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    MB: tl.constexpr,
    MH: tl.constexpr,
    MQ: tl.constexpr,
    MK: tl.constexpr,
    MASK_KIND: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    P: tl.constexpr,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    TD: tl.constexpr,
):
    bh = tl.program_id(1)
    b, h = bh // HQ, bh % HQ
    kh = h // (HQ // HK)
    qr = tl.program_id(0) * TQ + tl.arange(0, TQ)
    dim = tl.arange(0, TD)
    valid = (qr[:, None] < NQ) & (dim[None, :] < D)
    q = tl.load(Q + (bh * NQ + qr[:, None]) * D + dim[None, :], valid, 0)
    dout = tl.load(DO + (bh * NQ + qr[:, None]) * D + dim[None, :], valid, 0)
    lse = tl.load(LSE + bh * NQ + qr, qr < NQ, other=-float("inf"))
    delta = tl.load(Delta + bh * NQ + qr, qr < NQ, 0)
    dq = tl.full((TQ, TD), 0, tl.float32)
    for base in range(tl.cdiv(NK, TK)):
        kr = base * TK + tl.arange(0, TK)
        address = ((b * HK + kh) * NK + kr[:, None]) * D + dim[None, :]
        kv = (kr[:, None] < NK) & (dim[None, :] < D)
        k = tl.load(K + address, kv, 0)
        v = tl.load(V + address, kv, 0)
        scores = _scores(
            q,
            tl.trans(k),
            Mask,
            b,
            h,
            qr,
            kr,
            NQ,
            NK,
            MB,
            MH,
            MQ,
            MK,
            MASK_KIND,
            CAUSAL,
            SCALE,
        )
        safe = tl.where(lse == -float("inf"), 0, lse)
        prob = tl.exp2(scores - safe[:, None])
        dp = tl.dot(dout, tl.trans(v)).to(tl.float32) * _dropout(
            bh, qr, kr, Seed, Offset, NQ, NK, P
        )
        ds = prob * (dp - delta[:, None]) * SCALE
        dq = tl.dot(ds.to(k.dtype), k, dq)
    tl.store(DQ + (bh * NQ + qr[:, None]) * D + dim[None, :], dq, valid)


@triton.jit
def _backward_kv(
    Q,
    K,
    V,
    Mask,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    Seed,
    Offset,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    MB: tl.constexpr,
    MH: tl.constexpr,
    MQ: tl.constexpr,
    MK: tl.constexpr,
    MASK_KIND: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    P: tl.constexpr,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    TD: tl.constexpr,
):
    bkh = tl.program_id(1)
    b, kh = bkh // HK, bkh % HK
    kr = tl.program_id(0) * TK + tl.arange(0, TK)
    dim = tl.arange(0, TD)
    address = (bkh * NK + kr[:, None]) * D + dim[None, :]
    valid = (kr[:, None] < NK) & (dim[None, :] < D)
    k = tl.load(K + address, valid, 0)
    v = tl.load(V + address, valid, 0)
    dk = tl.full((TK, TD), 0, tl.float32)
    dv = tl.full((TK, TD), 0, tl.float32)
    for group in range(HQ // HK):
        h = kh * (HQ // HK) + group
        bh = b * HQ + h
        for base in range(tl.cdiv(NQ, TQ)):
            qr = base * TQ + tl.arange(0, TQ)
            qaddress = (bh * NQ + qr[:, None]) * D + dim[None, :]
            qvalid = (qr[:, None] < NQ) & (dim[None, :] < D)
            q = tl.load(Q + qaddress, qvalid, 0)
            dout = tl.load(DO + qaddress, qvalid, 0)
            lse = tl.load(LSE + bh * NQ + qr, qr < NQ, other=-float("inf"))
            delta = tl.load(Delta + bh * NQ + qr, qr < NQ, 0)
            scores = _scores(
                q,
                tl.trans(k),
                Mask,
                b,
                h,
                qr,
                kr,
                NQ,
                NK,
                MB,
                MH,
                MQ,
                MK,
                MASK_KIND,
                CAUSAL,
                SCALE,
            )
            safe = tl.where(lse == -float("inf"), 0, lse)
            prob = tl.exp2(scores - safe[:, None])
            drop = _dropout(bh, qr, kr, Seed, Offset, NQ, NK, P)
            dp = tl.dot(dout, tl.trans(v)).to(tl.float32) * drop
            ds = prob * (dp - delta[:, None]) * SCALE
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk)
            dv = tl.dot(tl.trans((prob * drop).to(dout.dtype)), dout, dv)
    tl.store(DK + address, dk, valid)
    tl.store(DV + address, dv, valid)


def _mask_args(mask, b, h, nq, nk):
    if mask is None:
        return (0, 0, 0, 0, 0)
    shape = mask.shape
    strides = mask.stride()
    if mask.ndim == 2:
        shape = (1, 1, *shape)
        strides = (0, 0, *strides)
    expected = (b, h, nq, nk)
    if len(shape) != 4 or any(x not in (1, y) for x, y in zip(shape, expected)):
        raise ValueError("mask must broadcast from [NQ,NK] or [B,HQ,NQ,NK]")
    return (
        *(s if x > 1 else 0 for s, x in zip(strides, shape)),
        1 if mask.dtype == torch.bool else 2,
    )


def _options(q, k, mask, causal, scale, p):
    b, hq, nq, d = q.shape
    return dict(
        HQ=hq,
        HK=k.shape[1],
        NQ=nq,
        NK=k.shape[2],
        D=d,
        **dict(
            zip(
                ("MB", "MH", "MQ", "MK", "MASK_KIND"),
                _mask_args(mask, b, hq, nq, k.shape[2]),
            )
        ),
        CAUSAL=causal,
        SCALE=scale,
        P=p,
        TD=triton.next_power_of_2(d),
        num_warps=4,
        num_stages=1
    )


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, causal, scale, p, seed, offset):
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        options = _options(q, k, mask, causal, scale, p)
        with torch.cuda.device(q.device):
            if q.numel():
                _forward[
                    (triton.cdiv(q.shape[2], 32), q.shape[0] * q.shape[1])
                ](
                    q,
                    k,
                    v,
                    q if mask is None else mask,
                    output,
                    lse,
                    seed,
                    offset,
                    TQ=32,
                    TK=64,
                    **options
                )
        ctx.save_for_backward(q, k, v, lse, mask)
        ctx.options = options
        ctx.seed, ctx.offset = seed, offset
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, lse, mask = ctx.saved_tensors
        dout = dout.contiguous()
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        delta = torch.empty_like(lse)
        options = dict(ctx.options)
        tq = 32 if q.shape[-1] == 256 else 32
        tk = 32 if q.shape[-1] == 256 else 32
        if q.shape[-1] == 256:
            options["num_warps"] = 16
        with torch.cuda.device(q.device):
            if q.numel():
                # Recompute softmax delta from unrounded probabilities and dP.
                # O*dO uses rounded low-precision O and can give spurious dK
                # even when a row has exactly one valid key.
                _delta[(triton.cdiv(q.shape[2], tq), q.shape[0] * q.shape[1])](
                    q,
                    k,
                    v,
                    q if mask is None else mask,
                    dout,
                    lse,
                    delta,
                    ctx.seed,
                    ctx.offset,
                    TQ=tq,
                    TK=64,
                    **options
                )
                _backward_q[
                    (triton.cdiv(q.shape[2], tq), q.shape[0] * q.shape[1])
                ](
                    q,
                    k,
                    v,
                    q if mask is None else mask,
                    dout,
                    lse,
                    delta,
                    dq,
                    ctx.seed,
                    ctx.offset,
                    TQ=tq,
                    TK=64,
                    **options
                )
            if k.numel():
                _backward_kv[
                    (triton.cdiv(k.shape[2], tk), k.shape[0] * k.shape[1])
                ](
                    q,
                    k,
                    v,
                    q if mask is None else mask,
                    dout,
                    lse,
                    delta,
                    dk,
                    dv,
                    ctx.seed,
                    ctx.offset,
                    TQ=tq,
                    TK=tk,
                    **options
                )
        return dq, dk, dv, None, None, None, None, None, None


def attention(
    q,
    k,
    v,
    *,
    mask=None,
    causal=False,
    scale=None,
    dropout_p=0.0,
    seed=0,
    offset=0
):
    """FP16/BF16 GQA forward and first-order Q/K/V gradients on gfx1201.

    Contiguous Q[B,HQ,NQ,D], K/V[B,HK,NK,D], HQ divisible by HK;
    D in {16,32,64,128,256}. Causal masking is upper-left (key<=query).
    Bool masks keep True positions; additive masks contain finite values or
    -inf, with shape [NQ,NK] or broadcastable [B,HQ,NQ,NK]. Fully masked
    rows return zero. Mask gradients and second derivatives are unsupported.
    Dropout uses explicit Philox4x32-10 seed/offset. Reusing both reproduces
    the mask, including graph replay; advance offset between training calls.
    Q/K/V and logits must be finite. These value preconditions are not scanned.
    """
    if not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in (q, k, v)):
        raise ValueError("expected Q/K/V four-dimensional tensors")
    if any(
        not x.is_cuda
        or x.device != q.device
        or x.dtype != q.dtype
        or not x.is_contiguous()
        for x in (q, k, v)
    ) or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q/K/V must be contiguous FP16/BF16 on the same GPU")
    b, h, nq, d = q.shape
    if (
        k.shape != v.shape
        or b != k.shape[0]
        or d != k.shape[-1]
        or h <= 0
        or k.shape[1] <= 0
        or h % k.shape[1]
        or d not in (16, 32, 64, 128, 256)
    ):
        raise ValueError("invalid GQA shapes or head dimension")
    if any(x.numel() >= 2**31 for x in (q, k, v)):
        raise ValueError("tensor offsets must fit signed 32-bit indexing")
    if not isinstance(causal, bool):
        raise ValueError("causal must be bool")
    scale = 1 / math.sqrt(d) if scale is None else float(scale)
    p = float(dropout_p)
    if not math.isfinite(scale) or not math.isfinite(p) or not 0 <= p < 1:
        raise ValueError("scale must be finite and dropout_p in [0,1)")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed < 2**63
    ):
        raise ValueError("seed must be an integer in [0,2**63)")
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or offset + b * h * nq * k.shape[2] >= 2**32
    ):
        raise ValueError("dropout counters must fit uint32 without wraparound")
    if mask is not None:
        if (
            not isinstance(mask, torch.Tensor)
            or not mask.is_cuda
            or mask.device != q.device
            or mask.dtype not in (torch.bool, torch.float32, q.dtype)
            or mask.requires_grad
        ):
            raise ValueError(
                "mask must be bool or floating, on input GPU, without gradients"
            )
        if (
            sum(
                (size - 1) * stride
                for size, stride in zip(mask.shape, mask.stride())
            )
            >= 2**31
        ):
            raise ValueError("mask offsets exceed signed 32-bit indexing")
    _mask_args(mask, b, h, nq, k.shape[2])
    _check_environment(q.device.index)
    return _Attention.apply(q, k, v, mask, causal, scale, p, seed, offset)
