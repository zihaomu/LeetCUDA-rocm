"""Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py"""

from typing import Optional

import torch
import triton
import triton.language as tl


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states_triton(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    tensors = (output, prefix_output, suffix_output, prefix_lse, suffix_lse)
    if output_lse is not None:
        tensors += (output_lse,)
    if (
        output.ndim != 3
        or not output.is_cuda
        or any(
            not t.is_contiguous()
            or t.device != output.device
            or t.requires_grad
            for t in tensors
        )
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or any(
            t.shape != output.shape or t.dtype != output.dtype
            for t in (prefix_output, suffix_output)
        )
        or any(
            t.shape != (output.shape[1], output.shape[0])
            or t.dtype != torch.float32
            for t in (prefix_lse, suffix_lse)
        )
        or (
            output_lse is not None
            and (
                output_lse.shape != prefix_lse.shape
                or output_lse.dtype != torch.float32
            )
        )
    ):
        raise ValueError(
            "merge requires matching contiguous GPU states and FP32 [heads,tokens] LSE"
        )
    if not output.numel():
        return
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)

    with torch.cuda.device(output.device):
        merge_attn_states_kernel[(num_tokens, num_query_heads)](
            output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            head_size,
            padded_head_size,
            output_lse is not None,
        )


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    empty = max_lse == -float("inf")
    max_lse = tl.where(empty, 0.0, max_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    out_se = tl.exp(p_lse) + tl.exp(s_lse)

    if OUTPUT_LSE:
        out_lse = tl.where(empty, -float("inf"), tl.log(out_se) + max_lse)
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    out_se = tl.where(empty, 1.0, out_se)
    p_scale = tl.exp(p_lse) / out_se
    s_scale = tl.exp(s_lse) / out_se
    out = tl.where(p_scale == 0, 0.0, p_out * p_scale) + tl.where(
        s_scale == 0, 0.0, s_out * s_scale
    )
    tl.store(
        output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        out,
        mask=head_mask,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["PADDED_HEAD_SIZE", "OUTPUT_LSE"],
    use_cuda_graph=False,
)
@triton.jit
def merge_attn_states_kernel_opt(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(
        prefix_lse + head_idx * num_tokens + token_idx, cache_modifier=".cg"
    )
    s_lse = tl.load(
        suffix_lse + head_idx * num_tokens + token_idx, cache_modifier=".cg"
    )
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    empty = max_lse == -float("inf")
    max_lse = tl.where(empty, 0.0, max_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.where(empty, -float("inf"), tl.log(out_se) + max_lse)
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
        cache_modifier=".cg",
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
        cache_modifier=".cg",
    )
    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    out_se = tl.where(empty, 1.0, out_se)
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = tl.where(p_scale == 0, 0.0, p_out * p_scale) + tl.where(
        s_scale == 0, 0.0, s_out * s_scale
    )
    tl.store(
        output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        out,
        mask=head_mask,
    )
