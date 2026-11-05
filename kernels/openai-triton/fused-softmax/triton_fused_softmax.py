"""
Adapted from Triton tutorial
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


# @torch.compile
def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    k_stages: tl.constexpr,
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=k_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


DEVICE = torch.device("cuda")


def get_device_properties(device_id=None):
    device_id = torch.cuda.current_device() if device_id is None else device_id
    return triton.runtime.driver.active.utils.get_device_properties(device_id)


@lru_cache(maxsize=128)
def _launch_config(device_id, n_cols, dtype):
    # Compile for the actual device, width and dtype. ROCm uses wave32 here;
    # neither register counts nor warp width should be assumed to match CUDA.
    with torch.cuda.device(device_id):
        props = get_device_properties(device_id)
        block = triton.next_power_of_2(n_cols)
        warps = 8
        stages = 4 if props["max_shared_mem"] > 200000 else 2
        probe = torch.empty((1, n_cols), dtype=dtype, device="cuda")
        kernel = softmax_kernel.warmup(
            probe,
            probe,
            n_cols,
            n_cols,
            1,
            n_cols,
            BLOCK_SIZE=block,
            k_stages=stages,
            num_warps=warps,
            grid=(1,),
        )
        kernel._init_handles()
        occupancy = props["max_num_regs"] // max(
            1, kernel.n_regs * props["warpSize"] * warps
        )
        if kernel.metadata.shared:
            occupancy = min(
                occupancy, props["max_shared_mem"] // kernel.metadata.shared
            )
        occupancy = min(
            occupancy,
            props.get("max_threads_per_sm", 1024)
            // (props["warpSize"] * warps),
        )
        return (
            props["multiprocessor_count"] * max(1, occupancy),
            block,
            warps,
            stages,
        )


def get_num_programs(x):
    return _launch_config(x.device.index, x.shape[1], x.dtype)[0]


def triton_softmax(x: torch.Tensor):
    """Row softmax for contiguous FP16/BF16/FP32 GPU matrices."""
    if (
        not x.is_cuda
        or x.ndim != 2
        or not x.is_contiguous()
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or x.requires_grad
    ):
        raise ValueError(
            "softmax requires a contiguous forward-only floating GPU matrix"
        )
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    if not x.numel():
        return y
    with torch.cuda.device(x.device):
        programs, block, warps, stages = _launch_config(
            x.device.index, n_cols, x.dtype
        )
        softmax_kernel[(min(programs, n_rows),)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=block,
            k_stages=stages,
            num_warps=warps,
        )
    return y


def demo():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    torch.testing.assert_close(triton_softmax(x), torch.softmax(x, dim=-1))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # argument names to use as an x-axis for the plot
        x_vals=[
            256 * i for i in range(1, 64)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton-fused-softmax",
            "torch-fused-softmax",
            "torch-naive-softmax",
        ],  # possible values for `line_arg``
        line_names=[
            "Triton Fused Softmax",
            "Torch Fused Softmax",
            "Torch Naive Softmax",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        xlabel="M",  # label name for the x-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={
            "N": 2048
        },  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    if provider == "torch-naive-softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    if provider == "triton-fused-softmax":
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    if provider == "torch-fused-softmax":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    demo()
    benchmark.run(show_plots=True, print_data=True, save_path="./")
