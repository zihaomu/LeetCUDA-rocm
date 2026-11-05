// Research kernels: make matrix accumulation precision explicit.
// Inputs/outputs: FP16. MODE 1 rounds state to FP16 after each K16 tile;
// MODE 2 retains FP32 state for the complete K reduction.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
using half8 = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float __attribute__((ext_vector_type(8)));

template <int MODE>
__global__ void matrix_gemv(const _Float16 *a, const _Float16 *b, _Float16 *c,
                            int m, int k) {
  const int lane = threadIdx.x % 32;
  const int row_base = blockIdx.x * 64 + threadIdx.x / 32 * 16;
  const int row = row_base + lane % 16;
  float8 acc = {};
  for (int base = 0; base < k; base += 16) {
    half8 av = {}, bv = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int col = base + lane / 16 * 8 + i;
      av[i] = row < m && col < k ? a[row * k + col] : (_Float16)0;
      bv[i] = lane % 16 == 0 && col < k ? b[col] : (_Float16)0;
    }
    if constexpr (MODE == 1) {
      float8 zero = {};
      const float8 partial =
          __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, zero);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[i] = (float)(_Float16)(acc[i] + partial[i]);
    } else {
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
    }
  }
  if (lane % 16 == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int output_row = row_base + lane / 16 * 8 + i;
      if (output_row < m)
        c[output_row] = (_Float16)acc[i];
    }
  }
}

template <int MODE>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() &&
                  c.device() == a.device(),
              "expected one GPU");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf &&
                  c.scalar_type() == at::kHalf,
              "expected FP16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2 && b.size(1) == 1 &&
                  c.size(1) == 1 && a.size(1) == b.size(0) &&
                  a.size(0) == c.size(0),
              "expected a[M,K], b[K,1], c[M,1]");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "expected contiguous inputs");
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b), "output aliases input");
  TORCH_CHECK(!a.requires_grad() && !b.requires_grad() && !c.requires_grad(),
              "forward only");
  TORCH_CHECK(a.numel() < INT32_MAX && a.size(1) < INT32_MAX - 16 &&
                  a.size(0) < INT32_MAX - 64,
              "32-bit index range exceeded");
  if (!a.size(0))
    return;
  const c10::cuda::CUDAGuard guard(a.device());
  matrix_gemv<MODE><<<(a.size(0) + 63) / 64, 128>>>(
      reinterpret_cast<const _Float16 *>(a.data_ptr()),
      reinterpret_cast<const _Float16 *>(b.data_ptr()),
      reinterpret_cast<_Float16 *>(c.data_ptr()), a.size(0), a.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

void hgemv_wmma_tile16_half_state(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor c) {
  launch<1>(a, b, c);
}
void hgemv_wmma_fp32_state(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<2>(a, b, c);
}

// The original interface retains FP16 I/O, but the AMD implementation
// explicitly strengthens the accumulation precision. It is not bitwise FP16 MMA
// emulation.
void hgemv_tensor_core_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<2>(a, b, c);
}
