// AMD scalar and packed CuTe GEMV counterparts: FP16 I/O, explicit FP32
// accumulation and wave reduction. Original FP16 experiments remain in
// hgemv_layouts.cu; these two paths keep distinct per-lane K partitions.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
template <bool Packed>
__global__ void vector_fp32(const _Float16 *a, const _Float16 *b, _Float16 *c,
                            int m, int k) {
  int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  float sum = 0;
  if constexpr (!Packed) {
    for (int col = lane; col < k; col += 32)
      if (row < m)
        sum = fmaf(float(a[row * k + col]), float(b[col]), sum);
  } else {
    for (int col = lane * 8; col < k; col += 256) {
      if (row < m) {
        H8 av = {}, bv = {};
        if (col + 8 <= k && k % 8 == 0 && uintptr_t(a) % 16 == 0 &&
            uintptr_t(b) % 16 == 0) {
          av = *reinterpret_cast<const H8 *>(a + row * k + col);
          bv = *reinterpret_cast<const H8 *>(b + col);
        } else {
#pragma unroll
          for (int j = 0; j < 8; j++)
            if (col + j < k) {
              av[j] = a[row * k + col + j];
              bv[j] = b[col + j];
            }
        }
        // Retain the original tile-local even/odd pair decomposition.
        float even = 0, odd = 0;
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          even = fmaf(float(av[j]), float(bv[j]), even);
          odd = fmaf(float(av[j + 1]), float(bv[j + 1]), odd);
        }
        sum += even + odd;
      }
    }
  }
#pragma unroll
  for (int shift = 16; shift; shift >>= 1)
    sum += __shfl_xor_sync(0xffffffffULL, sum, shift);
  if (lane == 0 && row < m)
    c[row] = (_Float16)sum;
}
void check(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() &&
                  c.device() == a.device(),
              "expected one GPU");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf &&
                  c.scalar_type() == at::kHalf,
              "expected FP16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2 && b.size(1) == 1 &&
                  c.size(1) == 1 && a.size(1) == b.size(0) &&
                  a.size(0) == c.size(0),
              "expected A[M,K], B[K,1], C[M,1]");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "expected contiguous");
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b), "output aliases input");
  TORCH_CHECK(!a.requires_grad() && !b.requires_grad() && !c.requires_grad(),
              "forward only");
  TORCH_CHECK(a.numel() < INT32_MAX && b.numel() < INT32_MAX &&
                  a.size(0) < INT32_MAX - 256 && a.size(1) < INT32_MAX - 256,
              "32-bit index range exceeded");
}
template <bool Packed>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check(a, b, c);
  if (!c.numel())
    return;
  c10::cuda::CUDAGuard guard(a.device());
  vector_fp32<Packed><<<(a.size(0) + 3) / 4, 128>>>(
      reinterpret_cast<const _Float16 *>(a.data_ptr()),
      reinterpret_cast<const _Float16 *>(b.data_ptr()),
      reinterpret_cast<_Float16 *>(c.data_ptr()), a.size(0), a.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
void hgemv_f16_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<false>(a, b, c);
}
void hgemv_f16x8_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<true>(a, b, c);
}
