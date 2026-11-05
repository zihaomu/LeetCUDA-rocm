// Explicit AMD counterparts of the scalar, packed and matrix-unit CuTe GEMVs.
// The WMMA ABI is documented by Clang's AMDGPUBuiltinReference and AMD GPUOpen:
// https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ half reduce_half(half sum) {
#pragma unroll
  for (int shift = 16; shift != 0; shift >>= 1)
    sum = __hadd(sum, __shfl_xor_sync(0xffffffffULL, sum, shift));
  return sum;
}

template <bool PACKED>
__global__ void vector_reduce(const half *a, const half *b, half *c, int m,
                              int k) {
  const int row = blockIdx.x * 4 + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  half sum = __float2half(0.0f);
  if constexpr (!PACKED) {
    for (int col = lane; col < k; col += 32)
      if (row < m)
        sum = __hfma(a[row * k + col], b[col], sum);
  } else {
    for (int col = lane * 8; col < k; col += 256) {
      if (row < m) {
        union alignas(16) Pack { uint4 bits; half2 pairs[4]; } pa, pb;
        pa.bits = *reinterpret_cast<const uint4 *>(a + row * k + col);
        pb.bits = *reinterpret_cast<const uint4 *>(b + col);
        // The original CuTe source left this per-tile half2 uninitialized.
        half2 pair = __float2half2_rn(0.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          pair = __hfma2(pa.pairs[i], pb.pairs[i], pair);
        sum = __hadd(sum, __hadd(__low2half(pair), __high2half(pair)));
      }
    }
  }
  sum = reduce_half(sum);
  if (lane == 0 && row < m)
    c[row] = sum;
}

using half8 = _Float16 __attribute__((ext_vector_type(8)));

__global__ void matrix_vector(const _Float16 *a, const _Float16 *b, _Float16 *c,
                              int m, int k) {
  const int lane = threadIdx.x % 32;
  const int row_base = blockIdx.x * 64 + threadIdx.x / 32 * 16;
  const int row = row_base + lane % 16;
  half8 acc = {};
  for (int base = 0; base < k; base += 16) {
    half8 av = {}, bv = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int col = base + lane / 16 * 8 + i;
      av[i] = row < m && col < k ? a[row * k + col] : (_Float16)0;
      // Only the first column of the logical B[16,16] tile is needed.
      bv[i] = lane % 16 == 0 && col < k ? b[col] : (_Float16)0;
    }
    acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(av, bv, acc);
  }
  if (lane % 16 == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int output_row = row_base + lane / 16 * 8 + i;
      if (output_row < m)
        c[output_row] = acc[i];
    }
  }
}

void check_gemv(const torch::Tensor &a, const torch::Tensor &b,
                const torch::Tensor &c, bool packed) {
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() && c.device() == a.device(),
              "HGEMV requires tensors on the same GPU");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf &&
                  c.scalar_type() == at::kHalf, "HGEMV requires FP16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2 && b.size(1) == 1 &&
                  c.size(1) == 1 && a.size(1) == b.size(0) && a.size(0) == c.size(0),
              "expected a[M,K], b[K,1], c[M,1]");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "HGEMV requires contiguous tensors");
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b), "HGEMV output aliases input");
  TORCH_CHECK(!a.requires_grad() && !b.requires_grad() && !c.requires_grad(),
              "forward-only interface");
  TORCH_CHECK(a.numel() <= std::numeric_limits<int>::max() &&
                  a.size(0) <= std::numeric_limits<int>::max() - 256 &&
                  a.size(1) <= std::numeric_limits<int>::max() - 256,
              "HGEMV exceeds 32-bit indexing including tile padding");
  if (packed)
    TORCH_CHECK(a.size(1) % 8 == 0 && uintptr_t(a.data_ptr()) % 16 == 0 &&
                    uintptr_t(b.data_ptr()) % 16 == 0,
                "packed HGEMV requires K divisible by 8 and 16-byte alignment");
}

template <int KIND>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check_gemv(a, b, c, KIND == 1);
  if (a.size(0) == 0)
    return;
  const c10::cuda::CUDAGuard guard(a.device());
  const int m = a.size(0), k = a.size(1);
  if constexpr (KIND == 2)
    matrix_vector<<<(m + 63) / 64, 128>>>(
        reinterpret_cast<const _Float16 *>(a.data_ptr()),
        reinterpret_cast<const _Float16 *>(b.data_ptr()),
        reinterpret_cast<_Float16 *>(c.data_ptr()), m, k);
  else
    vector_reduce<KIND == 1><<<(m + 3) / 4, 128>>>(
        reinterpret_cast<const half *>(a.data_ptr()),
        reinterpret_cast<const half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()), m, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void hgemv_f16_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<0>(a, b, c);
}
void hgemv_f16x8_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<1>(a, b, c);
}
void hgemv_tensor_core_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<2>(a, b, c);
}
