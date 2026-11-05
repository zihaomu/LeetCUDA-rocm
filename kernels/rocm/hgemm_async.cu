// Three original asynchronous HGEMM variants: three original SIMT microtiles
// with real two-slot register-prefetched LDS pipelines. Explicit FP32
// accumulator upgrade.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using H16 = _Float16 __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
using U4 = unsigned __attribute__((ext_vector_type(4)));

__device__ __forceinline__ U4 issue8(const _Float16 *p) {
  U4 r;
  asm volatile("global_load_b128 %0, %1, off" : "=v"(r) : "v"(p) : "memory");
  return r;
}
__device__ __forceinline__ H8 finish8(U4 &r) {
  asm volatile("s_wait_loadcnt 0" : "+v"(r)::"memory");
  return __builtin_bit_cast(H8, r);
}

template <int BK_, int TM_> struct Config {
  static constexpr int BM = 128, BN = 128, BK = BK_, TM = TM_, TN = 8;
  static constexpr int Threads = (BM / TM) * (BN / TN), Stride = 136;
  static constexpr int AV = BM * BK / (Threads * 8),
                       BV = BK * BN / (Threads * 8);
  static constexpr int Size = 2 * BK * Stride;
};
template <class T>
__device__ __forceinline__ void issue_tile(const _Float16 *a, const _Float16 *b,
                                           int m, int n, int k, int tile,
                                           U4 *ar, U4 *br) {
#pragma unroll
  for (int i = 0; i < T::AV; ++i) {
    int x = (threadIdx.x + i * T::Threads) * 8;
    int row = blockIdx.y * T::BM + x / T::BK, col = tile * T::BK + x % T::BK;
    if (row < m && col + 8 <= k && k % 8 == 0 &&
        reinterpret_cast<uintptr_t>(a) % 16 == 0)
      ar[i] = issue8(a + row * k + col);
    else {
      H8 h = {};
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (row < m && col + j < k)
          h[j] = a[row * k + col + j];
      ar[i] = __builtin_bit_cast(U4, h);
    }
  }
#pragma unroll
  for (int i = 0; i < T::BV; ++i) {
    int x = (threadIdx.x + i * T::Threads) * 8;
    int row = tile * T::BK + x / T::BN, col = blockIdx.x * T::BN + x % T::BN;
    if (row < k && col + 8 <= n && n % 8 == 0 &&
        reinterpret_cast<uintptr_t>(b) % 16 == 0)
      br[i] = issue8(b + row * n + col);
    else {
      H8 h = {};
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (row < k && col + j < n)
          h[j] = b[row * n + col + j];
      br[i] = __builtin_bit_cast(U4, h);
    }
  }
}
template <class T>
__device__ __forceinline__ void publish(_Float16 *lds, U4 *ar, U4 *br) {
#pragma unroll
  for (int i = 0; i < T::AV; ++i) {
    H8 h = finish8(ar[i]);
    int x = (threadIdx.x + i * T::Threads) * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      lds[(x % T::BK + j) * T::Stride + x / T::BK] = h[j];
  }
#pragma unroll
  for (int i = 0; i < T::BV; ++i) {
    H8 h = finish8(br[i]);
    int x = (threadIdx.x + i * T::Threads) * 8;
    *reinterpret_cast<H8 *>(lds + T::BK * T::Stride + (x / T::BN) * T::Stride +
                            x % T::BN) = h;
  }
}
template <int BK, int TM, bool Prefetch = true>
__global__ void __launch_bounds__((Config<BK, TM>::Threads))
    async_hgemm(const _Float16 *a, const _Float16 *b, _Float16 *c, int m, int n,
                int k) {
  using T = Config<BK, TM>;
  __shared__ __align__(16) _Float16 lds[2 * T::Size];
  float acc[TM][8] = {};
  U4 ar[T::AV], br[T::BV];
  int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
  int tiles = (k + BK - 1) / BK;
  if (tiles) {
    issue_tile<T>(a, b, m, n, k, 0, ar, br);
    publish<T>(lds, ar, br);
    __syncthreads();
  }
  for (int tile = 0; tile < tiles; ++tile) {
    if constexpr (Prefetch) {
      if (tile + 1 < tiles)
        issue_tile<T>(a, b, m, n, k, tile + 1, ar, br);
    } else if (tile > 0) {
      // Private synchronous scheduling control, with identical arithmetic.
      issue_tile<T>(a, b, m, n, k, tile, ar, br);
      publish<T>(lds + (tile % 2) * T::Size, ar, br);
      __syncthreads();
    }
    const _Float16 *cur = lds + (tile % 2) * T::Size;
#pragma unroll 1
    for (int kk = 0; kk < BK; ++kk) {
      float av[TM], bv[8];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        av[i] = float(cur[kk * T::Stride + ty * (TM / 2) + i % (TM / 2) +
                          (i / (TM / 2)) * 64]);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        bv[j] = float(cur[T::BK * T::Stride + kk * T::Stride + tx * 4 + j % 4 +
                          (j / 4) * 64]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    if constexpr (Prefetch)
      if (tile + 1 < tiles)
        publish<T>(lds + ((tile + 1) % 2) * T::Size, ar, br);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int row =
          blockIdx.y * 128 + ty * (TM / 2) + i % (TM / 2) + (i / (TM / 2)) * 64;
      int col = blockIdx.x * 128 + tx * 4 + j % 4 + (j / 4) * 64;
      if (row < m && col < n)
        c[row * n + col] = (_Float16)acc[i][j];
    }
}
template <int BK, int TM, bool Prefetch = true>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() &&
                  c.device() == a.device(),
              "expected one GPU");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2 &&
                  a.size(1) == b.size(0) && c.size(0) == a.size(0) &&
                  c.size(1) == b.size(1),
              "invalid GEMM shapes");
  for (auto t : {a, b, c}) {
    TORCH_CHECK(t.scalar_type() == at::kHalf && t.is_contiguous(),
                "expected contiguous FP16");
    TORCH_CHECK(!t.requires_grad(), "forward only");
    TORCH_CHECK(t.numel() < INT32_MAX, "index range exceeded");
  }
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b), "output aliases input");
  TORCH_CHECK(a.size(0) < INT32_MAX - 128 && a.size(1) < INT32_MAX - 32 &&
                  b.size(1) < INT32_MAX - 128 &&
                  (a.size(0) + 127) / 128 <= 65535,
              "index/grid range exceeded");
  if (!c.numel())
    return;
  c10::cuda::CUDAGuard guard(a.device());
  dim3 grid((b.size(1) + 127) / 128, (a.size(0) + 127) / 128);
  constexpr int threads = Config<BK, TM>::Threads;
  async_hgemm<BK, TM, Prefetch>
      <<<grid, threads>>>(reinterpret_cast<const _Float16 *>(a.data_ptr()),
                          reinterpret_cast<const _Float16 *>(b.data_ptr()),
                          reinterpret_cast<_Float16 *>(c.data_ptr()), a.size(0),
                          b.size(1), a.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(torch::Tensor a,
                                                  torch::Tensor b,
                                                  torch::Tensor c) {
  launch<16, 8>(a, b, c);
}
void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a,
                                                  torch::Tensor b,
                                                  torch::Tensor c) {
  launch<32, 8>(a, b, c);
}
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(torch::Tensor a,
                                                   torch::Tensor b,
                                                   torch::Tensor c) {
  launch<32, 16>(a, b, c);
}

#ifdef LEETCUDA_ASYNC_HGEMM_CONTROL
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_synchronous_control(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<16, 8, false>(a, b, c);
}
void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_synchronous_control(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<32, 8, false>(a, b, c);
}
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_synchronous_control(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  launch<32, 16, false>(a, b, c);
}
#endif
