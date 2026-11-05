// RDNA4 implementations of the original nine WMMA HGEMM entrypoints.
// FP16 I/O; deliberately FP32 accumulation (original NVIDIA accumulator FP16).
// Preserve block/warp output tiles, staging depth and block swizzle. cp.async
// is replaced by VGPR prefetch followed by explicit load waits and LDS writes.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using U4 = unsigned __attribute__((ext_vector_type(4)));

__device__ __forceinline__ U4 issue8(const _Float16 *p) {
  U4 r;
  asm volatile("global_load_b128 %0, %1, off" : "=v"(r) : "v"(p) : "memory");
  return r;
}
__device__ __forceinline__ H8 finish8(U4 &r) {
  // The dependency includes register moves, not just memory operations.
  asm volatile("s_wait_loadcnt 0" : "+v"(r)::"memory");
  return __builtin_bit_cast(H8, r);
}

__global__ void wmma_global(const _Float16 *a, const _Float16 *b, _Float16 *c,
                            int m, int n, int k) {
  int lane = threadIdx.x;
  int mr = blockIdx.y * 16 + lane % 16;
  int nc = blockIdx.x * 16 + lane % 16;
  F8 acc = {};
  for (int base = 0; base < k; base += 16) {
    H8 av = {}, bv = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int kk = base + lane / 16 * 8 + i;
      av[i] = mr < m && kk < k ? a[mr * k + kk] : (_Float16)0;
      bv[i] = nc < n && kk < k ? b[kk * n + nc] : (_Float16)0;
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int row = blockIdx.y * 16 + lane / 16 * 8 + i;
    if (row < m && nc < n)
      c[row * n + nc] = (_Float16)acc[i];
  }
}

template <int WM, int WN, int RM, int RN> struct Tile {
  static constexpr int BM = WM * RM * 16, BN = WN * RN * 16;
  static constexpr int Threads = WM * WN * 32;
  static constexpr int AV = (BM * 2 + Threads - 1) / Threads;
  static constexpr int BV = (BN * 2 + Threads - 1) / Threads;
  static constexpr int Size = (BM + BN) * 16;
};

template <class T>
__device__ __forceinline__ void
issue_tile(const _Float16 *a, const _Float16 *b, int m, int n, int k, int bm,
           int bn, int tile, U4 (&ar)[T::AV], U4 (&br)[T::BV]) {
  int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < T::AV; ++i) {
    int index = (tid + i * T::Threads) * 8;
    int row = bm + index / 16, col = tile * 16 + index % 16;
    if (index < T::BM * 16 && row < m && col + 7 < k && k % 8 == 0 &&
        reinterpret_cast<uintptr_t>(a) % 16 == 0) {
      ar[i] = issue8(a + row * k + col);
    } else {
      H8 h = {};
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (index < T::BM * 16 && row < m && col + j < k)
          h[j] = a[row * k + col + j];
      ar[i] = __builtin_bit_cast(U4, h);
    }
  }
#pragma unroll
  for (int i = 0; i < T::BV; ++i) {
    int index = (tid + i * T::Threads) * 8;
    int row = tile * 16 + index / T::BN, col = bn + index % T::BN;
    if (index < T::BN * 16 && row < k && col + 7 < n && n % 8 == 0 &&
        reinterpret_cast<uintptr_t>(b) % 16 == 0) {
      br[i] = issue8(b + row * n + col);
    } else {
      H8 h = {};
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (index < T::BN * 16 && row < k && col + j < n)
          h[j] = b[row * n + col + j];
      br[i] = __builtin_bit_cast(U4, h);
    }
  }
}

template <class T>
__device__ __forceinline__ void publish(_Float16 *smem, U4 (&ar)[T::AV],
                                        U4 (&br)[T::BV]) {
#pragma unroll
  for (int i = 0; i < T::AV; ++i) {
    H8 h = finish8(ar[i]);
    int index = (threadIdx.x + i * T::Threads) * 8;
    if (index < T::BM * 16) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        smem[index + j] = h[j];
    }
  }
#pragma unroll
  for (int i = 0; i < T::BV; ++i) {
    H8 h = finish8(br[i]);
    int index = (threadIdx.x + i * T::Threads) * 8;
    if (index < T::BN * 16) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        // Native AMD B fragment reads a contiguous K vector for each column.
        smem[T::BM * 16 + (index % T::BN + j) * 16 + index / T::BN] = h[j];
    }
  }
}

template <int WM, int WN, int RM, int RN, int Stages, bool Dynamic>
__global__ void __launch_bounds__(WM *WN * 32)
    wmma_tiled(const _Float16 *a, const _Float16 *b, _Float16 *c, int m, int n,
               int k) {
  using T = Tile<WM, WN, RM, RN>;
  extern __shared__ _Float16 dynamic_lds[];
  _Float16 *shared;
  if constexpr (Dynamic) {
    shared = dynamic_lds;
  } else {
    __shared__ _Float16 static_lds[Stages * T::Size];
    shared = static_lds;
  }
  // Original swizzle is a 3-D grid: z partitions the N tile dimension.
  int bn = (blockIdx.z * gridDim.x + blockIdx.x) * T::BN;
  int bm = blockIdx.y * T::BM;
  if (bn >= n || bm >= m)
    return; // uniform across the entire block
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int wr = warp / WN, wc = warp % WN;
  int tiles = (k + 15) / 16;
  U4 ar[T::AV], br[T::BV];
  F8 acc[RM][RN] = {};
  if constexpr (Stages > 1) {
#pragma unroll
    for (int t = 0; t < Stages - 1; ++t) {
      if (t < tiles) {
        issue_tile<T>(a, b, m, n, k, bm, bn, t, ar, br);
        publish<T>(shared + t * T::Size, ar, br);
      }
    }
    __syncthreads();
  }
  for (int t = 0; t < tiles; ++t) {
    int future = t + Stages - 1;
    if (future < tiles)
      issue_tile<T>(a, b, m, n, k, bm, bn, future, ar, br);
    if constexpr (Stages == 1) {
      publish<T>(shared, ar, br);
      __syncthreads();
    }
    const _Float16 *current = shared + (t % Stages) * T::Size;
#pragma unroll
    for (int i = 0; i < RM; ++i) {
      H8 av;
#pragma unroll
      for (int x = 0; x < 8; ++x)
        av[x] =
            current[((wr * RM + i) * 16 + lane % 16) * 16 + lane / 16 * 8 + x];
#pragma unroll
      for (int j = 0; j < RN; ++j) {
        H8 bv;
#pragma unroll
        for (int x = 0; x < 8; ++x)
          bv[x] = current[T::BM * 16 + ((wc * RN + j) * 16 + lane % 16) * 16 +
                          lane / 16 * 8 + x];
        acc[i][j] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc[i][j]);
      }
    }
    if constexpr (Stages > 1) {
      if (future < tiles)
        publish<T>(shared + (future % Stages) * T::Size, ar, br);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RM; ++i) {
#pragma unroll
    for (int j = 0; j < RN; ++j) {
      int col = bn + (wc * RN + j) * 16 + lane % 16;
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        int row = bm + (wr * RM + i) * 16 + lane / 16 * 8 + x;
        if (row < m && col < n)
          c[row * n + col] = (_Float16)acc[i][j][x];
      }
    }
  }
}

void check(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() &&
                  c.device() == a.device(),
              "expected one GPU");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf &&
                  c.scalar_type() == at::kHalf,
              "expected FP16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2 &&
                  a.size(1) == b.size(0) && a.size(0) == c.size(0) &&
                  b.size(1) == c.size(1),
              "expected A[M,K], B[K,N], C[M,N]");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "expected contiguous tensors");
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b), "output aliases input");
  TORCH_CHECK(!a.requires_grad() && !b.requires_grad() && !c.requires_grad(),
              "forward only");
  TORCH_CHECK(a.numel() < INT32_MAX && b.numel() < INT32_MAX &&
                  c.numel() < INT32_MAX && a.size(0) < INT32_MAX - 256 &&
                  a.size(1) < INT32_MAX - 256 && b.size(1) < INT32_MAX - 256,
              "32-bit index range exceeded");
}

template <int WM, int WN, int RM, int RN, int S, bool D>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c,
            bool swizzle = false, int stride = 256) {
  using T = Tile<WM, WN, RM, RN>;
  check(a, b, c);
  TORCH_CHECK(!swizzle || stride > 0, "swizzle_stride must be positive");
  if (!c.numel())
    return;
  c10::cuda::CUDAGuard guard(a.device());
  int m = a.size(0), k = a.size(1), n = b.size(1);
  int z = swizzle ? 1 + (n - 1) / stride : 1;
  int nx = (1 + (n - 1) / T::BN + z - 1) / z;
  dim3 grid(nx, (m + T::BM - 1) / T::BM, z);
  constexpr int bytes = D ? S * T::Size * sizeof(_Float16) : 0;
  static_assert(S * T::Size * sizeof(_Float16) <= 65536);
  wmma_tiled<WM, WN, RM, RN, S, D><<<grid, T::Threads, bytes>>>(
      reinterpret_cast<const _Float16 *>(a.data_ptr()),
      reinterpret_cast<const _Float16 *>(b.data_ptr()),
      reinterpret_cast<_Float16 *>(c.data_ptr()), m, n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int WM, int WN, int RM, int RN, bool D>
void staged(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages,
            bool swizzle, int stride) {
  constexpr int maximum = WM == 4 && WN == 4 ? 4 : 5;
  TORCH_CHECK(stages >= 2 && stages <= maximum,
              "unsupported stages for this tile");
  switch (stages) {
  case 2:
    launch<WM, WN, RM, RN, 2, D>(a, b, c, swizzle, stride);
    break;
  case 3:
    launch<WM, WN, RM, RN, 3, D>(a, b, c, swizzle, stride);
    break;
  case 4:
    launch<WM, WN, RM, RN, 4, D>(a, b, c, swizzle, stride);
    break;
  case 5:
    if constexpr (maximum == 5)
      launch<WM, WN, RM, RN, 5, D>(a, b, c, swizzle, stride);
    break;
  }
}
} // namespace

void hgemm_wmma_m16n16k16_naive(torch::Tensor a, torch::Tensor b,
                                torch::Tensor c) {
  check(a, b, c);
  if (!c.numel())
    return;
  c10::cuda::CUDAGuard guard(a.device());
  dim3 grid((b.size(1) + 15) / 16, (a.size(0) + 15) / 16);
  wmma_global<<<grid, 32>>>(reinterpret_cast<const _Float16 *>(a.data_ptr()),
                            reinterpret_cast<const _Float16 *>(b.data_ptr()),
                            reinterpret_cast<_Float16 *>(c.data_ptr()),
                            a.size(0), b.size(1), a.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#define BASIC(Name, WM, WN, RM, RN, S)                                         \
  void Name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {               \
    launch<WM, WN, RM, RN, S, false>(a, b, c);                                 \
  }
BASIC(hgemm_wmma_m16n16k16_mma4x2, 4, 2, 1, 1, 1)
BASIC(hgemm_wmma_m16n16k16_mma4x2_warp2x4, 4, 2, 2, 4, 1)
BASIC(hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async, 4, 2, 2, 4, 2)
// 32x8 NVIDIA atoms become 16x16 AMD atoms while preserving each warp's 64x32
// output.
BASIC(hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async, 2, 4, 4, 2, 2)
#undef BASIC
#define STAGED(Name, WM, WN, RM, RN, D)                                        \
  void Name(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages,     \
            bool swizzle, int stride) {                                        \
    staged<WM, WN, RM, RN, D>(a, b, c, stages, swizzle, stride);               \
  }
STAGED(hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages, 4, 2, 2, 4, false)
STAGED(hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem, 4, 2, 2, 4, true)
STAGED(hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem, 4, 2, 4, 4, true)
STAGED(hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem, 4, 4, 4, 4, true)
#undef STAGED
