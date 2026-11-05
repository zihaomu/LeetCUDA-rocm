// AMD matrix implementations of the ten MMA and one CuTe HGEMM interfaces.
// Explicit precision contract: FP16 I/O, FP32 accumulator. Upstream uses FP16.
// K32 variants retain two register fragments; x4 packs both in a wide LDS read.
// Stage5 K32 and stage3/4 CuTe reduce CTA geometry to fit 64 KiB, not stage
// depth.
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

template <int Mode, int Stages> struct Config {
  static constexpr bool TN = Mode >= 7;
  static constexpr bool Wide = Mode == 4 || Mode == 8 || Mode == 9;
  static constexpr bool Swizzle = Mode == 6 || Mode == 8 || Mode == 9;
  static constexpr bool Dynamic = Mode >= 2;
  static constexpr int BK = Mode >= 3 && Mode != 7 ? 32 : 16;
  static constexpr int BM = BK == 32 && Stages == 5 ? 64 : 128;
  static constexpr int BN = Mode == 9 && Stages == 2 ? 256 : 128;
  static constexpr int WM = 2, WN = 4, RM = BM / (WM * 16), RN = BN / (WN * 16);
  static constexpr int Threads = WM * WN * 32, Size = (BM + BN) * BK;
  static constexpr int AV = (BM * BK + Threads * 8 - 1) / (Threads * 8);
  static constexpr int BV = (BN * BK + Threads * 8 - 1) / (Threads * 8);
  __device__ static __forceinline__ int index(int row, int k) {
    if constexpr (Wide) {
      // Each lane's k[0:8] and k[16:24] are contiguous in one H16.
      k = (k & 7) | ((k & 16) >> 1) | ((k & 8) << 1);
    }
    if constexpr (Swizzle)
      k ^= Wide ? ((row & 1) * 16) : ((row & (BK / 8 - 1)) * 8);
    return row * BK + k;
  }
};

template <class T>
__device__ __forceinline__ void
issue_tile(const _Float16 *a, const _Float16 *b, int m, int n, int k, int bm,
           int bn, int tile, U4 (&ar)[T::AV], U4 (&br)[T::BV]) {
#pragma unroll
  for (int i = 0; i < T::AV; ++i) {
    int idx = (threadIdx.x + i * T::Threads) * 8;
    int row = bm + idx / T::BK, col = tile * T::BK + idx % T::BK;
    if (idx < T::BM * T::BK && row < m && col + 7 < k && k % 8 == 0 &&
        reinterpret_cast<uintptr_t>(a) % 16 == 0) {
      ar[i] = issue8(a + row * k + col);
    } else {
      H8 h = {};
#pragma unroll
      for (int x = 0; x < 8; ++x)
        if (idx < T::BM * T::BK && row < m && col + x < k)
          h[x] = a[row * k + col + x];
      ar[i] = __builtin_bit_cast(U4, h);
    }
  }
#pragma unroll
  for (int i = 0; i < T::BV; ++i) {
    int idx = (threadIdx.x + i * T::Threads) * 8;
    int row = T::TN ? bn + idx / T::BK : tile * T::BK + idx / T::BN;
    int col = T::TN ? tile * T::BK + idx % T::BK : bn + idx % T::BN;
    int height = T::TN ? n : k, width = T::TN ? k : n;
    if (idx < T::BN * T::BK && row < height && col + 7 < width &&
        width % 8 == 0 && reinterpret_cast<uintptr_t>(b) % 16 == 0) {
      br[i] = issue8(b + row * width + col);
    } else {
      H8 h = {};
#pragma unroll
      for (int x = 0; x < 8; ++x)
        if (idx < T::BN * T::BK && row < height && col + x < width)
          h[x] = b[row * width + col + x];
      br[i] = __builtin_bit_cast(U4, h);
    }
  }
}

template <class T>
__device__ __forceinline__ void publish(_Float16 *lds, U4 (&ar)[T::AV],
                                        U4 (&br)[T::BV]) {
#pragma unroll
  for (int i = 0; i < T::AV; ++i) {
    H8 h = finish8(ar[i]);
    int idx = (threadIdx.x + i * T::Threads) * 8;
    if (idx < T::BM * T::BK)
#pragma unroll
      for (int x = 0; x < 8; ++x)
        lds[T::index(idx / T::BK, idx % T::BK + x)] = h[x];
  }
#pragma unroll
  for (int i = 0; i < T::BV; ++i) {
    H8 h = finish8(br[i]);
    int idx = (threadIdx.x + i * T::Threads) * 8;
    if (idx < T::BN * T::BK)
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        int row = T::TN ? idx / T::BK : idx % T::BN + x;
        int kk = T::TN ? idx % T::BK + x : idx / T::BN;
        lds[T::BM * T::BK + T::index(row, kk)] = h[x];
      }
  }
}

template <class T, int Rows>
__device__ __forceinline__ void fragments(const _Float16 *p, int warp,
                                          H8 (&frags)[T::BK / 16][Rows]) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int r = 0; r < Rows; ++r) {
    int row = (warp * Rows + r) * 16 + lane % 16;
    if constexpr (T::Wide) {
      H16 both =
          *reinterpret_cast<const H16 *>(p + T::index(row, lane / 16 * 8));
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        frags[0][r][x] = both[x];
        frags[1][r][x] = both[8 + x];
      }
    } else {
#pragma unroll
      for (int kk = 0; kk < T::BK / 16; ++kk)
#pragma unroll
        for (int x = 0; x < 8; ++x)
          frags[kk][r][x] = p[T::index(row, kk * 16 + lane / 16 * 8 + x)];
    }
  }
}

template <int Mode, int Stages>
__global__ void __launch_bounds__(256)
    mma_tiled(const _Float16 *a, const _Float16 *b, _Float16 *c, int m, int n,
              int k) {
  using T = Config<Mode, Stages>;
  extern __shared__ __align__(32) _Float16 dynamic_lds[];
  _Float16 *lds;
  if constexpr (T::Dynamic) {
    lds = dynamic_lds;
  } else {
    __shared__ __align__(32) _Float16 static_lds[Stages * T::Size];
    lds = static_lds;
  }
  int bm = blockIdx.y * T::BM;
  int bn = (blockIdx.z * gridDim.x + blockIdx.x) * T::BN;
  if (bm >= m || bn >= n)
    return;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int wr = warp / T::WN, wc = warp % T::WN;
  int tiles = (k + T::BK - 1) / T::BK;
  U4 ar[T::AV], br[T::BV];
  F8 acc[T::RM][T::RN] = {};
  if constexpr (Stages > 1) {
#pragma unroll
    for (int t = 0; t < Stages - 1; ++t)
      if (t < tiles) {
        issue_tile<T>(a, b, m, n, k, bm, bn, t, ar, br);
        publish<T>(lds + t * T::Size, ar, br);
      }
    __syncthreads();
  }
  for (int t = 0; t < tiles; ++t) {
    int future = t + Stages - 1;
    if (future < tiles)
      issue_tile<T>(a, b, m, n, k, bm, bn, future, ar, br);
    if constexpr (Stages == 1) {
      publish<T>(lds, ar, br);
      __syncthreads();
    }
    const _Float16 *cur = lds + (t % Stages) * T::Size;
    if constexpr (Mode == 5) {
      // rr: load one B fragment per use instead of keeping both K panels of
      // the entire B warp tile live, with the same FP32 accumulation order.
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int i = 0; i < T::RM; ++i) {
          H8 av;
#pragma unroll
          for (int x = 0; x < 8; ++x)
            av[x] = cur[T::index((wr * T::RM + i) * 16 + lane % 16,
                                 kk * 16 + lane / 16 * 8 + x)];
#pragma unroll
          for (int j = 0; j < T::RN; ++j) {
            H8 bv;
#pragma unroll
            for (int x = 0; x < 8; ++x)
              bv[x] = cur[T::BM * T::BK +
                          T::index((wc * T::RN + j) * 16 + lane % 16,
                                   kk * 16 + lane / 16 * 8 + x)];
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                av, bv, acc[i][j]);
          }
        }
    } else {
      H8 af[T::BK / 16][T::RM], bf[T::BK / 16][T::RN];
      fragments<T, T::RM>(cur, wr, af);
      fragments<T, T::RN>(cur + T::BM * T::BK, wc, bf);
#pragma unroll
      for (int kk = 0; kk < T::BK / 16; ++kk)
#pragma unroll
        for (int i = 0; i < T::RM; ++i)
#pragma unroll
          for (int j = 0; j < T::RN; ++j)
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                af[kk][i], bf[kk][j], acc[i][j]);
    }
    if constexpr (Stages > 1)
      if (future < tiles)
        publish<T>(lds + (future % Stages) * T::Size, ar, br);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < T::RM; ++i)
#pragma unroll
    for (int j = 0; j < T::RN; ++j) {
      int col = bn + (wc * T::RN + j) * 16 + lane % 16;
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        int row = bm + (wr * T::RM + i) * 16 + lane / 16 * 8 + x;
        if (row < m && col < n)
          c[row * n + col] = (_Float16)acc[i][j][x];
      }
    }
}

__global__ void mma_naive(const _Float16 *a, const _Float16 *b, _Float16 *c,
                          int m, int n, int k) {
  int lane = threadIdx.x, col = blockIdx.x * 8 + lane % 16;
  int row = blockIdx.y * 16 + lane % 16;
  F8 acc = {};
  for (int base = 0; base < k; base += 16) {
    H8 av = {}, bv = {};
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      int kk = base + lane / 16 * 8 + x;
      if (row < m && kk < k)
        av[x] = a[row * k + kk];
      if (lane % 16 < 8 && col < n && kk < k)
        bv[x] = b[kk * n + col];
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
  }
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    int r = blockIdx.y * 16 + lane / 16 * 8 + x;
    if (lane % 16 < 8 && col < n && r < m)
      c[r * n + col] = (_Float16)acc[x];
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
              "expected A[M,K], B[K,N], C[M,N]; TN B stores packed [N,K]");
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

template <int Mode, int S>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c, bool swizzle,
            int stride) {
  using T = Config<Mode, S>;
  static_assert(S * T::Size * 2 <= 65536);
  int m = a.size(0), k = a.size(1), n = b.size(1);
  if (!c.numel())
    return;
  int z = swizzle ? 1 + (n - 1) / stride : 1;
  int nx = (1 + (n - 1) / T::BN + z - 1) / z;
  TORCH_CHECK(z <= 65535 && (m + T::BM - 1) / T::BM <= 65535,
              "grid limit exceeded");
  c10::cuda::CUDAGuard guard(a.device());
  dim3 grid(nx, (m + T::BM - 1) / T::BM, z);
  constexpr int bytes = T::Dynamic ? S * T::Size * 2 : 0;
  mma_tiled<Mode, S><<<grid, T::Threads, bytes>>>(
      reinterpret_cast<const _Float16 *>(a.data_ptr()),
      reinterpret_cast<const _Float16 *>(b.data_ptr()),
      reinterpret_cast<_Float16 *>(c.data_ptr()), m, n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Mode>
void staged(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages,
            bool swizzle, int stride) {
  check(a, b, c);
  TORCH_CHECK(!swizzle || stride > 0, "swizzle_stride must be positive");
  constexpr int MaxStage = Mode == 1 || Mode == 9 ? 4 : 5;
  // Original dispatch uses stage2 for every value outside the supported set.
  if (stages < 2 || stages > MaxStage)
    stages = 2;
  switch (stages) {
  case 2:
    launch<Mode, 2>(a, b, c, swizzle, stride);
    break;
  case 3:
    launch<Mode, 3>(a, b, c, swizzle, stride);
    break;
  case 4:
    launch<Mode, 4>(a, b, c, swizzle, stride);
    break;
  case 5:
    if constexpr (MaxStage == 5)
      launch<Mode, 5>(a, b, c, swizzle, stride);
    break;
  }
}
} // namespace

void hgemm_mma_m16n8k16_naive(torch::Tensor a, torch::Tensor b,
                              torch::Tensor c) {
  check(a, b, c);
  if (!c.numel())
    return;
  int m = a.size(0), k = a.size(1), n = b.size(1);
  TORCH_CHECK((m + 15) / 16 <= 65535, "grid limit exceeded");
  c10::cuda::CUDAGuard guard(a.device());
  dim3 grid((n + 7) / 8, (m + 15) / 16);
  mma_naive<<<grid, 32>>>(reinterpret_cast<const _Float16 *>(a.data_ptr()),
                          reinterpret_cast<const _Float16 *>(b.data_ptr()),
                          reinterpret_cast<_Float16 *>(c.data_ptr()), m, n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void hgemm_mma_m16n8k16_mma2x4_warp4x4(torch::Tensor a, torch::Tensor b,
                                       torch::Tensor c) {
  check(a, b, c);
  launch<0, 1>(a, b, c, false, 256);
}
#define STAGED(NAME, MODE)                                                     \
  void NAME(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages,     \
            bool swizzle, int stride) {                                        \
    staged<MODE>(a, b, c, stages, swizzle, stride);                            \
  }
STAGED(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages, 1)
STAGED(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem, 2)
STAGED(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem, 3)
STAGED(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4, 4)
STAGED(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr, 5)
STAGED(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle, 6)
STAGED(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn, 7)
STAGED(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4, 8)
STAGED(hgemm_mma_stages_block_swizzle_tn_cute, 9)
