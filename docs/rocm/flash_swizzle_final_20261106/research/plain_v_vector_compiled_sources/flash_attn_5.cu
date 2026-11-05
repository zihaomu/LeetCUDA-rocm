// Eight-wave split-KV: two Q groups x four KV groups, with real cross-wave
// max/sum reduction and P exchange in LDS. QK explicitly upgrades to FP32;
// PV and running output retain native FP16 accumulation/storage.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>

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
  asm volatile("s_wait_loadcnt 0" : "+v"(r)::"memory");
  return __builtin_bit_cast(H8, r);
}

template <int Rows, int D, int Threads>
__device__ __forceinline__ void issue(const _Float16 *src, int row0, int n,
                                      U4 *r) {
  constexpr int Count = (Rows * D + Threads * 8 - 1) / (Threads * 8);
#pragma unroll
  for (int i = 0; i < Count; ++i) {
    int x = (threadIdx.x + i * Threads) * 8;
    int row = row0 + x / D, col = x % D;
    if (x < Rows * D && row < n && reinterpret_cast<uintptr_t>(src) % 16 == 0) {
      r[i] = issue8(src + row * D + col);
    } else {
      H8 h = {};
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (x < Rows * D && row < n)
          h[j] = src[row * D + col + j];
      r[i] = __builtin_bit_cast(U4, h);
    }
  }
}

// XOR within 16-half panels. For V, apply after transposition, on its
// reduction coordinate. This is an involution, preserves eight-half vectors and
// changes LDS addresses without changing any matrix/softmax arithmetic.
template <int Cols, bool Swizzle>
__host__ __device__ __forceinline__ int lds_index(int row, int col) {
  return row * Cols + (Swizzle ? (col ^ (((row >> 2) & 1) << 3)) : col);
}

template <int Rows, int Cols, int Threads, bool Transpose = false,
          bool Swizzle = false>
__device__ __forceinline__ void publish(_Float16 *dst, U4 *r) {
  constexpr int Count = (Rows * Cols + Threads * 8 - 1) / (Threads * 8);
#pragma unroll
  for (int i = 0; i < Count; ++i) {
    H8 h = finish8(r[i]);
    int x = (threadIdx.x + i * Threads) * 8;
    if (x < Rows * Cols) {
      if constexpr (Swizzle && !Transpose) {
        // The permutation preserves each aligned eight-half vector. Express
        // that guarantee explicitly; per-element XOR inhibits LLVM combining.
        *reinterpret_cast<H8 *>(dst +
                                lds_index<Cols, true>(x / Cols, x % Cols)) = h;
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          dst[Transpose ? lds_index<Rows, Swizzle>(x % Cols + j, x / Cols)
                        : lds_index<Cols, Swizzle>(x / Cols, x % Cols + j)] =
              h[j];
      }
    }
  }
}

template <int D, int Stage> struct KVConfig {
  static constexpr int BR = 64, BC = 64, Threads = 256;
  static constexpr int Q = BR * D, Data = Q + Stage * BC * D;
  static constexpr int HalfElements = Data + BR * BC;
  static constexpr int Bytes = 2 * HalfElements + 2 * BR * 4 * sizeof(float);
  static constexpr int Loads = (BC * D + Threads * 8 - 1) / (Threads * 8);
  static constexpr int OD = D / 4, Tiles = (OD + 15) / 16;
};

template <int D, int Stage>
__global__ void __launch_bounds__(256)
    attention_split_kv(const _Float16 *q, const _Float16 *k, const _Float16 *v,
                       _Float16 *o, int n) {
  using C = KVConfig<D, Stage>;
  constexpr int BR = 64, BC = 64, Threads = 256;
  extern __shared__ __align__(16) _Float16 lds[];
  _Float16 *sq = lds, *sk = sq + C::Q;
  _Float16 *sv = sk + (Stage - 1) * BC * D, *p = lds + C::Data;
  float *maxima = reinterpret_cast<float *>(lds + C::HalfElements);
  float *sums = maxima + BR * 4;
  int base = blockIdx.y * n * D, q0 = blockIdx.x * BR;
  q += base;
  k += base;
  v += base;
  o += base;
  int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  int group = wave % 4, query_group = wave / 4;
  U4 qr[C::Loads], kr[C::Loads], vr[C::Loads];
  issue<BR, D, Threads>(q, q0, n, qr);
  publish<BR, D, Threads>(sq, qr);
  __syncthreads();
  H8 output[2][C::Tiles] = {};
  F8 maximum[2], denominator[2] = {};
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int x = 0; x < 8; ++x)
      maximum[i][x] = -INFINITY;
  float scale = rsqrtf(float(D));
  if constexpr (Stage == 2) {
    issue<BC, D, Threads>(k, 0, n, kr);
    publish<BC, D, Threads>(sk, kr);
    __syncthreads();
  }
  for (int key0 = 0; key0 < n; key0 += BC) {
    if constexpr (Stage == 1) {
      issue<BC, D, Threads>(k, key0, n, kr);
      publish<BC, D, Threads>(sk, kr);
      __syncthreads();
    } else
      issue<BC, D, Threads>(v, key0, n, vr);
    F8 score[2] = {};
#pragma unroll
    for (int d = 0; d < D / 16; ++d) {
      H8 b;
#pragma unroll
      for (int x = 0; x < 8; ++x)
        b[x] = sk[(group * 16 + lane % 16) * D + d * 16 + lane / 16 * 8 + x];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        H8 a;
#pragma unroll
        for (int x = 0; x < 8; ++x)
          a[x] = sq[(query_group * 32 + i * 16 + lane % 16) * D + d * 16 +
                    lane / 16 * 8 + x];
        score[i] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, score[i]);
      }
    }
    // Each of four KV waves owns 16 score columns. Its 32 query rows
    // participate in a cross-wave row maximum, then a cross-wave row sum.
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        score[i][x] =
            key0 + group * 16 + lane % 16 < n ? score[i][x] * scale : -INFINITY;
        float mx = score[i][x];
#pragma unroll
        for (int delta = 8; delta > 0; delta /= 2)
          mx = fmaxf(mx, __shfl_xor(mx, delta, 16));
        int row = query_group * 32 + i * 16 + lane / 16 * 8 + x;
        if (lane % 16 == 0)
          maxima[row * 4 + group] = mx;
      }
    __syncthreads();
    if constexpr (Stage == 1)
      issue<BC, D, Threads>(v, key0, n, vr);
    publish<BC, D, Threads, true>(sv, vr);
    if constexpr (Stage == 2)
      if (key0 + BC < n)
        issue<BC, D, Threads>(k, key0 + BC, n, kr);
    F8 alpha[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        int row = query_group * 32 + i * 16 + lane / 16 * 8 + x;
        float mx = maximum[i][x];
#pragma unroll
        for (int g = 0; g < 4; ++g)
          mx = fmaxf(mx, maxima[row * 4 + g]);
        alpha[i][x] = expf(maximum[i][x] - mx);
        float prob = expf(score[i][x] - mx), sum = prob;
        p[row * BC + group * 16 + lane % 16] = (_Float16)prob;
#pragma unroll
        for (int delta = 8; delta > 0; delta /= 2)
          sum += __shfl_xor(sum, delta, 16);
        if (lane % 16 == 0)
          sums[row * 4 + group] = sum;
        maximum[i][x] = mx;
      }
    // Publish all four probability partitions before every wave consumes P.
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        int row = query_group * 32 + i * 16 + lane / 16 * 8 + x;
        float sum = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g)
          sum += sums[row * 4 + g];
        denominator[i][x] = fmaf(alpha[i][x], denominator[i][x], sum);
      }
#pragma unroll
      for (int d = 0; d < C::Tiles; ++d) {
        H8 pv = {};
#pragma unroll
        for (int j = 0; j < BC / 16; ++j) {
          H8 a, b;
#pragma unroll
          for (int x = 0; x < 8; ++x) {
            a[x] = p[(query_group * 32 + i * 16 + lane % 16) * BC + j * 16 +
                     lane / 16 * 8 + x];
            int col = d * 16 + lane % 16;
            b[x] = col < C::OD ? sv[(group * C::OD + col) * BC + j * 16 +
                                    lane / 16 * 8 + x]
                               : (_Float16)0;
          }
          pv = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, pv);
        }
#pragma unroll
        for (int x = 0; x < 8; ++x)
          output[i][d][x] =
              (_Float16)fmaf(alpha[i][x], float(output[i][d][x]), float(pv[x]));
      }
    }
    if constexpr (Stage == 2)
      if (key0 + BC < n)
        publish<BC, D, Threads>(sk, kr);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int d = 0; d < C::Tiles; ++d)
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        int row = q0 + query_group * 32 + i * 16 + lane / 16 * 8 + x;
        int col = d * 16 + lane % 16;
        if (row < n && col < C::OD)
          o[row * D + group * C::OD + col] =
              (_Float16)(float(output[i][d][x]) / denominator[i][x]);
      }
}
void check(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {
  TORCH_CHECK(q.is_cuda() && k.device() == q.device() &&
                  v.device() == q.device() && o.device() == q.device(),
              "expected one GPU");
  TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() &&
                  q.sizes() == v.sizes() && q.sizes() == o.sizes(),
              "expected equal [B,H,N,D] shapes");
  for (auto t : {q, k, v, o}) {
    TORCH_CHECK(t.scalar_type() == at::kHalf && t.is_contiguous(),
                "expected contiguous FP16");
    TORCH_CHECK(!t.requires_grad(), "forward only");
  }
  TORCH_CHECK(!o.is_alias_of(q) && !o.is_alias_of(k) && !o.is_alias_of(v),
              "output aliases input");
  TORCH_CHECK(q.numel() < INT32_MAX && q.size(2) < INT32_MAX - 128 &&
                  q.size(0) <= 65535 && q.size(1) <= 65535 &&
                  q.size(0) * q.size(1) <= 65535,
              "index or grid range exceeded");
}

template <int D, int Stage>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
            torch::Tensor o) {
  using C = KVConfig<D, Stage>;
  static_assert(C::Bytes <= 65536);
  dim3 grid((q.size(2) + 63) / 64, q.size(0) * q.size(1));
  attention_split_kv<D, Stage><<<grid, 256, C::Bytes, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const _Float16 *>(q.data_ptr()),
      reinterpret_cast<const _Float16 *>(k.data_ptr()),
      reinterpret_cast<const _Float16 *>(v.data_ptr()),
      reinterpret_cast<_Float16 *>(o.data_ptr()), q.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
void flash_attn_mma_stages_split_kv(torch::Tensor q, torch::Tensor k,
                                    torch::Tensor v, torch::Tensor o,
                                    int stages) {
  check(q, k, v, o);
  int d = q.size(3);
  TORCH_CHECK(d == 32 || d == 64 || d == 96 || d == 128,
              "unsupported head dimension");
  if (!q.numel())
    return;
  c10::cuda::CUDAGuard guard(q.device());
#define HEAD(D)                                                                \
  case D:                                                                      \
    if (stages > 1)                                                            \
      launch<D, 2>(q, k, v, o);                                                \
    else                                                                       \
      launch<D, 1>(q, k, v, o);                                                \
    break;
  switch (d) { HEAD(32) HEAD(64) HEAD(96) HEAD(128) }
#undef HEAD
}
