// Original tiling-QK/QKV attention on gfx1201. Both products accumulate in
// FP32; the running output is rounded to FP16 after each key block at D>=256,
// matching the original CUDA storage contract (despite the acc_f32 name).
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
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

template <int Rows, int Cols, int D, int Threads>
__device__ __forceinline__ void issue(const _Float16 *src, int row0, int col0,
                                      int n, U4 *r) {
  constexpr int Count = (Rows * Cols + Threads * 8 - 1) / (Threads * 8);
#pragma unroll
  for (int i = 0; i < Count; ++i) {
    int x = (threadIdx.x + i * Threads) * 8;
    int row = row0 + x / Cols, col = col0 + x % Cols;
    if (x < Rows * Cols && row < n &&
        reinterpret_cast<uintptr_t>(src) % 16 == 0) {
      r[i] = issue8(src + row * D + col);
    } else {
      H8 h = {};
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (x < Rows * Cols && row < n)
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

template <int D, int Stage, bool TileV> struct Config {
  static constexpr int BR = D < 128 ? 64 : 128, BC = BR;
  static constexpr int OD = D < 128 ? D : 128;
  static constexpr int Threads = BR * 2;
  static constexpr int QK = (BR + BC) * 16;
  static constexpr int V = TileV ? BC * 16 : 16 * OD;
  static constexpr int Data = Stage * (QK > V ? QK : V);
  static constexpr int HalfElements = Data + BR * BC;
  static constexpr int QLoads = (BR * 16 + Threads * 8 - 1) / (Threads * 8);
  static constexpr int KLoads = (BC * 16 + Threads * 8 - 1) / (Threads * 8);
  static constexpr int VLoads = (V + Threads * 8 - 1) / (Threads * 8);
};

template <int D, int Stage, bool TileV, int Mask = 0>
__global__ void __launch_bounds__((D < 128 ? 64 : 128) * 2)
    attention_tiled(const _Float16 *q, const _Float16 *k, const _Float16 *v,
                    _Float16 *o, int n) {
  using C = Config<D, Stage, TileV>;
  constexpr int BR = C::BR, BC = C::BC, OD = C::OD, Threads = C::Threads;
  using State = std::conditional_t<(D >= 256), H8, F8>;
  extern __shared__ __align__(16) _Float16 smem[];
  _Float16 *p = smem + C::Data;
  int base = blockIdx.y * n * D;
  q += base;
  k += base;
  v += base;
  o += base;
  int q0 = blockIdx.x * BR, od0 = blockIdx.z * OD;
  int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  State output[OD / 16] = {};
  F8 maximum, denominator = {};
#pragma unroll
  for (int x = 0; x < 8; ++x)
    maximum[x] = -INFINITY;
  float scale = rsqrtf(float(D));
  for (int key0 = 0; key0 < n; key0 += BC) {
    F8 score[BC / 16] = {};
    {
      U4 qr[C::QLoads], kr[C::KLoads];
      if constexpr (Stage == 2) {
        issue<BR, 16, D, Threads>(q, q0, 0, n, qr);
        issue<BC, 16, D, Threads>(k, key0, 0, n, kr);
        publish<BR, 16, Threads, false, (Mask & 1) != 0>(smem, qr);
        publish<BC, 16, Threads, false, (Mask & 2) != 0>(smem + BR * 16, kr);
        __syncthreads();
      }
#pragma unroll 1
      for (int tile = 0; tile < D / 16; ++tile) {
        int future = tile + Stage - 1;
        if (future < D / 16) {
          issue<BR, 16, D, Threads>(q, q0, future * 16, n, qr);
          issue<BC, 16, D, Threads>(k, key0, future * 16, n, kr);
        }
        if constexpr (Stage == 1) {
          publish<BR, 16, Threads, false, (Mask & 1) != 0>(smem, qr);
          publish<BC, 16, Threads, false, (Mask & 2) != 0>(smem + BR * 16, kr);
          __syncthreads();
        }
        const _Float16 *curr = smem + (tile % Stage) * C::QK;
        H8 a;
        if constexpr ((Mask & 1) != 0) {
          a = *reinterpret_cast<const H8 *>(
              curr + lds_index<16, true>(wave * 16 + lane % 16, lane / 16 * 8));
        } else {
#pragma unroll
          for (int x = 0; x < 8; ++x)
            a[x] = curr[(wave * 16 + lane % 16) * 16 + lane / 16 * 8 + x];
        }
#pragma unroll
        for (int j = 0; j < BC / 16; ++j) {
          H8 b;
          if constexpr ((Mask & 2) != 0) {
            b = *reinterpret_cast<const H8 *>(
                curr + BR * 16 +
                lds_index<16, true>(j * 16 + lane % 16, lane / 16 * 8));
          } else {
#pragma unroll
            for (int x = 0; x < 8; ++x)
              b[x] =
                  curr[BR * 16 + (j * 16 + lane % 16) * 16 + lane / 16 * 8 + x];
          }
          score[j] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, score[j]);
        }
        if constexpr (Stage == 2) {
          if (future < D / 16) {
            _Float16 *next = smem + (future % Stage) * C::QK;
            publish<BR, 16, Threads, false, (Mask & 1) != 0>(next, qr);
            publish<BC, 16, Threads, false, (Mask & 2) != 0>(next + BR * 16,
                                                             kr);
          }
        }
        __syncthreads();
      }
    }
    F8 alpha;
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < BC / 16; ++j) {
        score[j][x] =
            key0 + j * 16 + lane % 16 < n ? score[j][x] * scale : -INFINITY;
        mx = fmaxf(mx, score[j][x]);
      }
#pragma unroll
      for (int delta = 8; delta > 0; delta /= 2)
        mx = fmaxf(mx, __shfl_xor(mx, delta, 16));
      mx = fmaxf(maximum[x], mx);
      alpha[x] = expf(maximum[x] - mx);
      float sum = 0;
#pragma unroll
      for (int j = 0; j < BC / 16; ++j) {
        float prob = expf(score[j][x] - mx);
        sum += prob;
        p[(wave * 16 + lane / 16 * 8 + x) * BC + j * 16 + lane % 16] =
            (_Float16)prob;
      }
#pragma unroll
      for (int delta = 8; delta > 0; delta /= 2)
        sum += __shfl_xor(sum, delta, 16);
      denominator[x] = fmaf(alpha[x], denominator[x], sum);
      maximum[x] = mx;
    }
    __syncthreads();
    U4 vr[C::VLoads];
    if constexpr (!TileV) {
      // tiling-QK: V is striped along its reduction dimension, 16 x OD.
      F8 pv[OD / 16] = {};
      if constexpr (Stage == 2) {
        issue<16, OD, D, Threads>(v, key0, od0, n, vr);
        publish<16, OD, Threads, true>(smem, vr);
        __syncthreads();
      }
#pragma unroll 1
      for (int tile = 0; tile < BC / 16; ++tile) {
        int future = tile + Stage - 1;
        if (future < BC / 16)
          issue<16, OD, D, Threads>(v, key0 + future * 16, od0, n, vr);
        if constexpr (Stage == 1) {
          publish<16, OD, Threads, true>(smem, vr);
          __syncthreads();
        }
        const _Float16 *curr = smem + (tile % Stage) * C::V;
        H8 a;
#pragma unroll
        for (int x = 0; x < 8; ++x)
          a[x] =
              p[(wave * 16 + lane % 16) * BC + tile * 16 + lane / 16 * 8 + x];
#pragma unroll
        for (int d = 0; d < OD / 16; ++d) {
          H8 b;
#pragma unroll
          for (int x = 0; x < 8; ++x)
            b[x] = curr[(d * 16 + lane % 16) * 16 + lane / 16 * 8 + x];
          pv[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, pv[d]);
        }
        if constexpr (Stage == 2) {
          if (future < BC / 16)
            publish<16, OD, Threads, true>(smem + (future % Stage) * C::V, vr);
        }
        __syncthreads();
      }
#pragma unroll
      for (int d = 0; d < OD / 16; ++d)
#pragma unroll
        for (int x = 0; x < 8; ++x)
          output[d][x] = fmaf(alpha[x], float(output[d][x]), pv[d][x]);
    } else {
      // tiling-QKV: V is striped along output D, BC x 16. Only one PV
      // accumulator tile is live at a time; the QK LDS region is reused.
      if constexpr (Stage == 2) {
        issue<BC, 16, D, Threads>(v, key0, od0, n, vr);
        publish<BC, 16, Threads, true, (Mask & 4) != 0>(smem, vr);
        __syncthreads();
      }
#pragma unroll
      for (int d = 0; d < OD / 16; ++d) {
        int future = d + Stage - 1;
        if (future < OD / 16)
          issue<BC, 16, D, Threads>(v, key0, od0 + future * 16, n, vr);
        if constexpr (Stage == 1) {
          publish<BC, 16, Threads, true, (Mask & 4) != 0>(smem, vr);
          __syncthreads();
        }
        const _Float16 *curr = smem + (d % Stage) * C::V;
        F8 pv = {};
#pragma unroll
        for (int j = 0; j < BC / 16; ++j) {
          H8 a, b;
#pragma unroll
          for (int x = 0; x < 8; ++x) {
            a[x] = p[(wave * 16 + lane % 16) * BC + j * 16 + lane / 16 * 8 + x];
            if constexpr ((Mask & 4) == 0)
              b[x] = curr[(lane % 16) * BC + j * 16 + lane / 16 * 8 + x];
          }
          if constexpr ((Mask & 4) != 0)
            b = *reinterpret_cast<const H8 *>(
                curr + lds_index<BC, true>(lane % 16, j * 16 + lane / 16 * 8));
          pv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, pv);
        }
#pragma unroll
        for (int x = 0; x < 8; ++x)
          output[d][x] = fmaf(alpha[x], float(output[d][x]), pv[x]);
        if constexpr (Stage == 2) {
          if (future < OD / 16)
            publish<BC, 16, Threads, true, (Mask & 4) != 0>(
                smem + (future % Stage) * C::V, vr);
        }
        __syncthreads();
      }
    }
  }
#pragma unroll
  for (int d = 0; d < OD / 16; ++d)
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      int row = q0 + wave * 16 + lane / 16 * 8 + x;
      if (row < n)
        o[row * D + od0 + d * 16 + lane % 16] =
            (_Float16)(float(output[d][x]) / denominator[x]);
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
template <int D, int Stage, bool TileV, int Mask = 0>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
            torch::Tensor o) {
  using C = Config<D, Stage, TileV>;
  static_assert(C::HalfElements * 2 <= 65536);
  dim3 grid((q.size(2) + C::BR - 1) / C::BR, q.size(0) * q.size(1), D / C::OD);
  attention_tiled<D, Stage, TileV, Mask>
      <<<grid, C::Threads, C::HalfElements * 2>>>(
          reinterpret_cast<const _Float16 *>(q.data_ptr()),
          reinterpret_cast<const _Float16 *>(k.data_ptr()),
          reinterpret_cast<const _Float16 *>(v.data_ptr()),
          reinterpret_cast<_Float16 *>(o.data_ptr()), q.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <bool TileV, int Mask = 0>
void dispatch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
              torch::Tensor o, int stages) {
  check(q, k, v, o);
  int d = q.size(3);
  TORCH_CHECK(d == 32 || d == 64 || d == 96 || d == 128 || d == 256 ||
                  d == 512 || d == 1024,
              "unsupported head dimension");
  if (!q.numel())
    return;
  c10::cuda::CUDAGuard guard(q.device());
#define HEAD(D)                                                                \
  case D:                                                                      \
    if (stages > 1)                                                            \
      launch<D, 2, TileV, Mask>(q, k, v, o);                                   \
    else                                                                       \
      launch<D, 1, TileV, Mask>(q, k, v, o);                                   \
    break;
  switch (d) {
    HEAD(32) HEAD(64) HEAD(96) HEAD(128) HEAD(256) HEAD(512) HEAD(1024)
  }
#undef HEAD
}
} // namespace

void flash_attn_mma_stages_split_q_tiling_qk_acc_f32(torch::Tensor q,
                                                     torch::Tensor k,
                                                     torch::Tensor v,
                                                     torch::Tensor o,
                                                     int stages) {
  dispatch<false>(q, k, v, o, stages);
}
void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32(torch::Tensor q,
                                                      torch::Tensor k,
                                                      torch::Tensor v,
                                                      torch::Tensor o,
                                                      int stages) {
  dispatch<true>(q, k, v, o, stages);
}

void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_q(torch::Tensor q,
                                                                torch::Tensor k,
                                                                torch::Tensor v,
                                                                torch::Tensor o,
                                                                int stages) {
  dispatch<true, 1>(q, k, v, o, stages);
}

void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qk(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    int stages) {
  dispatch<true, 3>(q, k, v, o, stages);
}

void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    int stages) {
  dispatch<true, 7>(q, k, v, o, stages);
}
