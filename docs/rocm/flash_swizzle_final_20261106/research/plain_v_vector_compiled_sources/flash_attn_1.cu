// RDNA4 split-Q attention: distinct shared-KV and shared-QKV LDS lifetimes.
// FP16 Q/K/V/P/O; FP32 QK, PV, online max/sum and output state, as upstream.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

template <int Rows, int D, int Threads, bool Transpose = false>
__device__ __forceinline__ void publish(_Float16 *dst, U4 *r) {
  constexpr int Count = (Rows * D + Threads * 8 - 1) / (Threads * 8);
#pragma unroll
  for (int i = 0; i < Count; ++i) {
    H8 h = finish8(r[i]);
    int x = (threadIdx.x + i * Threads) * 8;
    if (x < Rows * D) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int dst_index = Transpose ? ((x % D + j) * Rows + x / D) : x + j;
        dst[dst_index] = h[j];
      }
    }
  }
}

template <int D, int Stage, bool ShareQ, bool RR = false> struct Config {
  // D=256 needs fewer Q rows to fit the 64 KiB RDNA4 LDS limit.
  static constexpr int BR = D == 256 ? 64 : 128;
  static constexpr int BC =
      (!ShareQ && D >= 128) || (RR && D == 256 && Stage == 2) ? 32 : 64;
  // Two cooperating warp groups share Q/K/V loads while keeping half the
  // FP32 output state live in each lane. Only one group publishes shared P.
  static constexpr int Planes = RR && D > 64 ? 2 : 1;
  static constexpr int OD = ((D + Planes * 16 - 1) / (Planes * 16)) * 16;
  static constexpr int Threads = BR / 16 * 32 * Planes;
  static constexpr int Q = BR * D, KV = Stage * BC * D;
  static constexpr int Data = ShareQ ? (Q > KV ? Q : KV) : Q + KV;
  static constexpr int HalfElements = Data + BR * BC;
  static constexpr int QLoads = (Q + Threads * 8 - 1) / (Threads * 8);
  static constexpr int KVLoads = (BC * D + Threads * 8 - 1) / (Threads * 8);
};

template <int D, int Stage, bool ShareQ, bool RR = false>
__global__ void __launch_bounds__((D == 256 ? 64 : 128) * 2 *
                                  (RR && D > 64 ? 2 : 1))
    attention_shared(const _Float16 *q, const _Float16 *k, const _Float16 *v,
                     _Float16 *o, int n) {
  using C = Config<D, Stage, ShareQ, RR>;
  constexpr int BR = C::BR, BC = C::BC, OD = C::OD, Threads = C::Threads;
  extern __shared__ _Float16 lds[];
  _Float16 *sq = lds;
  _Float16 *sk = ShareQ ? lds : lds + C::Q;
  _Float16 *sv = sk + (Stage - 1) * BC * D;
  _Float16 *p = lds + C::Data;
  int base = blockIdx.y * n * D;
  q += base;
  k += base;
  v += base;
  o += base;
  int q0 = blockIdx.x * BR;
  int plane = (threadIdx.x / 32) % C::Planes;
  int od0 = plane * OD;
  int lane = threadIdx.x % 32, wave = threadIdx.x / 32 / C::Planes;
  H8 qcache[D / 16];
  {
    U4 qr[C::QLoads];
    issue<BR, D, Threads>(q, q0, n, qr);
    publish<BR, D, Threads>(sq, qr);
    __syncthreads();
    if constexpr (ShareQ) {
#pragma unroll
      for (int d = 0; d < D / 16; ++d)
#pragma unroll
        for (int x = 0; x < 8; ++x)
          qcache[d][x] =
              sq[(wave * 16 + lane % 16) * D + d * 16 + lane / 16 * 8 + x];
      // Q has moved to registers; its entire LDS region can now hold K/V.
      __syncthreads();
    }
  }
  U4 kr[C::KVLoads], vr[C::KVLoads];
  F8 output[OD / 16] = {};
  F8 maximum, denominator = {};
#pragma unroll
  for (int x = 0; x < 8; ++x)
    maximum[x] = -INFINITY;
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
    } else {
      // Keep pending V global loads in VGPRs while QK uses K in LDS.
      issue<BC, D, Threads>(v, key0, n, vr);
    }
    F8 score[BC / 16] = {};
#pragma unroll
    for (int d = 0; d < D / 16; ++d) {
      H8 a;
      if constexpr (ShareQ) {
        a = qcache[d];
      } else {
#pragma unroll
        for (int x = 0; x < 8; ++x)
          a[x] = sq[(wave * 16 + lane % 16) * D + d * 16 + lane / 16 * 8 + x];
      }
#pragma unroll
      for (int j = 0; j < BC / 16; ++j) {
        H8 b;
#pragma unroll
        for (int x = 0; x < 8; ++x)
          b[x] = sk[(j * 16 + lane % 16) * D + d * 16 + lane / 16 * 8 + x];
        score[j] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, score[j]);
      }
    }
    __syncthreads();
    if constexpr (Stage == 1)
      issue<BC, D, Threads>(v, key0, n, vr);
    publish<BC, D, Threads, true>(sv, vr);
    if constexpr (Stage == 2) {
      if (key0 + BC < n)
        issue<BC, D, Threads>(k, key0 + BC, n, kr);
    }
    // Each group of 16 lanes owns the same eight rows and distinct columns.
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
        int row = wave * 16 + lane / 16 * 8 + x;
        if (plane == 0)
          p[row * BC + j * 16 + lane % 16] = (_Float16)prob;
      }
#pragma unroll
      for (int delta = 8; delta > 0; delta /= 2)
        sum += __shfl_xor(sum, delta, 16);
      denominator[x] = fmaf(alpha[x], denominator[x], sum);
      maximum[x] = mx;
    }
    __syncthreads(); // Publish V and FP16 P before the PV matrix product.
#pragma unroll
    for (int d = 0; d < OD / 16; ++d) {
      F8 pv = {};
#pragma unroll
      for (int j = 0; j < BC / 16; ++j) {
        H8 a, b;
#pragma unroll
        for (int x = 0; x < 8; ++x) {
          a[x] = p[(wave * 16 + lane % 16) * BC + j * 16 + lane / 16 * 8 + x];
          b[x] = od0 + d * 16 + lane % 16 < D
                     ? sv[(od0 + d * 16 + lane % 16) * BC + j * 16 +
                          lane / 16 * 8 + x]
                     : (_Float16)0;
        }
        pv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, pv);
      }
#pragma unroll
      for (int x = 0; x < 8; ++x)
        output[d][x] = fmaf(alpha[x], output[d][x], pv[x]);
    }
    if constexpr (Stage == 2) {
      if (key0 + BC < n)
        publish<BC, D, Threads>(sk, kr);
    }
    __syncthreads();
  }
#pragma unroll
  for (int d = 0; d < OD / 16; ++d)
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      int row = q0 + wave * 16 + lane / 16 * 8 + x;
      if (row < n && od0 + d * 16 + lane % 16 < D)
        o[row * D + od0 + d * 16 + lane % 16] =
            (_Float16)(output[d][x] / denominator[x]);
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

template <int D, int Stage, bool ShareQ, bool RR = false>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
            torch::Tensor o) {
  using C = Config<D, Stage, ShareQ, RR>;
  static_assert(C::HalfElements * 2 <= 65536);
  dim3 grid((q.size(2) + C::BR - 1) / C::BR, q.size(0) * q.size(1));
  attention_shared<D, Stage, ShareQ, RR>
      <<<grid, C::Threads, C::HalfElements * 2, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const _Float16 *>(q.data_ptr()),
          reinterpret_cast<const _Float16 *>(k.data_ptr()),
          reinterpret_cast<const _Float16 *>(v.data_ptr()),
          reinterpret_cast<_Float16 *>(o.data_ptr()), q.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool ShareQ, bool RR = false>
void dispatch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
              torch::Tensor o, int stages) {
  check(q, k, v, o);
  int d = q.size(3);
  TORCH_CHECK(d == 32 || d == 64 || d == 96 || d == 128 ||
                  (d == 256 && (stages <= 1 || (RR && ShareQ))),
              "unsupported head dimension/stage combination");
  if (!q.numel())
    return;
  c10::cuda::CUDAGuard guard(q.device());
  // Match upstream: all stages <= 1 select 1; all stages > 1 select 2.
#define HEAD(D)                                                                \
  case D:                                                                      \
    if (stages > 1)                                                            \
      launch<D, 2, ShareQ, RR>(q, k, v, o);                                    \
    else                                                                       \
      launch<D, 1, ShareQ, RR>(q, k, v, o);                                    \
    break;
  switch (d) {
    HEAD(32)
    HEAD(64)
    HEAD(96)
    HEAD(128)
  case 256:
    if constexpr (RR && ShareQ) {
      if (stages > 1) {
        launch<256, 2, ShareQ, RR>(q, k, v, o);
        break;
      }
    }
    launch<256, 1, ShareQ, RR>(q, k, v, o);
    break;
  }
#undef HEAD
}
} // namespace

void flash_attn_mma_stages_split_q_shared_kv_acc_f32(torch::Tensor q,
                                                     torch::Tensor k,
                                                     torch::Tensor v,
                                                     torch::Tensor o,
                                                     int stages) {
  dispatch<false>(q, k, v, o, stages);
}
void flash_attn_mma_stages_split_q_shared_qkv_acc_f32(torch::Tensor q,
                                                      torch::Tensor k,
                                                      torch::Tensor v,
                                                      torch::Tensor o,
                                                      int stages) {
  dispatch<true>(q, k, v, o, stages);
}

void flash_attn_mma_stages_split_q_shared_kv_acc_f32_rr(torch::Tensor q,
                                                        torch::Tensor k,
                                                        torch::Tensor v,
                                                        torch::Tensor o,
                                                        int stages) {
  dispatch<false, true>(q, k, v, o, stages);
}

void flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr(torch::Tensor q,
                                                         torch::Tensor k,
                                                         torch::Tensor v,
                                                         torch::Tensor o,
                                                         int stages) {
  dispatch<true, true>(q, k, v, o, stages);
}
