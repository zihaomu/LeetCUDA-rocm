#pragma once
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace leetcuda_ffpa {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using U4 = unsigned __attribute__((ext_vector_type(4)));
constexpr int BR = 16, BC = 64, Waves = 8, Consumers = 256, Capacity = 3;
__device__ __forceinline__ U4 issue8(const _Float16 *p) {
  U4 value;
  asm volatile("global_load_b128 %0, %1, off"
               : "=v"(value)
               : "v"(p)
               : "memory");
  return value;
}
__device__ __forceinline__ H8 finish8(U4 &value) {
  asm volatile("s_wait_loadcnt 0" : "+v"(value)::"memory");
  return __builtin_bit_cast(H8, value);
}
__device__ __forceinline__ int index(int row, int col, int width) {
  return row * width + (col ^ ((row & 3) * 8));
}
__device__ __forceinline__ void publish(unsigned *p, unsigned value) {
  __threadfence_block();
  __syncwarp();
  if (threadIdx.x % 32 == 0)
    __hip_atomic_store(p, value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_WORKGROUP);
}
__device__ __forceinline__ bool wait(unsigned *p, unsigned value,
                                     unsigned *stats) {
  unsigned spins = 0;
  while (__hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) <
         value) {
    __builtin_amdgcn_s_sleep(1);
    if (stats && ++spins == (1u << 21)) {
      if (threadIdx.x % 32 == 0)
        atomicOr(stats + 7, 1u);
      return false;
    }
  }
  return true;
}
template <int QC> struct Shared {
  alignas(16) _Float16 qk[Capacity * (BR + BC) * QC];
  alignas(16) _Float16 v[Capacity * BC * 64];
  alignas(16) _Float16 p[BR * BC];
  float scores[BR * BC], maximum[BR], sum[BR], alpha[BR];
  unsigned qready[Capacity], vready[Capacity];
  unsigned qdone[Capacity][Waves], vdone[Capacity][Waves], barrier[Waves];
};
template <int Count> struct Pending {
  U4 values[Count];
};

// Global loads remain pending in VGPR until the current WMMA tile completes.
template <int D, int Width, int Threads, bool IsV>
__device__ __forceinline__ auto
issue_tile(const _Float16 *q, const _Float16 *k, const _Float16 *v, int nq,
           int nk, int qbase, int kbase, int chunk, int tid) {
  constexpr int Elements = (IsV ? BC : (BR + BC)) * Width;
  constexpr int Packs = (Elements / 8 + Threads - 1) / Threads;
  Pending<Packs> pending;
#pragma unroll
  for (int i = 0; i < Packs; i++) {
    int pack = tid + i * Threads, flat = pack * 8;
    U4 raw = {};
    if (flat < Elements) {
      int row = flat / Width, col = chunk * Width + flat % Width;
      bool from_q = !IsV && row < BR;
      const _Float16 *input = IsV ? v : (from_q ? q : k);
      int global_row = from_q ? qbase + row : kbase + (IsV ? row : row - BR);
      int rows = from_q ? nq : nk;
      if (global_row < rows && col + 8 <= D && uintptr_t(input) % 16 == 0)
        raw = issue8(input + size_t(global_row) * D + col);
      else {
        H8 values = {};
        if (global_row < rows)
#pragma unroll
          for (int x = 0; x < 8; x++)
            if (col + x < D)
              values[x] = input[size_t(global_row) * D + col + x];
        raw = __builtin_bit_cast(U4, values);
      }
    }
    pending.values[i] = raw;
  }
  return pending;
}
template <int Width, int Threads, bool IsV, int Count>
__device__ __forceinline__ void commit_tile(Pending<Count> &pending,
                                            _Float16 *dst, int tid) {
  constexpr int Elements = (IsV ? BC : (BR + BC)) * Width;
#pragma unroll
  for (int i = 0; i < Count; i++) {
    int flat = (tid + i * Threads) * 8;
    H8 value = finish8(pending.values[i]);
    if (flat < Elements)
      *reinterpret_cast<H8 *>(dst + index(flat / Width, flat % Width, Width)) =
          value;
  }
}
template <int D, int QC>
__device__ __forceinline__ void consume_qk(const _Float16 *tile, F8 &score,
                                           int wave, int lane) {
  if (wave < BC / 16) {
#pragma unroll
    for (int kk = 0; kk < QC; kk += 16) {
      H8 q, k;
#pragma unroll
      for (int x = 0; x < 8; x++) {
        int col = kk + lane / 16 * 8 + x;
        q[x] = tile[index(lane % 16, col, QC)];
        k[x] = tile[index(BR + wave * 16 + lane % 16, col, QC)];
      }
      score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q, k, score);
    }
  }
}
__device__ __forceinline__ void consume_v(const _Float16 *p, const _Float16 *v,
                                          F8 &acc, int wave, int lane) {
#pragma unroll
  for (int kk = 0; kk < BC; kk += 16) {
    H8 a, b;
#pragma unroll
    for (int x = 0; x < 8; x++) {
      a[x] = p[(lane % 16) * BC + kk + lane / 16 * 8 + x];
      b[x] = v[index(kk + lane / 16 * 8 + x, (wave % 4) * 16 + lane % 16, 64)];
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
  }
}
template <int Fragments>
__device__ __forceinline__ void
output_chunk(const _Float16 *p, const _Float16 *v, F8 (&acc)[Fragments],
             int chunk, int wave, int lane) {
  // Keep the V loop rolled while each accumulator address is compile-time
  // fixed.
#define FFPA_ACC(I)                                                            \
  case I:                                                                      \
    if constexpr (Fragments > I) {                                             \
      if (wave / 4 == chunk % 2)                                               \
        consume_v(p, v, acc[I], wave, lane);                                   \
      asm volatile("" : "+v"(acc[I])::"memory");                               \
    }                                                                          \
    break
  switch (chunk / 2) {
    FFPA_ACC(0);
    FFPA_ACC(1);
    FFPA_ACC(2);
    FFPA_ACC(3);
    FFPA_ACC(4);
    FFPA_ACC(5);
    FFPA_ACC(6);
    FFPA_ACC(7);
  }
#undef FFPA_ACC
}
template <bool WS, int QC>
__device__ __forceinline__ bool
consumer_barrier(Shared<QC> &s, int wave, unsigned epoch, unsigned *stats) {
  if constexpr (WS) {
    publish(&s.barrier[wave], epoch);
    for (int w = 0; w < Waves; w++)
      if (!wait(&s.barrier[w], epoch, stats))
        return false;
  } else
    __syncthreads();
  return true;
}
template <int Kind, int D, int QC>
__device__ __forceinline__ void softmax(Shared<QC> &s, int tid,
                                        unsigned *stats) {
  int row = tid / 16, lane = tid % 16;
  float scale = rsqrtf(float(D)) * 1.4426950408889634f;
  float maximum = -INFINITY;
  for (int col = lane; col < BC; col += 16)
    maximum = fmaxf(maximum, s.scores[row * BC + col] * scale);
#pragma unroll
  for (int off = 8; off; off /= 2)
    maximum = fmaxf(maximum, __shfl_xor(maximum, off, 16));
  float old = s.maximum[row], next = fmaxf(old, maximum);
  if constexpr (Kind == 2) {
    // Preserve the original non-WS FA4 conditional rescale in log2 space.
    if (old != -INFINITY && next > old) {
      if (stats && row == 0 && lane == 0)
        atomicOr(stats + 6, next - old <= 8.0f ? 1u : 2u);
      if (next - old <= 8.0f)
        next = old;
    }
  }
  float alpha = exp2f(old - next), sum = 0;
  for (int col = lane; col < BC; col += 16) {
    float prob = exp2f(s.scores[row * BC + col] * scale - next);
    sum += prob;
    s.p[row * BC + col] = (_Float16)prob;
  }
#pragma unroll
  for (int off = 8; off; off /= 2)
    sum += __shfl_xor(sum, off, 16);
  if (lane == 0) {
    s.maximum[row] = next;
    s.sum[row] = s.sum[row] * alpha + sum;
    s.alpha[row] = alpha;
  }
}

// Kind 0: cooperative VGPR prefetch, QK64/PV64.
// Kind 1: independent producer wave + eight consumer waves, two LDS rings.
// Kind 2: QK32/PV64, conditional rescale and LDS output staging.
template <int D, int Kind>
__global__ __launch_bounds__(Kind == 1 ? 288 : 256) void kernel(
    const _Float16 *q, const _Float16 *k, const _Float16 *v, _Float16 *out,
    float *lse, int nq, int nk, int sq, int sv, unsigned *audit, int delay) {
  constexpr bool WS = Kind == 1;
  constexpr int QC = Kind == 2 ? 32 : 64, QSize = (BR + BC) * QC,
                VSize = BC * 64;
  constexpr int DQ = D / QC, DV = D / 64, OutFragments = (D + 127) / 128;
  __shared__ Shared<QC> s;
  int tid = threadIdx.x, lane = tid % 32;
  int consumer = tid - (WS ? 32 : 0), wave = consumer / 32;
  int qbase = blockIdx.x * BR, bh = blockIdx.y, blocks = (nk + BC - 1) / BC;
  q += size_t(bh) * nq * D;
  k += size_t(bh) * nk * D;
  v += size_t(bh) * nk * D;
  out += size_t(bh) * nq * D;
  unsigned *stats = audit ? audit + (bh * gridDim.x + blockIdx.x) * 8 : nullptr;
  if (tid < BR) {
    s.maximum[tid] = -INFINITY;
    s.sum[tid] = 0;
    s.alpha[tid] = 0;
  }
  if (tid < Capacity) {
    s.qready[tid] = 0;
    s.vready[tid] = 0;
  }
  if (tid < Capacity * Waves) {
    s.qdone[tid / Waves][tid % Waves] = 0;
    s.vdone[tid / Waves][tid % Waves] = 0;
  }
  if (tid < Waves)
    s.barrier[tid] = 0;
  __syncthreads();
  if constexpr (WS) {
    if (tid < 32) {
      for (int block = 0; block < blocks; block++) {
#pragma unroll 1
        for (int chunk = 0; chunk < DQ; chunk++) {
          unsigned seq = block * DQ + chunk;
          int slot = seq % sq;
          if (seq >= unsigned(sq))
            for (int w = 0; w < Waves; w++)
              if (!wait(&s.qdone[slot][w], seq - sq + 1, stats))
                return;
          if (stats && delay == 1)
            for (int d = 0; d < 32; d++)
              __builtin_amdgcn_s_sleep(1);
          auto pending = issue_tile<D, QC, 32, false>(q, k, v, nq, nk, qbase,
                                                      block * BC, chunk, lane);
          commit_tile<QC, 32, false>(pending, s.qk + slot * QSize, lane);
          publish(&s.qready[slot], seq + 1);
          if (stats && lane == 0) {
            stats[0] = seq + 1;
            atomicMax(stats + 4, seq + 1 - atomicAdd(stats + 1, 0u));
          }
        }
        for (int chunk = 0; chunk < DV; chunk++) {
          unsigned seq = block * DV + chunk;
          int slot = seq % sv;
          if (seq >= unsigned(sv))
            for (int w = 0; w < Waves; w++)
              if (!wait(&s.vdone[slot][w], seq - sv + 1, stats))
                return;
          auto pending = issue_tile<D, 64, 32, true>(q, k, v, nq, nk, qbase,
                                                     block * BC, chunk, lane);
          commit_tile<64, 32, true>(pending, s.v + slot * VSize, lane);
          publish(&s.vready[slot], seq + 1);
          if (stats && lane == 0) {
            stats[2] = seq + 1;
            atomicMax(stats + 5, seq + 1 - atomicAdd(stats + 3, 0u));
          }
        }
      }
      return;
    }
  }
  F8 acc[OutFragments] = {};
  for (int block = 0; block < blocks; block++) {
    F8 score = {};
    if constexpr (!WS) {
      for (int chunk = 0; chunk < sq && chunk < DQ; chunk++) {
        auto pending = issue_tile<D, QC, Consumers, false>(
            q, k, v, nq, nk, qbase, block * BC, chunk, consumer);
        commit_tile<QC, Consumers, false>(pending, s.qk + chunk * QSize,
                                          consumer);
      }
      __syncthreads();
#pragma unroll 1
      for (int chunk = 0; chunk < DQ; chunk++) {
        constexpr int Count = ((QSize / 8) + Consumers - 1) / Consumers;
        Pending<Count> pending;
        if (chunk + sq < DQ)
          pending = issue_tile<D, QC, Consumers, false>(
              q, k, v, nq, nk, qbase, block * BC, chunk + sq, consumer);
        consume_qk<D, QC>(s.qk + (chunk % sq) * QSize, score, wave, lane);
        asm volatile("" : "+v"(score)::"memory");
        __syncthreads();
        if (chunk + sq < DQ)
          commit_tile<QC, Consumers, false>(
              pending, s.qk + (chunk % sq) * QSize, consumer);
        __syncthreads();
      }
    } else {
#pragma unroll 1
      for (int chunk = 0; chunk < DQ; chunk++) {
        unsigned seq = block * DQ + chunk;
        int slot = seq % sq;
        if (!wait(&s.qready[slot], seq + 1, stats))
          return;
        if (stats && delay == 2 && wave == 0)
          for (int d = 0; d < 256; d++)
            __builtin_amdgcn_s_sleep(1);
        consume_qk<D, QC>(s.qk + slot * QSize, score, wave, lane);
        if (stats && consumer == 0)
          atomicExch(stats + 1, seq + 1);
        publish(&s.qdone[slot][wave], seq + 1);
      }
    }
    if (wave < BC / 16) {
#pragma unroll
      for (int x = 0; x < 8; x++) {
        int row = lane / 16 * 8 + x, col = wave * 16 + lane % 16;
        s.scores[row * BC + col] = block * BC + col < nk ? score[x] : -INFINITY;
      }
    }
    if (!consumer_barrier<WS>(s, wave, 2 * block + 1, stats))
      return;
    softmax<Kind, D>(s, consumer, stats);
    if (!consumer_barrier<WS>(s, wave, 2 * block + 2, stats))
      return;
#pragma unroll
    for (int i = 0; i < OutFragments; i++)
#pragma unroll
      for (int x = 0; x < 8; x++) {
        float alpha = s.alpha[lane / 16 * 8 + x];
        if (alpha != 1.0f)
          acc[i][x] *= alpha;
      }
    if constexpr (!WS) {
      for (int chunk = 0; chunk < sv && chunk < DV; chunk++) {
        auto pending = issue_tile<D, 64, Consumers, true>(
            q, k, v, nq, nk, qbase, block * BC, chunk, consumer);
        commit_tile<64, Consumers, true>(pending, s.v + chunk * VSize,
                                         consumer);
      }
      __syncthreads();
    }
#pragma unroll 1
    for (int chunk = 0; chunk < DV; chunk++) {
      if constexpr (WS) {
        unsigned seq = block * DV + chunk;
        int slot = seq % sv;
        if (!wait(&s.vready[slot], seq + 1, stats))
          return;
        if (stats && delay == 2 && wave == 0)
          for (int d = 0; d < 256; d++)
            __builtin_amdgcn_s_sleep(1);
        output_chunk(s.p, s.v + slot * VSize, acc, chunk, wave, lane);
        if (stats && consumer == 0)
          atomicExch(stats + 3, seq + 1);
        publish(&s.vdone[slot][wave], seq + 1);
      } else {
        constexpr int Count = ((VSize / 8) + Consumers - 1) / Consumers;
        Pending<Count> pending;
        if (chunk + sv < DV)
          pending = issue_tile<D, 64, Consumers, true>(
              q, k, v, nq, nk, qbase, block * BC, chunk + sv, consumer);
        output_chunk(s.p, s.v + (chunk % sv) * VSize, acc, chunk, wave, lane);
        __syncthreads();
        if (chunk + sv < DV)
          commit_tile<64, Consumers, true>(pending, s.v + (chunk % sv) * VSize,
                                           consumer);
        __syncthreads();
      }
    }
  }
#pragma unroll
  for (int chunk = 0; chunk < DV; chunk++) {
    if constexpr (Kind == 2)
      __syncthreads();
    if (wave / 4 == chunk % 2) {
#pragma unroll
      for (int x = 0; x < 8; x++) {
        int row = lane / 16 * 8 + x, col = (wave % 4) * 16 + lane % 16;
        float denom = s.sum[row];
        _Float16 value =
            (_Float16)(denom > 0 ? acc[chunk / 2][x] / denom : 0.0f);
        if constexpr (Kind == 2)
          s.qk[row * 64 + col] = value;
        else if (qbase + row < nq)
          out[size_t(qbase + row) * D + chunk * 64 + col] = value;
      }
    }
    if constexpr (Kind == 2) {
      __syncthreads();
      if (consumer < BR * 64 / 8) {
        int flat = consumer * 8, row = flat / 64, col = flat % 64;
        if (qbase + row < nq) {
          H8 value = *reinterpret_cast<H8 *>(s.qk + flat);
          auto *dst = out + size_t(qbase + row) * D + chunk * 64 + col;
          if (uintptr_t(dst) % 16 == 0)
            *reinterpret_cast<H8 *>(dst) = value;
          else
            for (int x = 0; x < 8; x++)
              dst[x] = value[x];
        }
      }
    }
  }
  if (lse && consumer < BR && qbase + consumer < nq)
    lse[size_t(bh) * nq + qbase + consumer] =
        s.sum[consumer] > 0 ? (s.maximum[consumer] + log2f(s.sum[consumer])) *
                                  0.6931471805599453f
                            : -INFINITY;
}
template <int D, int Kind>
inline hipError_t launch(const _Float16 *q, const _Float16 *k,
                         const _Float16 *v, _Float16 *out, float *lse,
                         int batch_heads, int nq, int nk, int sq, int sv,
                         unsigned *audit, int delay, hipStream_t stream) {
  hipLaunchKernelGGL((kernel<D, Kind>), dim3((nq + BR - 1) / BR, batch_heads),
                     dim3(Kind == 1 ? 288 : 256), 0, stream, q, k, v, out, lse,
                     nq, nk, sq, sv, audit, delay);
  return hipGetLastError();
}
template <int D>
inline hipError_t variant(const _Float16 *q, const _Float16 *k,
                          const _Float16 *v, _Float16 *out, float *lse, int bh,
                          int nq, int nk, int kind, int sq, int sv,
                          unsigned *audit, int delay, hipStream_t stream) {
#define FFPA_CALL(K)                                                           \
  launch<D, K>(q, k, v, out, lse, bh, nq, nk, sq, sv, audit, delay, stream)
  if (kind == 0)
    return FFPA_CALL(0);
  if (kind == 1)
    return FFPA_CALL(1);
  return FFPA_CALL(2);
#undef FFPA_CALL
}
// Caller supplies contiguous same-device FP16 tensors and disjoint output/LSE.
// Index products are bounded to signed 32-bit in the public binding.
inline hipError_t forward(const _Float16 *q, const _Float16 *k,
                          const _Float16 *v, _Float16 *out, float *lse, int bh,
                          int nq, int nk, int d, int kind, int sq, int sv,
                          hipStream_t stream, unsigned *audit = nullptr,
                          int delay = 0) {
  if (kind < 0 || kind > 2 || sq < 1 || sq > 3 || sv < 1 || sv > 3 || bh < 0 ||
      nq < 0 || nk < 0)
    return hipErrorInvalidValue;
  if (bh == 0 || nq == 0)
    return hipSuccess;
#define FFPA_D(D)                                                              \
  case D:                                                                      \
    return variant<D>(q, k, v, out, lse, bh, nq, nk, kind, sq, sv, audit,      \
                      delay, stream)
  switch (d) {
    FFPA_D(64);
    FFPA_D(128);
    FFPA_D(192);
    FFPA_D(256);
    FFPA_D(320);
    FFPA_D(384);
    FFPA_D(448);
    FFPA_D(512);
    FFPA_D(1024);
  default:
    return hipErrorInvalidValue;
  }
#undef FFPA_D
}
} // namespace leetcuda_ffpa
