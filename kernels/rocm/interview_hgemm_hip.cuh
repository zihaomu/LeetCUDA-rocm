#pragma once
#include "interview_common_hip.cuh"
#include <stdexcept>

// Standalone AMD implementations of the five interview HGEMM schedules.
// Kind 0: plain TN multistage; 1: XOR LDS and register ping-pong;
// 2: CuTe-equivalent coordinate transform and LDS epilogue;
// 3: software wavegroup, one producer wave; 4: four producer/four consumer
// waves. NVIDIA TMA/WGMMA instructions are replaced by real loads and sequence
// counters.
namespace leetcuda_interview_hgemm {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using U4 = unsigned __attribute__((ext_vector_type(4)));
template <int Kind, int Stages> struct Config {
  static_assert(Kind >= 0 && Kind <= 5 && Stages >= 2 &&
                Stages <= ((Kind == 3 || Kind == 5) ? 5 : 4));
  static constexpr bool WS = Kind == 3 || Kind == 4;
  static constexpr int BM = WS || (Kind == 1 && Stages > 2) ? 64 : 128;
  // Match the original CuTe FP32 configuration's BN=128. The FP16 state
  // control uses the same smaller CTA to keep the diagnostic free of spills.
  static constexpr int BN = Kind == 1 && Stages > 2 ? 64 : 128;
  static constexpr int BK = Kind == 0 || Kind == 5                    ? 16
                            : Kind == 1 || (Kind == 4 && Stages == 2) ? 64
                                                                      : 32;
  static constexpr int Producers = Kind == 4 ? 4 : 1;
  static constexpr int Waves = WS ? 4 : 8;
  static constexpr int Threads = WS ? (Producers + Waves) * 32 : Waves * 32;
  static constexpr int WM = 2, WN = Waves / WM;
  static constexpr int RM = BM / (16 * WM), RN = BN / (16 * WN);
  static constexpr int Slot = (BM + BN) * BK;
  __host__ __device__ static int index(int row, int k) {
    int offset = row * BK + k;
    if constexpr (Kind == 2)
      return offset ^ ((offset >> 3) & 56);
    if constexpr (Kind == 1 || Kind == 5)
      return row * BK + (k ^ (((row >> 2) & 1) * 8));
    if constexpr (Kind != 0)
      return row * BK + (k ^ ((row & (BK / 8 - 1)) * 8));
    return offset;
  }
};
__device__ inline U4 issue(const _Float16 *p) {
  U4 value;
  asm volatile("global_load_b128 %0, %1, off"
               : "=v"(value)
               : "v"(p)
               : "memory");
  return value;
}
__device__ inline H8 finish(U4 &value) {
  asm volatile("s_wait_loadcnt 0" : "+v"(value)::"memory");
  return __builtin_bit_cast(H8, value);
}
template <class T, int Workers>
__device__ inline void
load_vectors(const _Float16 *a, const _Float16 *b, int m, int n, int k, int bm,
             int bn, int tile,
             U4 (&values)[(T::Slot + Workers * 8 - 1) / (Workers * 8)]) {
  constexpr int Count = (T::Slot + Workers * 8 - 1) / (Workers * 8);
#pragma unroll
  for (int i = 0; i < Count; i++) {
    int idx = (threadIdx.x + i * Workers) * 8;
    bool second = idx >= T::BM * T::BK;
    int local = second ? idx - T::BM * T::BK : idx;
    int row = (second ? bn : bm) + local / T::BK;
    int col = tile * T::BK + local % T::BK;
    const _Float16 *p = second ? b : a;
    int rows = second ? n : m;
    if (idx < T::Slot && row < rows && col + 8 <= k && k % 8 == 0 &&
        uintptr_t(p) % 16 == 0)
      values[i] = issue(p + row * k + col);
    else {
      H8 v = {};
      if (idx < T::Slot && row < rows)
        for (int x = 0; x < 8; x++)
          if (col + x < k)
            v[x] = p[row * k + col + x];
      values[i] = __builtin_bit_cast(U4, v);
    }
  }
}
template <class T, int Workers>
__device__ inline void
publish(_Float16 *lds,
        U4 (&values)[(T::Slot + Workers * 8 - 1) / (Workers * 8)]) {
#pragma unroll
  for (int i = 0; i < (T::Slot + Workers * 8 - 1) / (Workers * 8); i++) {
    int idx = (threadIdx.x + i * Workers) * 8;
    H8 v = finish(values[i]);
    if (idx < T::Slot) {
      bool second = idx >= T::BM * T::BK;
      int local = second ? idx - T::BM * T::BK : idx;
#pragma unroll
      for (int x = 0; x < 8; x++)
        lds[(second ? T::BM * T::BK : 0) +
            T::index(local / T::BK, local % T::BK + x)] = v[x];
    }
  }
}
template <class T>
__device__ inline void fragments(const _Float16 *lds, int wave, int kk,
                                 H8 (&af)[T::RM], H8 (&bf)[T::RN]) {
  int lane = threadIdx.x % 32, wm = wave % T::WM, wn = wave / T::WM;
#pragma unroll
  for (int i = 0; i < T::RM; i++)
#pragma unroll
    for (int x = 0; x < 8; x++)
      af[i][x] = lds[T::index(wm * T::RM * 16 + i * 16 + lane % 16,
                              kk * 16 + lane / 16 * 8 + x)];
#pragma unroll
  for (int j = 0; j < T::RN; j++)
#pragma unroll
    for (int x = 0; x < 8; x++)
      bf[j][x] =
          lds[T::BM * T::BK + T::index(wn * T::RN * 16 + j * 16 + lane % 16,
                                       kk * 16 + lane / 16 * 8 + x)];
}
template <class T, bool AccF32>
__device__ inline void multiply(H8 (&af)[T::RM], H8 (&bf)[T::RN],
                                F8 (&acc)[T::RM][T::RN]) {
#pragma unroll
  for (int i = 0; i < T::RM; i++)
#pragma unroll
    for (int j = 0; j < T::RN; j++) {
      acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[i], bf[j],
                                                                   acc[i][j]);
      if constexpr (!AccF32)
#pragma unroll
        for (int x = 0; x < 8; x++)
          acc[i][j][x] = float(_Float16(acc[i][j][x]));
    }
}
template <int Kind, class T, bool AccF32>
__device__ inline void consume(const _Float16 *lds, int wave,
                               F8 (&acc)[T::RM][T::RN]) {
  if constexpr (Kind == 1) {
    H8 af[2][T::RM], bf[2][T::RN];
    fragments<T>(lds, wave, 0, af[0], bf[0]);
#pragma unroll
    for (int kk = 0; kk < T::BK / 16; kk++) {
      if (kk + 1 < T::BK / 16)
        fragments<T>(lds, wave, kk + 1, af[(kk + 1) % 2], bf[(kk + 1) % 2]);
      multiply<T, AccF32>(af[kk % 2], bf[kk % 2], acc);
    }
  } else if constexpr (Kind == 2) {
#pragma unroll 1
    for (int kk = 0; kk < T::BK / 16; kk++) {
      H8 af[T::RM], bf[T::RN];
      fragments<T>(lds, wave, kk, af, bf);
      multiply<T, AccF32>(af, bf, acc);
    }
  } else {
#pragma unroll
    for (int kk = 0; kk < T::BK / 16; kk++) {
      H8 af[T::RM], bf[T::RN];
      fragments<T>(lds, wave, kk, af, bf);
      multiply<T, AccF32>(af, bf, acc);
    }
  }
}
__device__ inline void release(unsigned *p, unsigned value) {
  __threadfence_block();
  __syncwarp();
  if (threadIdx.x % 32 == 0)
    __hip_atomic_store(p, value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_WORKGROUP);
}
template <bool Audit>
__device__ inline bool acquire(unsigned *p, unsigned value, unsigned *stats) {
  unsigned spins = 0;
  while (__hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) <
         value) {
    __builtin_amdgcn_s_sleep(1);
    if constexpr (Audit)
      if (++spins == (1u << 20)) {
        if (threadIdx.x % 32 == 0)
          atomicOr(stats + 6, 1u);
        return false;
      }
  }
  return true;
}
template <int Kind, int Stages, bool AccF32, bool Audit = false,
          bool Delay = false>
__global__ void __launch_bounds__((Config<Kind, Stages>::Threads))
    kernel(const _Float16 *a, const _Float16 *b, _Float16 *c, int m, int n,
           int k, unsigned *metrics) {
  using T = Config<Kind, Stages>;
  __shared__ __align__(32) _Float16 lds[Stages * T::Slot];
  __shared__ unsigned ready[T::WS ? Stages : 1][T::WS ? T::Producers : 1];
  __shared__ unsigned done[T::WS ? Stages : 1][T::WS ? 4 : 1];
  int bm = blockIdx.y * T::BM,
      bn = (blockIdx.z * gridDim.x + blockIdx.x) * T::BN;
  if (bm >= m || bn >= n)
    return;
  int tid = threadIdx.x, lane = tid % 32;
  int wave = T::WS ? tid / 32 - T::Producers : tid / 32;
  int tiles = (k + T::BK - 1) / T::BK;
  F8 acc[T::RM][T::RN] = {};
  unsigned *stats = nullptr;
  if constexpr (Audit)
    stats =
        metrics +
        ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * 16;
  if constexpr (!T::WS) {
    U4 values[(T::Slot + T::Threads * 8 - 1) / (T::Threads * 8)];
    for (int t = 0; t < Stages - 1 && t < tiles; t++) {
      load_vectors<T, T::Threads>(a, b, m, n, k, bm, bn, t, values);
      publish<T, T::Threads>(lds + t * T::Slot, values);
    }
    __syncthreads();
    for (int t = 0; t < tiles; t++) {
      int future = t + Stages - 1;
      if (future < tiles)
        load_vectors<T, T::Threads>(a, b, m, n, k, bm, bn, future, values);
      consume<Kind, T, AccF32>(lds + t % Stages * T::Slot, wave, acc);
      if (future < tiles)
        publish<T, T::Threads>(lds + future % Stages * T::Slot, values);
      __syncthreads();
    }
  } else {
    if (tid < Stages * T::Producers)
      ready[tid / T::Producers][tid % T::Producers] = 0;
    if (tid < Stages * 4)
      done[tid / 4][tid % 4] = 0;
    __syncthreads();
    if (tid < T::Producers * 32) {
      U4 values[(T::Slot + T::Producers * 32 * 8 - 1) /
                (T::Producers * 32 * 8)];
      for (int t = 0; t < tiles; t++) {
        int slot = t % Stages;
        if (t >= Stages)
          for (int w = 0; w < 4; w++)
            if (!acquire<Audit>(&done[slot][w], t - Stages + 1, stats))
              return;
        load_vectors<T, T::Producers * 32>(a, b, m, n, k, bm, bn, t, values);
        publish<T, T::Producers * 32>(lds + slot * T::Slot, values);
        if constexpr (Audit)
          if (lane == 0) {
            stats[8 + tid / 32] = t + 1;
            if (tid == 0) {
              unsigned slowest = t;
              for (int w = 0; w < 4; w++) {
                unsigned latest = 0;
                for (int s = 0; s < Stages; s++)
                  latest = max(latest,
                               __hip_atomic_load(&done[s][w], __ATOMIC_ACQUIRE,
                                                 __HIP_MEMORY_SCOPE_WORKGROUP));
                slowest = min(slowest, latest);
              }
              stats[0] = t + 1;
              stats[5] = max(stats[5], unsigned(t + 1) - slowest);
            }
          }
        release(&ready[slot][tid / 32], t + 1);
      }
    } else {
      for (int t = 0; t < tiles; t++) {
        int slot = t % Stages;
        for (int p = 0; p < T::Producers; p++)
          if (!acquire<Audit>(&ready[slot][p], t + 1, stats))
            return;
        if constexpr (Delay)
          if (wave == 0)
            for (int d = 0; d < 2048; d++)
              __builtin_amdgcn_s_sleep(1);
        consume<Kind, T, AccF32>(lds + slot * T::Slot, wave, acc);
        if constexpr (Audit)
          if (lane == 0)
            stats[wave + 1] = t + 1;
        release(&done[slot][wave], t + 1);
      }
    }
  }
  if constexpr (Kind == 2) {
    // A/B storage is dead: reuse it as 16-row C stripes for coalesced stores.
    int wm = wave % T::WM, wn = wave / T::WM;
    for (int stripe = 0; stripe < T::BM / 16; stripe++) {
      __syncthreads();
#pragma unroll
      for (int i = 0; i < T::RM; i++)
        if (wm * T::RM + i == stripe)
#pragma unroll
          for (int j = 0; j < T::RN; j++)
#pragma unroll
            for (int x = 0; x < 8; x++)
              lds[(lane / 16 * 8 + x) * T::BN + wn * T::RN * 16 + j * 16 +
                  lane % 16] = _Float16(acc[i][j][x]);
      __syncthreads();
      for (int v = tid; v < 16 * T::BN; v += T::Threads) {
        int r = bm + stripe * 16 + v / T::BN, col = bn + v % T::BN;
        if (r < m && col < n)
          c[r * n + col] = lds[v];
      }
    }
  } else if (!T::WS || tid >= T::Producers * 32) {
    int wm = wave % T::WM, wn = wave / T::WM;
#pragma unroll
    for (int i = 0; i < T::RM; i++)
#pragma unroll
      for (int j = 0; j < T::RN; j++)
#pragma unroll
        for (int x = 0; x < 8; x++) {
          int r = bm + wm * T::RM * 16 + i * 16 + lane / 16 * 8 + x;
          int col = bn + wn * T::RN * 16 + j * 16 + lane % 16;
          if (r < m && col < n)
            c[r * n + col] = _Float16(acc[i][j][x]);
        }
  }
}
template <int Kind, int Stages, bool AccF32 = true, bool Audit = false,
          bool Delay = false>
inline hipError_t launch(const _Float16 *a, const _Float16 *b, _Float16 *c,
                         int m, int n, int k, bool block_swizzle,
                         hipStream_t stream, unsigned *metrics = nullptr,
                         int swizzle_stride = 2048) {
  using T = Config<Kind, Stages>;
  static_assert(Stages * T::Slot * 2 +
                    (T::WS ? Stages * (T::Producers + 4) * 4 : 0) <=
                65536);
  if (block_swizzle && swizzle_stride <= 0)
    return hipErrorInvalidValue;
  int z = block_swizzle ? 1 + (n - 1) / swizzle_stride : 1;
  dim3 grid(((n + T::BN - 1) / T::BN + z - 1) / z, (m + T::BM - 1) / T::BM, z);
  kernel<Kind, Stages, AccF32, Audit, Delay>
      <<<grid, T::Threads, 0, stream>>>(a, b, c, m, n, k, metrics);
  return hipGetLastError();
}
} // namespace leetcuda_interview_hgemm
