#pragma once
#include "interview_common_hip.cuh"
#include <climits>

// Independent interview FA schedules. All qualified arithmetic uses FP32 QK,
// PV and online state, FP16 P and output. Kind 7 also retains upstream FP16
// Q prescaling. No quadratic global allocation or vendor-library fallback.
namespace leetcuda_interview_fa {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using U4 = unsigned __attribute__((ext_vector_type(4)));
template <int Kind, int D> struct Config {
  static_assert(Kind >= 0 && Kind <= 7 && Kind != 6);
  static constexpr bool Cute = Kind >= 3, Split = Kind == 2 || Kind == 5;
  static constexpr bool Persist = Kind == 7, WS = Kind != 0 && Kind != 3;
  static constexpr int BR = WS ? 32 : 64;
  static constexpr int BC = Split ? (D == 128 ? 16 : 32) : (D >= 96 ? 32 : 64);
  static constexpr int Groups = Split ? 2 : 1, Waves = BR / 16;
  static constexpr int Threads = (Waves * Groups + (WS ? 1 : 0)) * 32;
  static constexpr int KC = Persist ? 2 : 4,
                       VC = (Kind == 1 || Kind == 4 || Persist) ? 2 : 1;
  static constexpr int Pad = Kind == 0 ? 8 : 0;
  static constexpr int Q = BR * (D + Pad), KV = BC * (D + Pad);
  static constexpr int P = Cute ? 0 : Groups * BR * BC;
  static constexpr int Elements = Q + Groups * (KC + VC) * KV + P;
  template <int Rows>
  __host__ __device__ static int index(int row, int col, bool padded = false) {
    if constexpr (Kind == 0) {
      if (padded)
        return row * (D + 8) + col;
      return (col / 16) * Rows * 16 + row * 16 +
             ((col % 16) ^ (((row >> 2) & 1) * 8));
    } else {
      // Position-independent GMMA swizzle, including D=96's 64-byte atom.
      constexpr int width = D == 96 ? 32 : 64, mask = D == 96 ? 24 : 56;
      int flat = (col / width) * Rows * width + row * width + col % width;
      return flat ^ ((flat >> 3) & mask);
    }
  }
};
template <class C> struct Shared {
  alignas(1024) _Float16 data[C::Elements];
  unsigned qready, abort;
  unsigned kr[C::Groups][C::KC], vr[C::Groups][C::VC];
  unsigned kd[C::Groups][C::KC][C::Waves], vd[C::Groups][C::VC][C::Waves];
};
__device__ __forceinline__ U4 issue8(const _Float16 *p) {
  U4 r;
  asm volatile("global_load_b128 %0, %1, off" : "=v"(r) : "v"(p) : "memory");
  return r;
}
__device__ __forceinline__ H8 finish8(U4 &r) {
  asm volatile("s_wait_loadcnt 0" : "+v"(r)::"memory");
  return __builtin_bit_cast(H8, r);
}
template <int Count> struct Pending {
  U4 value[Count];
};
template <class C, int D, int Rows, int Workers>
__device__ __forceinline__ auto issue_tile(const _Float16 *src, int row0,
                                           int rows, int worker) {
  constexpr int Count = (Rows * D + Workers * 8 - 1) / (Workers * 8);
  Pending<Count> p;
#pragma unroll
  for (int i = 0; i < Count; i++) {
    int flat = (worker + i * Workers) * 8, row = row0 + flat / D,
        col = flat % D;
    U4 value = {};
    if (flat < Rows * D && row < rows) {
      if (uintptr_t(src) % 16 == 0)
        value = issue8(src + size_t(row) * D + col);
      else {
        H8 h;
        for (int x = 0; x < 8; x++)
          h[x] = src[size_t(row) * D + col + x];
        value = __builtin_bit_cast(U4, h);
      }
    }
    p.value[i] = value;
  }
  return p;
}
template <class C, int D, int Rows, int Workers, int Count>
__device__ __forceinline__ void publish_tile(_Float16 *dst, Pending<Count> &p,
                                             int worker, bool padded) {
#pragma unroll
  for (int i = 0; i < Count; i++) {
    int flat = (worker + i * Workers) * 8;
    H8 h = finish8(p.value[i]);
    if (flat < Rows * D) {
      // All supported layouts preserve aligned eight-half vectors.
      *reinterpret_cast<H8 *>(
          dst + C::template index<Rows>(flat / D, flat % D, padded)) = h;
    }
  }
}
__device__ __forceinline__ void release(unsigned *p, unsigned value) {
  __threadfence_block();
  __syncwarp();
  if (threadIdx.x % 32 == 0)
    __hip_atomic_store(p, value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_WORKGROUP);
}
__device__ __forceinline__ bool acquire(unsigned *p, unsigned value,
                                        unsigned *abort, unsigned *stats) {
  unsigned spins = 0;
  while (__hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) <
         value) {
    if (__hip_atomic_load(abort, __ATOMIC_RELAXED,
                          __HIP_MEMORY_SCOPE_WORKGROUP))
      return false;
    __builtin_amdgcn_s_sleep(1);
    if (stats && ++spins == (1u << 22)) {
      __hip_atomic_store(abort, 1u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_WORKGROUP);
      if (threadIdx.x % 32 == 0)
        atomicOr(stats + 7, 1u);
      return false;
    }
  }
  return true;
}
template <class C, int D>
__device__ __forceinline__ void q_fragment(const _Float16 *q, int wave, int kk,
                                           bool padded, H8 &a) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int x = 0; x < 8; x++)
    a[x] = q[C::template index<C::BR>(wave * 16 + lane % 16,
                                      kk * 16 + lane / 16 * 8 + x, padded)];
}
template <class C, int D>
__device__ __forceinline__ void k_fragments(const _Float16 *k, int kk,
                                            bool padded, H8 (&b)[C::BC / 16]) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int j = 0; j < C::BC / 16; j++)
#pragma unroll
    for (int x = 0; x < 8; x++)
      b[j][x] = k[C::template index<C::BC>(
          j * 16 + lane % 16, kk * 16 + lane / 16 * 8 + x, padded)];
}
template <int Kind, int D, class C>
__device__ __forceinline__ void
qk(const _Float16 *q, const _Float16 *k, int wave, int padmask,
   H8 (&cached_q)[D / 16], F8 (&score)[C::BC / 16]) {
  if constexpr (C::Persist) {
#pragma unroll
    for (int kk = 0; kk < D / 16; kk++) {
      H8 bf[C::BC / 16];
      k_fragments<C, D>(k, kk, false, bf);
#pragma unroll
      for (int j = 0; j < C::BC / 16; j++)
        score[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            cached_q[kk], bf[j], score[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
  } else if constexpr (C::Cute) {
    H8 a0, a1, b0[C::BC / 16], b1[C::BC / 16];
    q_fragment<C, D>(q, wave, 0, false, a0);
    k_fragments<C, D>(k, 0, false, b0);
#pragma unroll 1
    for (int kk = 0; kk < D / 16; kk += 2) {
      q_fragment<C, D>(q, wave, kk + 1, false, a1);
      k_fragments<C, D>(k, kk + 1, false, b1);
#pragma unroll
      for (int j = 0; j < C::BC / 16; j++)
        score[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0[j],
                                                                    score[j]);
      if (kk + 2 < D / 16) {
        q_fragment<C, D>(q, wave, kk + 2, false, a0);
        k_fragments<C, D>(k, kk + 2, false, b0);
      }
#pragma unroll
      for (int j = 0; j < C::BC / 16; j++)
        score[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1[j],
                                                                    score[j]);
    }
  } else {
#pragma unroll 1
    for (int kk = 0; kk < D / 16; kk++) {
      H8 a, b[C::BC / 16];
      q_fragment<C, D>(q, wave, kk, (padmask & 1) != 0, a);
      k_fragments<C, D>(k, kk, (padmask & 2) != 0, b);
#pragma unroll
      for (int j = 0; j < C::BC / 16; j++)
        score[j] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b[j], score[j]);
    }
  }
}
template <int Kind, int D, class C>
__device__ __forceinline__ void
probabilities(F8 (&score)[C::BC / 16], F8 &maximum, F8 &sum, F8 &alpha,
              int qbase, int wave, int keybase, int nq, int nk, bool causal,
              float scale) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int x = 0; x < 8; x++) {
    float mx = -INFINITY;
    int qr = qbase + wave * 16 + lane / 16 * 8 + x;
#pragma unroll
    for (int j = 0; j < C::BC / 16; j++) {
      int kr = keybase + j * 16 + lane % 16;
      bool valid = kr < nk && (!causal || kr <= qr + nk - nq);
      score[j][x] =
          valid ? (C::Persist ? score[j][x] : score[j][x] * scale) : -INFINITY;
      mx = fmaxf(mx, score[j][x]);
    }
#pragma unroll
    for (int delta = 8; delta > 0; delta /= 2)
      mx = fmaxf(mx, __shfl_xor(mx, delta, 16));
    float old = maximum[x];
    mx = fmaxf(mx, old);
    float eff = mx, next = mx, rescale = 1.0f;
    if constexpr (C::Persist) {
      float diff = old - mx;
      bool keep = diff >= -8.0f || mx == -INFINITY;
      eff = keep ? old : mx;
      next = eff;
      rescale = keep ? 1.0f : exp2f(diff);
    } else {
      rescale = mx == -INFINITY ? 1.0f : expf(old - mx);
    }
    maximum[x] = next;
    alpha[x] = rescale;
    float part = 0;
#pragma unroll
    for (int j = 0; j < C::BC / 16; j++) {
      float p = eff == -INFINITY ? 0.0f
                                 : (C::Persist ? exp2f(score[j][x] - eff)
                                               : expf(score[j][x] - eff));
      part += p;
      score[j][x] = p;
    }
#pragma unroll
    for (int delta = 8; delta > 0; delta /= 2)
      part += __shfl_xor(part, delta, 16);
    sum[x] = fmaf(alpha[x], sum[x], part);
  }
}
// Explicit WMMA accumulator -> A operand register permutation. Each output
// row spans 16 lanes; every selected source lane is in this same wave.
template <int Bit> __device__ __forceinline__ F8 transpose_step(F8 a) {
  F8 b;
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int r = 0; r < 8; r++) {
    float exchanged = __shfl_xor(a[r ^ Bit], Bit, 8);
    b[r] = ((lane & Bit) != (r & Bit)) ? exchanged : a[r];
  }
  return b;
}
__device__ __forceinline__ H8 p_register_fragment(F8 probability) {
  F8 t = transpose_step<4>(transpose_step<2>(transpose_step<1>(probability)));
  H8 result;
  int lane = threadIdx.x % 32;
  int src = ((lane & 8) << 1) | ((lane & 16) >> 1) | (lane & 7);
#pragma unroll
  for (int x = 0; x < 8; x++)
    result[x] = _Float16(__shfl(t[x], src, 32));
  return result;
}
template <int Kind, int D, class C>
__device__ __forceinline__ void pv(const _Float16 *v, _Float16 *p, int wave,
                                   int padmask, F8 (&score)[C::BC / 16],
                                   F8 alpha, F8 (&output)[D / 16]) {
  int lane = threadIdx.x % 32;
  H8 pf[C::BC / 16];
  if constexpr (C::Cute) {
#pragma unroll
    for (int j = 0; j < C::BC / 16; j++)
      pf[j] = p_register_fragment(score[j]);
  } else {
#pragma unroll
    for (int j = 0; j < C::BC / 16; j++)
#pragma unroll
      for (int x = 0; x < 8; x++)
        p[(wave * 16 + lane / 16 * 8 + x) * C::BC + j * 16 + lane % 16] =
            _Float16(score[j][x]);
    __syncwarp();
#pragma unroll
    for (int j = 0; j < C::BC / 16; j++)
#pragma unroll
      for (int x = 0; x < 8; x++)
        pf[j][x] =
            p[(wave * 16 + lane % 16) * C::BC + j * 16 + lane / 16 * 8 + x];
  }
#pragma unroll
  for (int d = 0; d < D / 16; d++) {
    F8 cur = {};
#pragma unroll
    for (int j = 0; j < C::BC / 16; j++) {
      H8 vf;
#pragma unroll
      for (int x = 0; x < 8; x++)
        vf[x] =
            v[C::template index<C::BC>(j * 16 + lane / 16 * 8 + x,
                                       d * 16 + lane % 16, (padmask & 4) != 0)];
      cur = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pf[j], vf, cur);
    }
#pragma unroll
    for (int x = 0; x < 8; x++)
      output[d][x] = fmaf(alpha[x], output[d][x], cur[x]);
    __builtin_amdgcn_sched_barrier(0);
  }
}

template <int Kind, int D, class C>
__device__ __forceinline__ void
cooperative(Shared<C> &s, const _Float16 *q, const _Float16 *k,
            const _Float16 *v, int qbase, int nq, int nk, int sk, int padmask,
            float scale, int wave, F8 &maximum, F8 &sum, F8 (&output)[D / 16],
            unsigned *stats) {
  int tid = threadIdx.x, tiles = (nk + C::BC - 1) / C::BC;
  _Float16 *sq = s.data;
  _Float16 *keys = sq + C::Q;
  _Float16 *values = keys + C::KC * C::KV;
  _Float16 *p = values + C::VC * C::KV;
  auto qr = issue_tile<C, D, C::BR, C::Threads>(q, qbase, nq, tid);
  publish_tile<C, D, C::BR, C::Threads>(sq, qr, tid, (padmask & 1) != 0);
  for (int t = 0; t < sk - 1 && t < tiles; t++) {
    auto kr = issue_tile<C, D, C::BC, C::Threads>(k, t * C::BC, nk, tid);
    publish_tile<C, D, C::BC, C::Threads>(keys + t * C::KV, kr, tid,
                                          (padmask & 2) != 0);
  }
  __syncthreads();
  H8 cached[D / 16]; // dead in non-persistent instantiations
  for (int t = 0; t < tiles; t++) {
    int future = t + sk - 1;
    auto kr = issue_tile<C, D, C::BC, C::Threads>(k, future * C::BC, nk, tid);
    if (sk == 1) {
      publish_tile<C, D, C::BC, C::Threads>(keys, kr, tid, (padmask & 2) != 0);
      __syncthreads();
    }
    auto vr = issue_tile<C, D, C::BC, C::Threads>(v, t * C::BC, nk, tid);
    F8 score[C::BC / 16] = {};
    qk<Kind, D, C>(sq, keys + (t % sk) * C::KV, wave, padmask, cached, score);
    if (sk > 1 && future < tiles)
      publish_tile<C, D, C::BC, C::Threads>(keys + (future % sk) * C::KV, kr,
                                            tid, (padmask & 2) != 0);
    publish_tile<C, D, C::BC, C::Threads>(values, vr, tid, (padmask & 4) != 0);
    __syncthreads();
    F8 alpha;
    probabilities<Kind, D, C>(score, maximum, sum, alpha, qbase, wave,
                              t * C::BC, nq, nk, false, scale);
    pv<Kind, D, C>(values, p, wave, padmask, score, alpha, output);
    __syncthreads();
  }
  if (stats && tid == 0) {
    stats[1] += tiles;
    stats[2] += tiles;
    stats[3] += tiles;
    stats[4] += tiles;
    stats[13]++;
  }
}
template <int Capacity, int Waves>
__device__ __forceinline__ void depth(unsigned (&done)[Capacity][Waves],
                                      int active, int t, unsigned *stats,
                                      int index) {
  if (!stats || threadIdx.x % 32)
    return;
  unsigned slowest = t;
  for (int w = 0; w < Waves; w++) {
    unsigned latest = 0;
    for (int slot = 0; slot < active; slot++)
      latest = max(latest, __hip_atomic_load(&done[slot][w], __ATOMIC_ACQUIRE,
                                             __HIP_MEMORY_SCOPE_WORKGROUP));
    slowest = min(slowest, latest);
  }
  atomicMax(stats + index, unsigned(t + 1) - slowest);
}
template <int Kind, int D, class C, bool IsV>
__device__ __forceinline__ bool produce(Shared<C> &s, const _Float16 *source,
                                        int nk, int cid, int local, int active,
                                        unsigned *stats) {
  int lane = threadIdx.x % 32, slot = local % active,
      tile = cid + local * C::Groups;
  unsigned *ready;
  _Float16 *target;
  if constexpr (IsV) {
    for (int w = 0; w < C::Waves; w++)
      if (local >= active &&
          !acquire(&s.vd[cid][slot][w], local - active + 1, &s.abort, stats))
        return false;
    ready = &s.vr[cid][slot];
    target = s.data + C::Q + C::Groups * C::KC * C::KV +
             (cid * C::VC + slot) * C::KV;
  } else {
    for (int w = 0; w < C::Waves; w++)
      if (local >= active &&
          !acquire(&s.kd[cid][slot][w], local - active + 1, &s.abort, stats))
        return false;
    ready = &s.kr[cid][slot];
    target = s.data + C::Q + (cid * C::KC + slot) * C::KV;
  }
  auto pending = issue_tile<C, D, C::BC, 32>(source, tile * C::BC, nk, lane);
  publish_tile<C, D, C::BC, 32>(target, pending, lane, false);
  if constexpr (IsV)
    depth(s.vd[cid], active, local, stats, 9 + 2 * cid);
  else
    depth(s.kd[cid], active, local, stats, 8 + 2 * cid);
  if (stats && lane == 0)
    atomicAdd(stats + (IsV ? 2 : 1), 1u);
  release(ready, local + 1);
  return true;
}
template <int Kind, int D, class C>
__device__ __forceinline__ void producer(Shared<C> &s, const _Float16 *q,
                                         const _Float16 *k, const _Float16 *v,
                                         int qbase, int nq, int nk, int sk,
                                         int sv, bool causal, unsigned *stats) {
  int lane = threadIdx.x % 32, tiles = (nk + C::BC - 1) / C::BC;
  if constexpr (C::Persist)
    if (causal) {
      int last = qbase + C::BR - 1 + nk - nq;
      tiles = last < 0 ? 0 : min(tiles, last / C::BC + 1);
    }
  {
    auto pending = issue_tile<C, D, C::BR, 32>(q, qbase, nq, lane);
    publish_tile<C, D, C::BR, 32>(s.data, pending, lane, false);
    if (stats && lane == 0)
      atomicAdd(stats + 13, 1u);
    release(&s.qready, 1);
  }
  for (int c = 0; c < C::Groups; c++)
    for (int t = 0; t < sk && c + t * C::Groups < tiles; t++)
      if (!produce<Kind, D, C, false>(s, k, nk, c, t, sk, stats))
        return;
  for (int c = 0; c < C::Groups; c++)
    for (int t = 0; t < sv && c + t * C::Groups < tiles; t++)
      if (!produce<Kind, D, C, true>(s, v, nk, c, t, sv, stats))
        return;
  for (int tile = 0; tile < tiles; tile++) {
    int c = tile % C::Groups, local = tile / C::Groups;
    if (c + (local + sk) * C::Groups < tiles &&
        !produce<Kind, D, C, false>(s, k, nk, c, local + sk, sk, stats))
      return;
    if (c + (local + sv) * C::Groups < tiles &&
        !produce<Kind, D, C, true>(s, v, nk, c, local + sv, sv, stats))
      return;
  }
}
template <int Kind, int D, class C>
__device__ __forceinline__ void
consumer(Shared<C> &s, int cid, int wave, int qbase, int nq, int nk, int sk,
         int sv, bool causal, float scale, int delay, F8 &maximum, F8 &sum,
         F8 (&output)[D / 16], unsigned *stats) {
  int lane = threadIdx.x % 32, tiles = (nk + C::BC - 1) / C::BC;
  if constexpr (C::Persist)
    if (causal) {
      int last = qbase + C::BR - 1 + nk - nq;
      tiles = last < 0 ? 0 : min(tiles, last / C::BC + 1);
    }
  if (!acquire(&s.qready, 1, &s.abort, stats))
    return;
  H8 cached_q[D / 16];
  if constexpr (C::Persist) {
#pragma unroll
    for (int kk = 0; kk < D / 16; kk++) {
      q_fragment<C, D>(s.data, wave, kk, false, cached_q[kk]);
#pragma unroll
      for (int x = 0; x < 8; x++)
        cached_q[kk][x] =
            _Float16(float(cached_q[kk][x]) * (scale * 1.4426950408889634f));
    }
  }
  _Float16 *keys = s.data + C::Q + cid * C::KC * C::KV;
  _Float16 *values =
      s.data + C::Q + C::Groups * C::KC * C::KV + cid * C::VC * C::KV;
  _Float16 *p = nullptr;
  if constexpr (!C::Cute)
    p = s.data + C::Q + C::Groups * (C::KC + C::VC) * C::KV +
        cid * C::BR * C::BC;
  for (int tile = cid, local = 0; tile < tiles; tile += C::Groups, local++) {
    if (stats && delay == 1 && wave == 0)
      for (int i = 0; i < 2048; i++)
        __builtin_amdgcn_s_sleep(1);
    if (!acquire(&s.kr[cid][local % sk], local + 1, &s.abort, stats))
      return;
    F8 score[C::BC / 16] = {};
    qk<Kind, D, C>(s.data, keys + (local % sk) * C::KV, wave, 0, cached_q,
                   score);
    release(&s.kd[cid][local % sk][wave], local + 1);
    if (stats && wave == 0 && lane == 0)
      atomicAdd(stats + 3 + cid * 2, 1u);
    F8 alpha;
    probabilities<Kind, D, C>(score, maximum, sum, alpha, qbase, wave,
                              tile * C::BC, nq, nk, causal, scale);
    if (!acquire(&s.vr[cid][local % sv], local + 1, &s.abort, stats))
      return;
    if (stats && delay == 2 && wave == 0)
      for (int i = 0; i < 2048; i++)
        __builtin_amdgcn_s_sleep(1);
    pv<Kind, D, C>(values + (local % sv) * C::KV, p, wave, 0, score, alpha,
                   output);
    release(&s.vd[cid][local % sv][wave], local + 1);
    if (stats && wave == 0 && lane == 0)
      atomicAdd(stats + 4 + cid * 2, 1u);
  }
}
template <int Kind, int D, bool Audit>
__global__ void __launch_bounds__((Config<Kind, D>::Threads))
    kernel(const _Float16 *q, const _Float16 *k, const _Float16 *v,
           _Float16 *out, float *lse, int batch, int hq, int hk, int nq, int nk,
           int sk, int sv, int padmask, bool causal, float scale,
           unsigned *metrics, int delay) {
  using C = Config<Kind, D>;
  __shared__ Shared<C> s;
  static_assert(sizeof(Shared<C>) <= 65536);
  int tid = threadIdx.x, lane = tid % 32,
      role = C::WS ? tid / 32 - 1 : tid / 32;
  bool is_consumer = !C::WS || tid >= 32;
  int cid = is_consumer ? role / C::Waves : 0,
      wave = is_consumer ? role % C::Waves : 0;
  int qtiles = (nq + C::BR - 1) / C::BR, total = batch * hq * qtiles;
  unsigned *stats = Audit ? metrics + blockIdx.x * 16 : nullptr;
  for (int work = blockIdx.x; work < total; work += gridDim.x) {
    if constexpr (!C::Persist)
      if (work != int(blockIdx.x))
        break;
    // Reset only after every role finished the previous persistent query tile.
    if (tid == 0) {
      s.abort = 0;
      s.qready = 0;
    }
    for (int i = tid; i < C::Groups * C::KC; i += C::Threads)
      reinterpret_cast<unsigned *>(s.kr)[i] = 0;
    for (int i = tid; i < C::Groups * C::VC; i += C::Threads)
      reinterpret_cast<unsigned *>(s.vr)[i] = 0;
    for (int i = tid; i < C::Groups * C::KC * C::Waves; i += C::Threads)
      reinterpret_cast<unsigned *>(s.kd)[i] = 0;
    for (int i = tid; i < C::Groups * C::VC * C::Waves; i += C::Threads)
      reinterpret_cast<unsigned *>(s.vd)[i] = 0;
    __syncthreads();
    int bh = work / qtiles, qbase = (work % qtiles) * C::BR;
    int kv = (bh / hq) * hk + (bh % hq) / (hq / hk);
    const _Float16 *qq = q + size_t(bh) * nq * D;
    const _Float16 *kk = k + size_t(kv) * nk * D;
    const _Float16 *vv = v + size_t(kv) * nk * D;
    F8 maximum, sum = {}, output[D / 16] = {};
#pragma unroll
    for (int x = 0; x < 8; x++)
      maximum[x] = -INFINITY;
    if constexpr (C::WS) {
      if (!is_consumer)
        producer<Kind, D, C>(s, qq, kk, vv, qbase, nq, nk, sk, sv, causal,
                             stats);
      else
        consumer<Kind, D, C>(s, cid, wave, qbase, nq, nk, sk, sv, causal, scale,
                             delay, maximum, sum, output, stats);
    } else
      cooperative<Kind, D, C>(s, qq, kk, vv, qbase, nq, nk, sk, padmask, scale,
                              wave, maximum, sum, output, stats);
    __syncthreads();
    if (s.abort)
      return;
    if constexpr (C::Split) {
      // All Q/K/V lifetimes ended. Reuse their LDS for WG1's partial state.
      float *partial = reinterpret_cast<float *>(s.data);
      if (is_consumer && cid == 1) {
#pragma unroll
        for (int x = 0; x < 8; x++) {
          int row = wave * 16 + lane / 16 * 8 + x;
          if (lane % 16 == 0) {
            partial[C::BR * D + row] = maximum[x];
            partial[C::BR * D + C::BR + row] = sum[x];
          }
#pragma unroll
          for (int d = 0; d < D / 16; d++)
            partial[row * D + d * 16 + lane % 16] = output[d][x];
        }
      }
      __syncthreads();
      if (is_consumer && cid == 0) {
#pragma unroll
        for (int x = 0; x < 8; x++) {
          int row = wave * 16 + lane / 16 * 8 + x;
          float m1 = partial[C::BR * D + row],
                l1 = partial[C::BR * D + C::BR + row];
          float mx = fmaxf(maximum[x], m1),
                a = maximum[x] == -INFINITY ? 0.0f : expf(maximum[x] - mx),
                b = m1 == -INFINITY ? 0.0f : expf(m1 - mx);
          sum[x] = fmaf(a, sum[x], b * l1);
          maximum[x] = mx;
#pragma unroll
          for (int d = 0; d < D / 16; d++)
            output[d][x] = fmaf(a, output[d][x],
                                b * partial[row * D + d * 16 + lane % 16]);
        }
      }
      __syncthreads();
    }
    if (is_consumer && cid == 0) {
#pragma unroll
      for (int x = 0; x < 8; x++) {
        int row = wave * 16 + lane / 16 * 8 + x, global = qbase + row;
        if (lse && lane % 16 == 0 && global < nq)
          lse[size_t(bh) * nq + global] =
              sum[x] > 0 ? (C::Persist ? maximum[x] * 0.6931471805599453f
                                       : maximum[x]) +
                               logf(sum[x])
                         : -INFINITY;
#pragma unroll
        for (int d = 0; d < D / 16; d++) {
          _Float16 value =
              sum[x] > 0 ? _Float16(output[d][x] / sum[x]) : _Float16(0);
          if constexpr (C::Cute)
            s.data[C::template index<C::BR>(row, d * 16 + lane % 16)] = value;
          else if (global < nq)
            out[(size_t(bh) * nq + global) * D + d * 16 + lane % 16] = value;
        }
      }
    }
    __syncthreads();
    if constexpr (C::Cute)
      for (int flat = tid; flat < C::BR * D; flat += C::Threads) {
        int row = flat / D, col = flat % D;
        if (qbase + row < nq)
          out[(size_t(bh) * nq + qbase + row) * D + col] =
              s.data[C::template index<C::BR>(row, col)];
      }
    __syncthreads();
    if (stats && tid == 0) {
      stats[0]++;
      stats[12]++;
    }
  }
}
template <int D>
__global__ void copy_kernel(const _Float16 *q, _Float16 *out, int rows) {
  using C = Config<3, D>;
  __shared__ __align__(1024) _Float16 data[64 * D];
  auto pending =
      issue_tile<C, D, 64, 128>(q, blockIdx.x * 64, rows, threadIdx.x);
  publish_tile<C, D, 64, 128>(data, pending, threadIdx.x, false);
  __syncthreads();
  for (int i = threadIdx.x; i < 64 * D; i += 128)
    if (blockIdx.x * 64 + i / D < rows)
      out[size_t(blockIdx.x) * 64 * D + i] =
          data[C::template index<64>(i / D, i % D)];
}
template <int Kind, int D>
inline hipError_t
launch(const _Float16 *q, const _Float16 *k, const _Float16 *v, _Float16 *o,
       float *lse, int batch, int hq, int hk, int nq, int nk, int sk, int sv,
       int padmask, bool causal, float scale, hipStream_t stream,
       unsigned *metrics = nullptr, int delay = 0, int persistent_ctas = 0) {
  using C = Config<Kind, D>;
  if (batch <= 0 || hq <= 0 || hk <= 0 || hq % hk || nq <= 0 || nk <= 0 ||
      sk < 1 || sk > C::KC || sv < 1 || sv > C::VC || padmask < 0 ||
      padmask > 7 || (Kind != 0 && padmask) ||
      (!C::Persist && (causal || nq != nk || hq != hk)))
    return hipErrorInvalidValue;
  int total = batch * hq * ((nq + C::BR - 1) / C::BR), grid = total;
  if constexpr (C::Persist) {
    if (!causal || persistent_ctas > 0) {
      if (persistent_ctas <= 0) {
        int device;
        hipError_t e = hipGetDevice(&device);
        if (e != hipSuccess)
          return e;
        e = hipDeviceGetAttribute(
            &persistent_ctas, hipDeviceAttributeMultiprocessorCount, device);
        if (e != hipSuccess)
          return e;
      }
      grid = min(total, persistent_ctas);
    }
  }
  if (metrics)
    kernel<Kind, D, true><<<grid, C::Threads, 0, stream>>>(
        q, k, v, o, lse, batch, hq, hk, nq, nk, sk, sv, padmask, causal, scale,
        metrics, delay);
  else
    kernel<Kind, D, false><<<grid, C::Threads, 0, stream>>>(
        q, k, v, o, lse, batch, hq, hk, nq, nk, sk, sv, padmask, causal, scale,
        nullptr, 0);
  return hipGetLastError();
}
} // namespace leetcuda_interview_fa
