#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
namespace leetcuda_cute_dsl {
// Original scalar DSL example rounds the FP32 accumulator to half before its
// FP32 output store. Keep this observable contract, with bounded tail stores.
__global__ void naive(const float *a, const float *b, float *c, int m, int n,
                      int k) {
  int row = blockIdx.y * 32 + threadIdx.y, col = blockIdx.x * 32 + threadIdx.x;
  if (row >= m || col >= n)
    return;
  float sum = 0;
  for (int x = 0; x < k; x++)
    sum = fmaf(a[size_t(row) * k + x], b[size_t(x) * n + col], sum);
  c[size_t(row) * n + col] = float(__float2half(sum));
}
__device__ __forceinline__ float issue(const float *p) {
  float x;
  asm volatile("global_load_b32 %0, %1, off" : "=v"(x) : "v"(p) : "memory");
  return x;
}
__device__ __forceinline__ float finish(float x) {
  asm volatile("s_wait_loadcnt 0" : "+v"(x)::"memory");
  return x;
}
struct Pending {
  float a[4], b[4];
};
__device__ __forceinline__ Pending load(const float *a, const float *b, int m,
                                        int n, int k, int tile) {
  Pending p;
  int row = threadIdx.x / 8, kk = threadIdx.x % 8 + tile * 8;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int ar = blockIdx.x * 128 + row + i * 32,
        br = blockIdx.y * 128 + row + i * 32;
    p.a[i] = ar < m && kk < k ? issue(a + size_t(ar) * k + kk) : 0;
    p.b[i] = br < n && kk < k ? issue(b + size_t(br) * k + kk) : 0;
  }
  return p;
}
template <int Pitch>
__device__ __forceinline__ void publish(float *sa, float *sb, Pending &p) {
  int row = threadIdx.x / 8, kk = threadIdx.x % 8;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    sa[kk * Pitch + row + i * 32] = finish(p.a[i]);
    sb[kk * Pitch + row + i * 32] = finish(p.b[i]);
  }
}
template <int Pitch>
__device__ __forceinline__ void fragments(const float *sa, const float *sb,
                                          int kk, float (&a)[8],
                                          float (&b)[8]) {
  int row = threadIdx.x / 16, col = threadIdx.x % 16;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    a[i] = sa[kk * Pitch + row + i * 16];
    b[i] = sb[kk * Pitch + col + i * 16];
  }
}
template <bool Pipeline>
__global__ __launch_bounds__(256) void tiled(const float *a, const float *bt,
                                             float *c, int m, int n, int k,
                                             unsigned *metrics) {
  constexpr int S = Pipeline ? 3 : 1, Pitch = Pipeline ? 128 : 132,
                Tile = Pitch * 8;
  __shared__ float sa[S * Tile], sb[S * Tile];
  float accum[8][8] = {};
  int tiles = (k + 7) / 8;
  unsigned *stats =
      metrics ? metrics + (size_t(blockIdx.x) * gridDim.y + blockIdx.y) * 8
              : nullptr;
  if constexpr (Pipeline) {
    for (int t = 0; t < 2 && t < tiles; t++) {
      auto p = load(a, bt, m, n, k, t);
      publish<Pitch>(sa + t * Tile, sb + t * Tile, p);
      if (stats && threadIdx.x == 0) {
        stats[0]++;
        stats[2] |= 1u << t;
      }
    }
    __syncthreads();
  }
  for (int t = 0; t < tiles; t++) {
    int slot = t % S;
    Pending pending;
    if constexpr (Pipeline) {
      pending = load(a, bt, m, n, k, t + 2);
      if (stats && threadIdx.x == 0 && t + 2 < tiles) {
        stats[4]++;
        stats[3] = 3;
      }
    } else {
      pending = load(a, bt, m, n, k, t);
      publish<Pitch>(sa, sb, pending);
      __syncthreads();
      if (stats && threadIdx.x == 0) {
        stats[0]++;
        stats[2] = 1;
        stats[3] = 1;
      }
    }
    float ar[2][8], br[2][8];
    fragments<Pitch>(sa + slot * Tile, sb + slot * Tile, 0, ar[0], br[0]);
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      if constexpr (Pipeline) {
        if (kk < 7)
          fragments<Pitch>(sa + slot * Tile, sb + slot * Tile, kk + 1,
                           ar[(kk + 1) % 2], br[(kk + 1) % 2]);
      } else if (kk)
        fragments<Pitch>(sa, sb, kk, ar[0], br[0]);
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
          accum[i][j] = fmaf(ar[Pipeline ? kk % 2 : 0][i],
                             br[Pipeline ? kk % 2 : 0][j], accum[i][j]);
      __builtin_amdgcn_sched_barrier(0);
    }
    if constexpr (Pipeline)
      if (t + 2 < tiles) {
        publish<Pitch>(sa + ((t + 2) % 3) * Tile, sb + ((t + 2) % 3) * Tile,
                       pending);
        if (stats && threadIdx.x == 0) {
          stats[0]++;
          stats[2] |= 1u << ((t + 2) % 3);
        }
      }
    __syncthreads();
    if (stats && threadIdx.x == 0) {
      stats[1]++;
      stats[5]++;
    }
  }
  int row = threadIdx.x / 16 + blockIdx.x * 128,
      col = threadIdx.x % 16 + blockIdx.y * 128;
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (row + i * 16 < m && col + j * 16 < n)
        c[size_t(row + i * 16) * n + col + j * 16] = accum[i][j];
}
} // namespace leetcuda_cute_dsl
