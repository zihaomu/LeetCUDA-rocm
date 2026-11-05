#pragma once
#include <hip/hip_runtime.h>
namespace leetcuda_swizzle_examples {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using U4 = unsigned __attribute__((ext_vector_type(4)));
__host__ __device__ inline int xor_column(int row, int col) {
  return (col & 7) | ((((col >> 3) ^ (row >> 2)) & 1) * 8);
}
__device__ __forceinline__ H8 load8(const _Float16 *p) {
  U4 u;
  asm volatile("global_load_b128 %0, %1, off" : "=v"(u) : "v"(p) : "memory");
  asm volatile("s_wait_loadcnt 0" : "+v"(u)::"memory");
  return __builtin_bit_cast(H8, u);
}
// Five original definitions: plain naive/tiled, XOR naive/tiled, and the
// coordinate-printing single-tile demo with its direct register epilogue.
// Explicit FP32 accumulation upgrade; FP16 input and output remain unchanged.
template <int Kind, int APad = 0, int BPad = 0>
__global__ __launch_bounds__(Kind == 1 || Kind == 3 ? 256 : 32) void gemm(
    const _Float16 *a, const _Float16 *b, _Float16 *c, int m, int n, int k) {
  constexpr bool Large = Kind == 1 || Kind == 3, Swizzle = Kind >= 2,
                 SmemOutput = Kind == 0 || Kind == 2;
  constexpr int BM = Large ? 128 : 16, BN = Large ? 128 : 8,
                Threads = Large ? 256 : 32, RM = Large ? 4 : 1,
                RN = Large ? 2 : 1;
  __shared__ _Float16 sa[BM][16 + APad], sb[16][BN + BPad],
      sc[SmemOutput ? BM * BN : 1];
  int tid = threadIdx.x, lane = tid % 32, wm = Large ? (tid / 32) % 2 : 0,
      wn = Large ? (tid / 32) / 2 : 0;
  int base_m = blockIdx.y * BM, base_n = blockIdx.x * BN;
  F8 acc[RM][RN] = {};
  for (int kt = 0; kt < k; kt += 16) {
    int row = tid / 2, col = (tid % 2) * 8;
    H8 av = {};
    if (base_m + row < m && kt + col + 7 < k && k % 8 == 0 &&
        uintptr_t(a) % 16 == 0)
      av = load8(a + size_t(base_m + row) * k + kt + col);
    else {
#pragma unroll
      for (int x = 0; x < 8; x++)
        if (base_m + row < m && kt + col + x < k)
          av[x] = a[size_t(base_m + row) * k + kt + col + x];
    }
    *reinterpret_cast<H8 *>(&sa[row][Swizzle ? xor_column(row, col) : col]) =
        av;
    if (tid < 16 * BN / 8) {
      int kr = tid / (BN / 8), nc = tid % (BN / 8) * 8;
      H8 bv = {};
      if (kt + kr < k && base_n + nc + 7 < n && n % 8 == 0 &&
          uintptr_t(b) % 16 == 0)
        bv = load8(b + size_t(kt + kr) * n + base_n + nc);
      else {
#pragma unroll
        for (int x = 0; x < 8; x++)
          if (kt + kr < k && base_n + nc + x < n)
            bv[x] = b[size_t(kt + kr) * n + base_n + nc + x];
      }
      *reinterpret_cast<H8 *>(&sb[kr][nc]) = bv;
    }
    __syncthreads();
    H8 af[RM], bf[RN];
#pragma unroll
    for (int i = 0; i < RM; i++) {
      int ar = wm * RM * 16 + i * 16 + lane % 16;
#pragma unroll
      for (int x = 0; x < 8; x++) {
        int kk = lane / 16 * 8 + x;
        af[i][x] = sa[ar][Swizzle ? xor_column(ar, kk) : kk];
      }
    }
#pragma unroll
    for (int j = 0; j < RN; j++) {
      int bc = wn * RN * 16 + j * 16 + lane % 16;
#pragma unroll
      for (int x = 0; x < 8; x++)
        bf[j][x] = bc < BN ? sb[lane / 16 * 8 + x][bc] : _Float16(0);
    }
#pragma unroll
    for (int i = 0; i < RM; i++)
#pragma unroll
      for (int j = 0; j < RN; j++)
        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            af[i], bf[j], acc[i][j]);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RM; i++)
#pragma unroll
    for (int j = 0; j < RN; j++)
#pragma unroll
      for (int x = 0; x < 8; x++) {
        int row = wm * RM * 16 + i * 16 + lane / 16 * 8 + x,
            col = wn * RN * 16 + j * 16 + lane % 16;
        if constexpr (SmemOutput) {
          if (col < BN)
            sc[row * BN + col] = _Float16(acc[i][j][x]);
        } else if (col < BN && base_m + row < m && base_n + col < n)
          c[size_t(base_m + row) * n + base_n + col] = _Float16(acc[i][j][x]);
      }
  if constexpr (SmemOutput) {
    __syncthreads();
    if (lane < BM) {
      int row = base_m + lane;
      if (row < m && base_n + 7 < n && n % 8 == 0 && uintptr_t(c) % 16 == 0)
        *reinterpret_cast<H8 *>(c + size_t(row) * n + base_n) =
            *reinterpret_cast<H8 *>(sc + lane * BN);
      else if (row < m) {
#pragma unroll
        for (int j = 0; j < BN; j++)
          if (base_n + j < n)
            c[size_t(row) * n + base_n + j] = sc[lane * BN + j];
      }
    }
  }
}
} // namespace leetcuda_swizzle_examples
