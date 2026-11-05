#pragma once
#include "interview_common_hip.cuh"
#include <initializer_list>

// Standalone interview TF32: preserve the original 128x128x8 CTA and
// 4x2 waves, each holding 2x4 accumulator fragments. gfx1201 has no TF32
// WMMA. Two exact BF16 components per TF32 operand require four BF16 WMMAs.
// This is the original single-TF32 arithmetic, not the B6 TF32x3 upgrade.
namespace interview_tf32 {
using B8 = short __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using F4 = float __attribute__((ext_vector_type(4)));
using U4 = unsigned __attribute__((ext_vector_type(4)));
__device__ inline float round_tf32(float x) {
  unsigned bits = __float_as_uint(x);
  if ((bits & 0x7fffffff) >= 0x7f800000)
    return x;
  return __uint_as_float((bits + 0x1000u) & 0xffffe000u);
}
__device__ inline void parts(float x, short &hi, short &lo) {
  unsigned bits = __float_as_uint(x);
  hi = short(bits >> 16);
  lo = short(__float_as_uint(x - __uint_as_float(bits & 0xffff0000u)) >> 16);
}
__device__ inline U4 issue(const float *p) {
  U4 value;
  asm volatile("global_load_b128 %0, %1, off"
               : "=v"(value)
               : "v"(p)
               : "memory");
  return value;
}
__device__ inline F4 finish(U4 &value) {
  asm volatile("s_wait_loadcnt 0" : "+v"(value)::"memory");
  return __builtin_bit_cast(F4, value);
}
template <int PA, int PB>
__device__ inline void publish(float *a, float *b, unsigned *flags, U4 &av,
                               U4 &bv) {
  int t = threadIdx.x, ar = t / 2, ac = (t % 2) * 4;
  int br = t / 32, bc = (t % 32) * 4;
  F4 va = finish(av), vb = finish(bv);
  bool unsafe = false;
  for (int j = 0; j < 4; j++) {
    a[ar * (8 + PA) + ac + j] = va[j];
    b[br * (128 + PB) + bc + j] = vb[j];
    for (float x : {va[j], vb[j]}) {
      float magnitude = fabsf(x);
      unsafe |=
          magnitude != 0 && !(magnitude >= 0x1p-96f && magnitude <= 0x1p112f);
    }
  }
  unsigned any = __any_sync(0xffffffffULL, unsafe);
  if (t % 32 == 0)
    flags[t / 32] = any;
}
} // namespace interview_tf32

__global__ void f32x4_tf32x4_kernel(float *x, float *y, int N) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i < N) {
    float4 a = FLOAT4(x[i]);
    FLOAT4(y[i]) =
        make_float4(interview_tf32::round_tf32(a.x), interview_tf32::round_tf32(a.y),
                    interview_tf32::round_tf32(a.z), interview_tf32::round_tf32(a.w));
  }
}

template <const int kWMmaM = 16, const int kWMmaN = 16, const int kWMmaK = 8,
          const int kWMmaTileM = 4, const int kWMmaTileN = 2,
          const int kValTileM = 2, const int kValTileN = 4, const int kPadA = 4,
          const int kPadB = 4, const int kStages = 2>
__global__ void __launch_bounds__(256)
    sgemm_tf32(float *A, float *B, float *C, int M, int N, int K) {
  using namespace interview_tf32;
  static_assert(kWMmaM == 16 && kWMmaN == 16 && kWMmaK == 8 &&
                kWMmaTileM == 4 && kWMmaTileN == 2 && kValTileM == 2 &&
                kValTileN == 4);
  static_assert(kStages >= 2 && kStages <= 4 && kPadA >= 0 && kPadB >= 0);
  constexpr int AS = 128 * (8 + kPadA), BS = 8 * (128 + kPadB);
  extern __shared__ float smem_tf32[];
  float *sa = smem_tf32, *sb = sa + kStages * AS;
  __shared__ unsigned flags[kStages][8];
  int t = threadIdx.x, lane = t % 32, wave = t / 32;
  int wm = wave / 2, wn = wave % 2;
  int bm = blockIdx.y * 128, bn = blockIdx.x * 128, tiles = K / 8;
  F8 accum[2][4] = {};
  U4 av, bv;
  auto load = [&](int tile) {
    av = issue(A + (bm + t / 2) * K + tile * 8 + (t % 2) * 4);
    bv = issue(B + (tile * 8 + t / 32) * N + bn + (t % 32) * 4);
  };
  for (int i = 0; i < kStages - 1 && i < tiles; i++) {
    load(i);
    publish<kPadA, kPadB>(sa + i * AS, sb + i * BS, flags[i], av, bv);
  }
  __syncthreads();
  for (int tile = 0; tile < tiles; tile++) {
    int next = tile + kStages - 1, slot = tile % kStages;
    // Outstanding global loads overlap computation on the current LDS slot.
    if (next < tiles)
      load(next);
    float *a = sa + slot * AS, *b = sb + slot * BS;
    unsigned unsafe = 0;
    for (int w = 0; w < 8; w++)
      unsafe |= flags[slot][w];
    if (unsafe) {
      // Exact single-TF32 model outside BF16 decomposition's safe exponent
      // interval. This is a real scalar FMA path, not a matrix-instruction
      // claim.
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
          for (int x = 0; x < 8; x++) {
            int row = wm * 32 + i * 16 + lane / 16 * 8 + x;
            int col = wn * 64 + j * 16 + lane % 16;
            float sum = 0;
            for (int k = 0; k < 8; k++)
              sum = fmaf(round_tf32(a[row * (8 + kPadA) + k]),
                         round_tf32(b[k * (128 + kPadB) + col]), sum);
            accum[i][j][x] += sum;
          }
    } else {
      B8 af[2][2] = {}, bf[4][2] = {};
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int x = 0; x < 8; x++) {
          float value =
              lane < 16 ? a[(wm * 32 + i * 16 + lane) * (8 + kPadA) + x] : 0;
          short hi, lo;
          parts(round_tf32(value), hi, lo);
          af[i][0][x] = hi;
          af[i][1][x] = lo;
        }
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int x = 0; x < 8; x++) {
          float value =
              lane < 16 ? b[x * (128 + kPadB) + wn * 64 + j * 16 + lane] : 0;
          short hi, lo;
          parts(round_tf32(value), hi, lo);
          bf[j][0][x] = hi;
          bf[j][1][x] = lo;
        }
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          F8 value = {};
#pragma unroll
          for (int aa = 1; aa >= 0; aa--)
#pragma unroll
            for (int bb = 1; bb >= 0; bb--)
              value = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                  af[i][aa], bf[j][bb], value);
          accum[i][j] += value;
        }
    }
    if (next < tiles) {
      int future_slot = next % kStages;
      publish<kPadA, kPadB>(sa + future_slot * AS, sb + future_slot * BS,
                            flags[future_slot], av, bv);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int x = 0; x < 8; x++) {
        int row = bm + wm * 32 + i * 16 + lane / 16 * 8 + x;
        int col = bn + wn * 64 + j * 16 + lane % 16;
        C[row * N + col] = accum[i][j][x];
      }
}
