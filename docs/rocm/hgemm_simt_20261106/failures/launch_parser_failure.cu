// Explicit FP32-state upgrades of eleven original synchronous SIMT variants.
// Original tile geometry, vector widths, BCF layouts and LDS slot dataflow are
// retained on aligned full tiles. Bounded native scalar tails extend the API.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
namespace {
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
__device__ __forceinline__ float accumulate(half a, half b, float c) {
  return fmaf(__half2float(a), __half2float(b), c);
}
__device__ __forceinline__ half2 pack2(const float *x) {
  return __floats2half2_rn(x[0], x[1]);
}
__device__ __forceinline__ float2 pack4(const float *x) {
  float2 out;
#pragma unroll
  for (int j = 0; j < 4; j++)
    reinterpret_cast<half *>(&out)[j] = __float2half_rn(x[j]);
  return out;
}
__device__ __forceinline__ float4 pack8(const float *x) {
  float4 out;
#pragma unroll
  for (int j = 0; j < 8; j++)
    reinterpret_cast<half *>(&out)[j] = __float2half_rn(x[j]);
  return out;
}

__global__ void hgemm_naive_f16_fp32_kernel(half *a, half *b, half *c, int M,
                                            int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M && n < N) {
    float psum = 0.0;
#pragma unroll
    for (int k = 0; k < K; k++) {

      psum = accumulate(a[m, K + k] * b[k * N + n], psum);
    }
    c[m * N + n] = psum;
  }
}
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void hgemm_sliced_k_f16_fp32_kernel(half *a, half *b, half *c, int M,
                                               int N, int K) {

  __shared__ half s_a[BM][BK], s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;

  int load_smem_a_m = tid / 32;
  int load_smem_a_k = tid % 32;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = tid % 32;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum = accumulate(s_a[comp_smem_a_m][k], s_b[k][comp_smem_b_n], sum);
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_fp32_kernel(half *a, half *b,
                                                       half *c, int M, int N,
                                                       int K) {

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;
  __shared__ half s_a[BM][BK], s_b[BK][BN];

  int load_smem_a_m = tid / 2;

  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;

  int load_smem_b_k = tid / 32;
  int load_smem_b_n = (tid % 32) * 4;

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  float r_c[TM][TN] = {0.f};

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    HALF2(s_a[load_smem_a_m][load_smem_a_k + 0]) =
        HALF2(a[load_gmem_a_addr + 0]);
    HALF2(s_a[load_smem_a_m][load_smem_a_k + 2]) =
        HALF2(a[load_gmem_a_addr + 2]);

    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    HALF2(s_b[load_smem_b_k][load_smem_b_n + 0]) =
        HALF2(b[load_gmem_b_addr + 0]);
    HALF2(s_b[load_smem_b_k][load_smem_b_n + 2]) =
        HALF2(b[load_gmem_b_addr + 2]);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {

#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {

          int comp_smem_a_m = ty * TM + m;
          int comp_smem_b_n = tx * TN + n;
          r_c[m][n] = accumulate(s_a[comp_smem_a_m][k], s_b[k][comp_smem_b_n],
                                 r_c[m][n]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 2) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      HALF2(c[store_gmem_c_addr]) = pack2(&r_c[m][n]);
    }
  }
}
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_fp32_kernel(half *a, half *b,
                                                            half *c, int M,
                                                            int N, int K) {

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;
  __shared__ half s_a[BM][BK], s_b[BK][BN];

  int load_smem_a_m = tid / 2;

  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;

  int load_smem_b_k = tid / 32;
  int load_smem_b_n = (tid % 32) * 4;

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  float r_c[TM][TN] = {0.f};

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) =
        LDST64BITS(a[load_gmem_a_addr]);

    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    LDST64BITS(s_b[load_smem_b_k][load_smem_b_n]) =
        LDST64BITS(b[load_gmem_b_addr]);

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; k++) {

#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {

          int comp_smem_a_m = ty * TM + m;
          int comp_smem_b_n = tx * TN + n;

          r_c[m][n] = accumulate(s_a[comp_smem_a_m][k], s_b[k][comp_smem_b_n],
                                 r_c[m][n]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      LDST64BITS(c[store_gmem_c_addr]) = pack4(&r_c[m][n]);
    }
  }
}
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_bcf_fp32_kernel(half *a, half *b,
                                                           half *c, const int M,
                                                           const int N,
                                                           const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[BK][BM];
  __shared__ half s_b[BK][BN];

  half r_load_a[TM / 2];
  half r_load_b[TN / 2];
  half r_comp_a[TM];
  half r_comp_b[TN];
  float r_c[TM][TN] = {0.f};

  int load_a_smem_m = tid / 2;

  int load_a_smem_k = (tid & 1) << 2;

  int load_b_smem_k = tid / 32;

  int load_b_smem_n = (tid & 31) << 2;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;
  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    HALF2(r_load_a[0]) = HALF2(a[load_a_gmem_addr + 0]);
    HALF2(r_load_a[2]) = HALF2(a[load_a_gmem_addr + 2]);
    HALF2(r_load_b[0]) = HALF2(b[load_b_gmem_addr + 0]);
    HALF2(r_load_b[2]) = HALF2(b[load_b_gmem_addr + 2]);

    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    HALF2(s_b[load_b_smem_k][load_b_smem_n + 0]) = HALF2(r_load_b[0]);
    HALF2(s_b[load_b_smem_k][load_b_smem_n + 2]) = HALF2(r_load_b[2]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      HALF2(r_comp_a[0]) = HALF2(s_a[tk][ty * TM / 2]);
      HALF2(r_comp_a[2]) = HALF2(s_a[tk][ty * TM / 2 + 2]);
      HALF2(r_comp_a[4]) = HALF2(s_a[tk][ty * TM / 2 + BM / 2]);
      HALF2(r_comp_a[6]) = HALF2(s_a[tk][ty * TM / 2 + BM / 2 + 2]);

      HALF2(r_comp_b[0]) = HALF2(s_b[tk][tx * TN / 2]);
      HALF2(r_comp_b[2]) = HALF2(s_b[tk][tx * TN / 2 + 2]);
      HALF2(r_comp_b[4]) = HALF2(s_b[tk][tx * TN / 2 + BN / 2]);
      HALF2(r_comp_b[6]) = HALF2(s_b[tk][tx * TN / 2 + BN / 2 + 2]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    HALF2(c[store_c_gmem_addr + 0]) = pack2(&r_c[i][0]);
    HALF2(c[store_c_gmem_addr + 2]) = pack2(&r_c[i][2]);
    HALF2(c[store_c_gmem_addr + BN / 2 + 0]) = pack2(&r_c[i][4]);
    HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = pack2(&r_c[i][6]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    HALF2(c[store_c_gmem_addr + 0]) = pack2(&r_c[i + TM / 2][0]);
    HALF2(c[store_c_gmem_addr + 2]) = pack2(&r_c[i + TM / 2][2]);
    HALF2(c[store_c_gmem_addr + BN / 2 + 0]) = pack2(&r_c[i + TM / 2][4]);
    HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = pack2(&r_c[i + TM / 2][6]);
  }
}
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_bcf_fp32_kernel(
    half *a, half *b, half *c, const int M, const int N, const int K) {

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[BK][BM + OFFSET];
  __shared__ half s_b[BK][BN + OFFSET];

  half r_load_a[TM / 2];
  half r_load_b[TN / 2];
  half r_comp_a[TM];
  half r_comp_b[TN];
  float r_c[TM][TN] = {0.f};

  int load_a_smem_m = tid / 2;

  int load_a_smem_k = (tid & 1) << 2;

  int load_b_smem_k = tid / 32;

  int load_b_smem_n = (tid & 31) << 2;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;
  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

    LDST64BITS(s_b[load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {

      LDST64BITS(r_comp_a[0]) = LDST64BITS(s_a[tk][ty * TM / 2]);
      LDST64BITS(r_comp_a[4]) = LDST64BITS(s_a[tk][ty * TM / 2 + BM / 2]);

      LDST64BITS(r_comp_b[0]) = LDST64BITS(s_b[tk][tx * TN / 2]);
      LDST64BITS(r_comp_b[4]) = LDST64BITS(s_b[tk][tx * TN / 2 + BN / 2]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {

          r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    LDST64BITS(c[store_c_gmem_addr]) = pack4(&r_c[i][0]);
    LDST64BITS(c[store_c_gmem_addr + BN / 2]) = pack4(&r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    LDST64BITS(c[store_c_gmem_addr]) = pack4(&r_c[i + TM / 2][0]);
    LDST64BITS(c[store_c_gmem_addr + BN / 2]) = pack4(&r_c[i + TM / 2][4]);
  }
}
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_fp32_kernel(
    half *a, half *b, half *c, const int M, const int N, const int K) {

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[BK][BM + OFFSET];
  __shared__ half s_b[BK][BN + OFFSET];

  half r_load_a[TM / 2];
  half r_load_b[TN / 2];
  half r_comp_a[TM];
  half r_comp_b[TN];
  float r_c[TM][TN] = {0.f};

  int load_a_smem_m = tid / 2;

  int load_a_smem_k = (tid & 1) << 2;

  int load_b_smem_k = tid / 32;

  int load_b_smem_n = (tid & 31) << 2;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;
  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

    LDST64BITS(s_b[load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {

      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[tk][ty * TM]);

      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[tk][tx * TN]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {

          r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty * TM + i;
    int store_c_gmem_n = bx * BN + tx * TN;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    LDST128BITS(c[store_c_gmem_addr]) = pack8(&r_c[i][0]);
  }
}
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_fp32_kernel(
    half *a, half *b, half *c, const int M, const int N, const int K) {

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[2][BK][BM + OFFSET];
  __shared__ half s_b[2][BK][BN + OFFSET];

  half r_load_a[TM / 2];
  half r_load_b[TN / 2];
  half r_comp_a[TM];
  half r_comp_b[TN];
  float r_c[TM][TN] = {0.f};

  int load_a_smem_m = tid / 2;

  int load_a_smem_k = (tid & 1) << 2;

  int load_b_smem_k = tid / 32;

  int load_b_smem_n = (tid & 31) << 2;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;
  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    LDST64BITS(s_b[0][load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);
  }

  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
    LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    LDST64BITS(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) =
        LDST64BITS(r_load_b[0]);

    __syncthreads();
  }

#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    LDST128BITS(r_comp_a[0]) =
        LDST128BITS(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM]);
    LDST128BITS(r_comp_b[0]) =
        LDST128BITS(s_b[((K + BK - 1) / BK - 1) & 1][tk][tx * TN]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty * TM + i;
    int store_c_gmem_n = bx * BN + tx * TN;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    LDST128BITS(c[store_c_gmem_addr]) = pack8(&r_c[i][0]);
  }
}
template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void
hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_fp32_kernel(half *a, half *b, half *c,
                                                   int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;

  __shared__ half s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];
  half r_load_a[TM];
  half r_load_b[TN];
  half r_comp_a[TM];
  half r_comp_b[TN];
  float r_c[TM][TN] = {0.f};

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;

  int load_smem_b_k = tid / 16;
  int load_smem_b_n = (tid % 16) * 8;

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  {
    int load_gmem_a_k = load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    LDST128BITS(s_b[0][load_smem_b_k][load_smem_b_n]) =
        (LDST128BITS(b[load_gmem_b_addr]));
    LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
  }
  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
    LDST128BITS(r_load_b[0]) = LDST128BITS(b[load_gmem_b_addr]);

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
    LDST128BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]) =
        (LDST128BITS(r_load_b[0]));

    __syncthreads();
  }

#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    LDST128BITS(r_comp_a[0]) =
        LDST128BITS(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM]);
    LDST128BITS(r_comp_b[0]) =
        LDST128BITS(s_b[((K + BK - 1) / BK - 1) & 1][tk][tx * TN]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int store_gmem_c_m = by * BM + ty * TM + i;
    int store_gmem_c_n = bx * BN + tx * TN;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    LDST128BITS(c[store_gmem_c_addr]) = pack8(&r_c[i][0]);
  }
}
template <const int BM = 128, const int BN = 128, const int BK = 32,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void
hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_fp32_kernel(half *a, half *b, half *c,
                                                   int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;

  __shared__ half s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];
  half r_load_a[16];
  half r_comp_a[TM];
  half r_comp_b[TN];
  float r_c[TM][TN] = {0.f};

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 16;

  int load_smem_b_k = tid / 8;
  int load_smem_b_n = (tid % 8) * 16;

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  {
    int load_gmem_a_k = load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

#pragma unroll
    for (int i = 0; i < 16; i += 8) {
      LDST128BITS(s_b[0][load_smem_b_k][load_smem_b_n + i]) =
          (LDST128BITS(b[load_gmem_b_addr + i]));
      LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
  }
  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 16; i += 8) {
      LDST128BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n + i]) =
          (LDST128BITS(b[load_gmem_b_addr + i]));
      LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
    __syncthreads();
  }

#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    LDST128BITS(r_comp_a[0]) =
        LDST128BITS(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM]);
    LDST128BITS(r_comp_b[0]) =
        LDST128BITS(s_b[((K + BK - 1) / BK - 1) & 1][tk][tx * TN]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int store_gmem_c_m = by * BM + ty * TM + i;
    int store_gmem_c_n = bx * BN + tx * TN;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    LDST128BITS(c[store_gmem_c_addr]) = pack8(&r_c[i][0]);
  }
}
template <const int BM = 128, const int BN = 128, const int BK = 32,
          const int TM = 16, const int TN = 8, const int OFFSET = 0>
__global__ void
hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_fp32_kernel(half *a, half *b, half *c,
                                                    int M, int N, int K) {

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;

  __shared__ half s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];
  half r_load_a[32];
  half r_comp_a[TM];
  half r_comp_b[TN];
  float r_c[TM][TN] = {0.f};

  int load_smem_a_m = tid;
  int load_smem_a_k = 0;

  int load_smem_b_k = tid / 4;
  int load_smem_b_n = (tid % 4) * 32;

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  {
    int load_gmem_a_k = load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

#pragma unroll
    for (int i = 0; i < 32; i += 8) {
      LDST128BITS(s_b[0][load_smem_b_k][load_smem_b_n + i]) =
          (LDST128BITS(b[load_gmem_b_addr + i]));
      LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
  }
  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
      LDST128BITS(r_comp_a[8]) = LDST128BITS(s_a[smem_sel][tk][ty * TM + 8]);
      LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 32; i += 8) {
      LDST128BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n + i]) =
          (LDST128BITS(b[load_gmem_b_addr + i]));
      LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
    }
    __syncthreads();
  }

#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    LDST128BITS(r_comp_a[0]) =
        LDST128BITS(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM]);
    LDST128BITS(r_comp_a[8]) =
        LDST128BITS(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM + 8]);
    LDST128BITS(r_comp_b[0]) =
        LDST128BITS(s_b[((K + BK - 1) / BK - 1) & 1][tk][tx * TN]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = accumulate(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int store_gmem_c_m = by * BM + ty * TM + i;
    int store_gmem_c_n = bx * BN + tx * TN;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    LDST128BITS(c[store_gmem_c_addr]) = pack8(&r_c[i][0]);
  }
}

void validate(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  for (const auto &t : {a, b, c}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kHalf && t.dim() == 2 &&
                    t.is_contiguous(),
                "expected contiguous rank-2 FP16 GPU tensors");
    TORCH_CHECK(!t.requires_grad(), "forward only");
    TORCH_CHECK(t.numel() < INT32_MAX - 16384 && t.size(0) < INT32_MAX - 256 &&
                    t.size(1) < INT32_MAX - 256,
                "index limit");
  }
  TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
              "same device required");
  TORCH_CHECK(a.size(1) == b.size(0) && c.size(0) == a.size(0) &&
                  c.size(1) == b.size(1),
              "shape mismatch");
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b),
              "output alias unsupported");
  TORCH_CHECK((a.size(0) + 31) / 32 <= 65535, "grid y limit");
}
} // namespace

void hgemm_naive_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 32 == 0 && N % 32 == 0 && K % 1 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 2 == 0;
  if (true) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_naive_f16_fp32_kernel<<<dim3(N / 32, M / 32), dim3(32, 32)>>>(
        A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_sliced_k_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 32 == 0 && N % 32 == 0 && K % 32 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 2 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_sliced_k_f16_fp32_kernel<32, 32, 32>
        <<<dim3(N / 32, M / 32), dim3(32, 32)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_8x8_sliced_k_f16x4(torch::Tensor a, torch::Tensor b,
                                torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 4 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_8x8_sliced_k_f16x4_fp32_kernel<128, 128, 8, 8, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 16)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_8x8_sliced_k_f16x4_pack(torch::Tensor a, torch::Tensor b,
                                     torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 8 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_8x8_sliced_k_f16x4_pack_fp32_kernel<128, 128, 8, 8, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 16)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_8x8_sliced_k_f16x4_bcf(torch::Tensor a, torch::Tensor b,
                                    torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 4 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_8x8_sliced_k_f16x4_bcf_fp32_kernel<128, 128, 8, 8, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 16)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_8x8_sliced_k_f16x4_pack_bcf(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 8 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_8x8_sliced_k_f16x4_pack_bcf_fp32_kernel<128, 128, 8, 8, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 16)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_8x8_sliced_k_f16x8_pack_bcf(torch::Tensor a, torch::Tensor b,
                                         torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 16 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_8x8_sliced_k_f16x8_pack_bcf_fp32_kernel<128, 128, 8, 8, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 16)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(torch::Tensor a, torch::Tensor b,
                                              torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 16 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_fp32_kernel<128, 128, 8, 8, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 16)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b,
                                            torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 16 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 16 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_fp32_kernel<128, 128, 16, 8, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 16)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b,
                                            torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 32 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 16 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_fp32_kernel<128, 128, 32, 8, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 16)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(torch::Tensor a, torch::Tensor b,
                                             torch::Tensor c) {
  validate(a, b, c);
  c10::cuda::CUDAGuard guard(a.device());
  int M = a.size(0), N = b.size(1), K = a.size(1);
  if (!M || !N)
    return;
  auto A = reinterpret_cast<half *>(a.data_ptr());
  auto B = reinterpret_cast<half *>(b.data_ptr());
  auto C = reinterpret_cast<half *>(c.data_ptr());
  bool full = K > 0 && M % 128 == 0 && N % 128 == 0 && K % 32 == 0 &&
              (uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) % 16 == 0;
  if (!full) {
    hgemm_naive_f16_fp32_kernel<<<dim3((N + 31) / 32, (M + 31) / 32),
                                  dim3(32, 32)>>>(A, B, C, M, N, K);
  } else {
    hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_fp32_kernel<128, 128, 32, 16, 8>
        <<<dim3(N / 128, M / 128), dim3(16, 8)>>>(A, B, C, M, N, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
