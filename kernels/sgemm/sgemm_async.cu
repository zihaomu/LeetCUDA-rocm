#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#ifdef LEETCUDA_ROCM
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#endif

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                                               \
    asm volatile(                                                                                  \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                                               \
    asm volatile(                                                                                  \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

template <const int BM = 64,
    const int BN = 64,
    const int BK = 16,
    const int TM = 8,
    const int TN = 4,
    const int OFFSET = 0>
__global__ void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // block(BN/TN, BM/TM) -> (x=16,y=8), 128 threads
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    // 2*(16*64*4)=8KB, 8+8=16KB, 128KB/16=8 blocks
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[8]; // load 8 values per thread
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0}; // 8x4

    // 128 threads, tx: 0~15, ty: 0~7
    int load_a_smem_m = tid / 2;                // (0,1,2,...,63)
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
    int load_b_smem_k = tid / 8;                // 0~15
    int load_b_smem_n = (tid % 8) * 8;          // (0,8,16,...,56)
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (FLOAT4(b[load_b_gmem_addr + i]));
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) =
                (FLOAT4(b[load_b_gmem_addr + i]));
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // ty: 0~7, (0~7)*8=64; tx: 0~15, (0~15)*4=64
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM + 4]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM + 4]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[((K + BK - 1) / BK - 1) & 1][tk][tx * TN]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        int store_c_gmem_n = bx * BN + tx * TN;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    }
}

#ifndef LEETCUDA_ROCM
template <const int BM = 64,
    const int BN = 64,
    const int BK = 16,
    const int TM = 8,
    const int TN = 4,
    const int OFFSET = 0>
__global__ void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // block(BN/TN, BM/TM) -> (x=16,y=8), 128 threads
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    // 2*(16*64*4)=8KB, 8+8=16KB, 128KB/16=8 blocks
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[8]; // load 8 values per thread
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0}; // 8x4

    // 128 threads, tx: 0~15, ty: 0~7
    int load_a_smem_m = tid / 2;                // (0,1,2,...,63)
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
    int load_b_smem_k = tid / 8;                // 0~15
    int load_b_smem_n = (tid % 8) * 8;          // (0,8,16,...,56)
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_b_smem_ptr =
            __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM + 4]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM + 4]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        int store_c_gmem_n = bx * BN + tx * TN;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    }
}
#endif // LEETCUDA_ROCM

#ifdef LEETCUDA_ROCM
// RDNA4 has no direct global-to-LDS instruction. Preserve the original two LDS
// buffers and per-thread output layout, but prefetch A and B into VGPRs while
// computing the previous tile. The explicit load-counter wait is required:
// these inline-assembly loads are not tracked by LLVM's memory hazard pass.
namespace leetcuda_amd_async {
using Vec4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ Vec4 prefetch(const float *source) {
  Vec4 value;
  asm volatile("global_load_b128 %0, %1, off"
               : "=v"(value)
               : "v"(source)
               : "memory");
  return value;
}

template <int LB>
__device__ __forceinline__ void wait_loads(Vec4 (&a)[2], Vec4 (&b)[LB / 4]) {
  // Read/write constraints make every later use depend on this wait, including
  // register-only moves that a memory clobber alone would not constrain.
  if constexpr (LB == 8) {
    asm volatile("s_wait_loadcnt 0"
                 : "+v"(a[0]), "+v"(a[1]), "+v"(b[0]), "+v"(b[1])::"memory");
  } else {
    asm volatile("s_wait_loadcnt 0"
                 : "+v"(a[0]), "+v"(a[1]), "+v"(b[0]), "+v"(b[1]), "+v"(b[2]),
                   "+v"(b[3])::"memory");
  }
}

template <int BM, int BN, int TN>
__device__ __forceinline__ void issue(const float *a, const float *b, int n,
                                      int k, int tile, int tid, Vec4 (&ra)[2],
                                      Vec4 (&rb)[(TN == 16 ? 16 : 8) / 4]) {
  constexpr int LB = TN == 16 ? 16 : 8;
  const int am = blockIdx.y * BM + tid / 2;
  const int ak = tile * 16 + (tid % 2) * 8;
  const int bk = tile * 16 + tid / (BN / LB);
  const int bn = blockIdx.x * BN + (tid % (BN / LB)) * LB;
#pragma unroll
  for (int i = 0; i < 2; ++i)
    ra[i] = prefetch(a + am * k + ak + 4 * i);
#pragma unroll
  for (int i = 0; i < LB / 4; ++i)
    rb[i] = prefetch(b + bk * n + bn + 4 * i);
}

template <int BM, int BN, int TN>
__device__ __forceinline__ void
stage(Vec4 (&ra)[2], Vec4 (&rb)[(TN == 16 ? 16 : 8) / 4], float (&sa)[16][BM],
      float (&sb)[16][BN], int tid) {
  constexpr int LB = TN == 16 ? 16 : 8;
  wait_loads<LB>(ra, rb);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    sa[(tid % 2) * 8 + i][tid / 2] = ra[i / 4][i % 4];
#pragma unroll
  for (int i = 0; i < LB / 4; ++i)
    *reinterpret_cast<Vec4 *>(
        &sb[tid / (BN / LB)][(tid % (BN / LB)) * LB + i * 4]) = rb[i];
}

template <int BM, int BN, int TN>
__device__ __forceinline__ void compute(const float (&sa)[16][BM],
                                        const float (&sb)[16][BN],
                                        float (&acc)[8][TN]) {
  // Full K unrolling spilled the 8x16 variant's live prefetch registers;
  // scratch-load waits then drained pending global reads before computation.
#pragma unroll 1
  for (int kk = 0; kk < 16; ++kk) {
    float av[8], bv[TN];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = TN == 8 ? threadIdx.y * 4 + (i % 4) + (i / 4) * (BM / 2)
                              : threadIdx.y * 8 + i;
      av[i] = sa[kk][row];
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = TN == 8 ? threadIdx.x * 4 + (j % 4) + (j / 4) * (BN / 2)
                              : threadIdx.x * TN + j;
      bv[j] = sb[kk][col];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
        acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
  }
}

template <int BM, int BN, int TN>
__global__ void sgemm_prefetch_kernel(const float *a, const float *b, float *c,
                                      int m, int n, int k) {
  __shared__ float sa[2][16][BM];
  __shared__ float sb[2][16][BN];
  constexpr int LB = TN == 16 ? 16 : 8;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  Vec4 ra[2], rb[LB / 4];
  float acc[8][TN] = {};
  issue<BM, BN, TN>(a, b, n, k, 0, tid, ra, rb);
  stage<BM, BN, TN>(ra, rb, sa[0], sb[0], tid);
  __syncthreads();
  for (int tile = 1; tile < k / 16; ++tile) {
    issue<BM, BN, TN>(a, b, n, k, tile, tid, ra, rb);
    __builtin_amdgcn_sched_barrier(0);
    compute<BM, BN, TN>(sa[(tile - 1) & 1], sb[(tile - 1) & 1], acc);
    __builtin_amdgcn_sched_barrier(0);
    stage<BM, BN, TN>(ra, rb, sa[tile & 1], sb[tile & 1], tid);
    __syncthreads();
  }
  compute<BM, BN, TN>(sa[(k / 16 - 1) & 1], sb[(k / 16 - 1) & 1], acc);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = TN == 8 ? threadIdx.y * 4 + (i % 4) + (i / 4) * (BM / 2)
                            : threadIdx.y * 8 + i;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const int col =
          TN == 8 ? threadIdx.x * 4 + (j / 4) * (BN / 2) : threadIdx.x * TN + j;
      const Vec4 result = {acc[i][j], acc[i][j + 1], acc[i][j + 2],
                           acc[i][j + 3]};
      *reinterpret_cast<Vec4 *>(c + (blockIdx.y * BM + row) * n +
                                blockIdx.x * BN + col) = result;
    }
  }
}

template <int BM, int BN, int TN>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  for (const auto &t : {a, b, c}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 &&
                    t.dim() == 2 && t.is_contiguous() && !t.requires_grad(),
                "expected contiguous inference-only GPU FP32 matrices");
    TORCH_CHECK(t.device() == a.device(), "all matrices must share a device");
    TORCH_CHECK(t.numel() <= INT_MAX &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "matrix index overflow or missing 16-byte alignment");
  }
  const int m = a.size(0), k = a.size(1), n = b.size(1);
  TORCH_CHECK(m > 0 && n > 0 && k > 0 && m % BM == 0 && n % BN == 0 &&
                  k % 16 == 0,
              "async SGEMM requires M%", BM, "=0, N%", BN,
              "=0, K%16=0 and positive dimensions");
  TORCH_CHECK(b.size(0) == k && c.size(0) == m && c.size(1) == n,
              "GEMM shape mismatch");
  const auto begin = reinterpret_cast<uintptr_t>(c.data_ptr()),
             end = begin + c.nbytes();
  for (const auto &t : {a, b}) {
    const auto p = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK(end <= p || p + t.nbytes() <= begin,
                "output must not overlap inputs");
  }
  const c10::cuda::CUDAGuard guard(a.device());
  const dim3 block(BN / TN, BM / 8), grid(n / BN, m / BM);
  sgemm_prefetch_kernel<BM, BN, TN><<<grid, block>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace leetcuda_amd_async

void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a,
                                                 torch::Tensor b,
                                                 torch::Tensor c) {
  leetcuda_amd_async::launch<64, 64, 4>(a, b, c);
}
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a,
                                                 torch::Tensor b,
                                                 torch::Tensor c) {
  leetcuda_amd_async::launch<128, 128, 8>(a, b, c);
}
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async(torch::Tensor a,
                                                  torch::Tensor b,
                                                  torch::Tensor c) {
  leetcuda_amd_async::launch<128, 256, 16>(a, b, c);
}
#endif // LEETCUDA_ROCM

template <const int BM = 128,
    const int BN = 128,
    const int BK = 16,
    const int TM = 8,
    const int TN = 8,
    const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[8]; // load 8 values per thread
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0}; // 8x8

    // 256 threads, tx: 0~15, ty: 0~7
    int load_a_smem_m = tid / 2;                // (0,1,2,...,128)
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
    int load_b_smem_k = tid / 16;               // 0~15
    int load_b_smem_n = (tid % 16) * 8;         // (0,8,16,...,128)
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (FLOAT4(b[load_b_gmem_addr + i]));
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) =
                (FLOAT4(b[load_b_gmem_addr + i]));
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[((K + BK - 1) / BK - 1) & 1][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[((K + BK - 1) / BK - 1) & 1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

#ifndef LEETCUDA_ROCM
template <const int BM = 128,
    const int BN = 128,
    const int BK = 16,
    const int TM = 8,
    const int TN = 8,
    const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[8]; // load 8 values per thread
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0}; // 8x8

    // 256 threads, tx: 0~15, ty: 0~7
    int load_a_smem_m = tid / 2;                // (0,1,2,...,128)
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
    int load_b_smem_k = tid / 16;               // 0~15
    int load_b_smem_n = (tid % 16) * 8;         // (0,8,16,...,128)
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_b_smem_ptr =
            __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
// 2 cp.async issue, 16 bytes = 4 float.
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
#endif // LEETCUDA_ROCM

// t 8x16, k 16, w/o async
template <const int BM = 128,
    const int BN = 256,
    const int BK = 16,
    const int TM = 8,
    const int TN = 16,
    const int OFFSET = 0>
__global__ void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    // (16*128*4)=8KB (16*256*4)=16KB 24KB
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[8];
    float r_comp_a[TM];        // 8
    float r_comp_b[TN];        // 16
    float r_c[TM][TN] = {0.0}; // 8x16

    // 256 threads, tx: 0~15, ty: 0~7
    // a: load 8 values per thread
    int load_a_smem_m = tid / 2;                // (0,1,2,...,128)
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
    // b: load 16 values per thread
    int load_b_smem_k = tid / 16;        // 0~15
    int load_b_smem_n = (tid % 16) * 16; // (0,16,32,...,256)
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        // gmem -> smem
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

// b: load 16 values per thread
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n + i]) = (FLOAT4(b[load_b_gmem_addr + i]));
        }
// a: load 8 values per thread
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // online transpose
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        // gmem -> smem
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

// b: load 16 values per thread
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + i]) =
                (FLOAT4(b[load_b_gmem_addr + i]));
        }

// a: load 8 values per thread
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
// smem -> regristers
#pragma unroll
            for (int r = 0; r < 8; r += 4) {
                FLOAT4(r_comp_a[r]) = FLOAT4(s_a[smem_sel][tk][ty * TM + r]);
            }
#pragma unroll
            for (int r = 0; r < 16; r += 4) {
                FLOAT4(r_comp_b[r]) = FLOAT4(s_b[smem_sel][tk][tx * TN + r]);
            }

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // online transpose
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
// smem -> regristers
#pragma unroll
        for (int r = 0; r < 8; r += 4) {
            FLOAT4(r_comp_a[r]) = FLOAT4(s_a[((K + BK - 1) / BK - 1) & 1][tk][ty * TM + r]);
        }
#pragma unroll
        for (int r = 0; r < 16; r += 4) {
            FLOAT4(r_comp_b[r]) = FLOAT4(s_b[((K + BK - 1) / BK - 1) & 1][tk][tx * TN + r]);
        }

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_c_gmem_m = by * BM + ty * TM + m;
            int store_c_gmem_n = bx * BN + tx * TN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

// t 8x16, k 16, async
#ifndef LEETCUDA_ROCM
template <const int BM = 128,
    const int BN = 256,
    const int BK = 16,
    const int TM = 8,
    const int TN = 16,
    const int OFFSET = 0>
__global__ void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_kernel(
    float* a, float* b, float* c, const int M, const int N, const int K) {
    // block(BN/TN, BM/TM) -> (x=16,y=16), 256 threads
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    // (16*128*4)=8KB (16*256*4)=16KB 24KB
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[8];
    float r_comp_a[TM];        // 8
    float r_comp_b[TN];        // 16
    float r_c[TM][TN] = {0.0}; // 8x16

    // 256 threads, tx: 0~15, ty: 0~7
    // a: load 8 values per thread
    int load_a_smem_m = tid / 2;                // (0,1,2,...,128)
    int load_a_smem_k = (tid % 2 == 0) ? 0 : 8; // (0,8)
    // b: load 16 values per thread
    int load_b_smem_k = tid / 16;        // 0~15
    int load_b_smem_n = (tid % 16) * 16; // (0,16,32,...,256)
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        // gmem -> smem
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        // b: load 16 values per thread
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
// 4 cp.async issue, 16 bytes = 4 float, 4x4=16
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

// a: load 8 values per thread
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // online transpose
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        // gmem -> smem
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        // b: load 16 values per thread
        uint32_t load_b_smem_ptr =
            __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
// 4 cp.async issue, 16 bytes = 4 float, 4x4=16
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

// a: load 8 values per thread
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));
        }

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
// smem -> regristers
#pragma unroll
            for (int r = 0; r < 8; r += 4) {
                FLOAT4(r_comp_a[r]) = FLOAT4(s_a[smem_sel][tk][ty * TM + r]);
            }
#pragma unroll
            for (int r = 0; r < 16; r += 4) {
                FLOAT4(r_comp_b[r]) = FLOAT4(s_b[smem_sel][tk][tx * TN + r]);
            }

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // online transpose
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
// smem -> regristers
#pragma unroll
        for (int r = 0; r < 8; r += 4) {
            FLOAT4(r_comp_a[r]) = FLOAT4(s_a[1][tk][ty * TM + r]);
        }
#pragma unroll
        for (int r = 0; r < 16; r += 4) {
            FLOAT4(r_comp_b[r]) = FLOAT4(s_b[1][tk][tx * TN + r]);
        }

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_c_gmem_m = by * BM + ty * TM + m;
            int store_c_gmem_n = bx * BN + tx * TN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}
#endif // LEETCUDA_ROCM

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                                       \
    if (((T).options().dtype() != (th_type))) {                                                    \
        std::cout << "Tensor Info:" << (T).options() << std::endl;                                 \
        throw std::runtime_error("values must be " #th_type);                                      \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                                        \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                                          \
        throw std::runtime_error("Tensor size mismatch!");                                         \
    }

// 8x4, k16
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 4;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}

#ifndef LEETCUDA_ROCM
void sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 4;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}
#endif // LEETCUDA_ROCM

// 8x8, k16
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}

#ifndef LEETCUDA_ROCM
void sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}
#endif // LEETCUDA_ROCM

// 8x16, k16
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 16;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}

#ifndef LEETCUDA_ROCM
void sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 16;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float*>(a.data_ptr()),
            reinterpret_cast<float*>(b.data_ptr()),
            reinterpret_cast<float*>(c.data_ptr()),
            M,
            N,
            K);
}
#endif // LEETCUDA_ROCM
