// B6: gfx1201 TF32x3 residual-corrected SGEMM. This is an explicit
// input-precision upgrade over the original single TF32 product, not a
// native TF32 instruction or ordinary FP32 alias. Inputs remain unchanged.
// Exceptional exponent tiles use a documented FP32 scalar FMA path.
// A is approximated by RNA_TF32(A) + RNA_TF32(A - RNA_TF32(A)); likewise B.
// Each TF32 operand is exactly decomposed into two BF16 components within
// the qualified exponent range; 12 BF16 WMMAs evaluate the three products.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
namespace {
using B8 = short __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
using F4 = float __attribute__((ext_vector_type(4)));
using U4 = unsigned __attribute__((ext_vector_type(4)));
__device__ float tf32(float x) {
  unsigned b = __float_as_uint(x), magnitude = b & 0x7fffffff;
  if (magnitude >= 0x7f800000)
    return x;
  return __uint_as_float((b + 0x1000u) & 0xffffe000u);
}
__device__ void parts(float x, short &hi, short &lo) {
  unsigned u = __float_as_uint(x);
  hi = short(u >> 16);
  float residual = x - __uint_as_float(u & 0xffff0000u);
  lo = short(__float_as_uint(residual) >> 16);
}
__device__ U4 issue(const float *p) {
  U4 u;
  asm volatile("global_load_b128 %0, %1, off" : "=v"(u) : "v"(p) : "memory");
  return u;
}
__device__ F4 finish(U4 &u) {
  asm volatile("s_wait_loadcnt 0" : "+v"(u)::"memory");
  return __builtin_bit_cast(F4, u);
}
__device__ void issue_tile(const float *a, const float *b, int m, int n, int k,
                           int bm, int bn, int t, U4 &av, U4 &bv) {
  int i = threadIdx.x * 4;
  int r = bm + i / 8, c = t * 8 + i % 8;
  if (r < m && c + 4 <= k && k % 4 == 0 && uintptr_t(a) % 16 == 0)
    av = issue(a + r * k + c);
  else {
    F4 v = {};
    for (int j = 0; j < 4; j++)
      if (r < m && c + j < k)
        v[j] = a[r * k + c + j];
    av = __builtin_bit_cast(U4, v);
  }
  r = t * 8 + i / 128;
  c = bn + i % 128;
  if (r < k && c + 4 <= n && n % 4 == 0 && uintptr_t(b) % 16 == 0)
    bv = issue(b + r * n + c);
  else {
    F4 v = {};
    for (int j = 0; j < 4; j++)
      if (r < k && c + j < n)
        v[j] = b[r * n + c + j];
    bv = __builtin_bit_cast(U4, v);
  }
}
__device__ void publish(float *p, unsigned *flags, U4 &av, U4 &bv) {
  int i = threadIdx.x * 4;
  F4 a = finish(av), b = finish(bv);
  bool unsafe = false;
  for (int j = 0; j < 4; j++) {
    p[i + j] = a[j];
    p[1024 + (i % 128 + j) * 8 + i / 128] = b[j];
    for (float x : {a[j], b[j]}) {
      float magnitude = fabsf(x);
      unsafe |=
          magnitude != 0 && !(magnitude >= 0x1p-96f && magnitude <= 0x1p112f);
    }
  }
  unsigned any = __any_sync(0xffffffffULL, unsafe);
  if (threadIdx.x % 32 == 0)
    flags[threadIdx.x / 32] = any;
}
template <bool Dynamic, int Stages, bool Correct>
__global__ void __launch_bounds__(256)
    tf32_gemm(const float *a, const float *b, float *c, int m, int n, int k) {
  extern __shared__ float dynamic[];
  float *lds;
  __shared__ unsigned flags[Stages * 8];
  if constexpr (Dynamic)
    lds = dynamic;
  else {
    __shared__ float stat[Stages * 2048];
    lds = stat;
  }
  int bm = blockIdx.y * 128, bn = (blockIdx.z * gridDim.x + blockIdx.x) * 128;
  if (bm >= m || bn >= n)
    return;
  int lane = threadIdx.x % 32, wave = threadIdx.x / 32, wr = wave / 4,
      wc = wave % 4;
  F8 acc[4][2] = {};
  U4 ar, br;
  int tiles = (k + 7) / 8;
  for (int t = 0; t < Stages - 1; t++)
    if (t < tiles) {
      issue_tile(a, b, m, n, k, bm, bn, t, ar, br);
      publish(lds + t * 2048, flags + t * 8, ar, br);
    }
  __syncthreads();
  for (int t = 0; t < tiles; t++) {
    int future = t + Stages - 1;
    if (future < tiles)
      issue_tile(a, b, m, n, k, bm, bn, future, ar, br);
    float *cur = lds + (t % Stages) * 2048;
    unsigned unsafe = 0;
#pragma unroll
    for (int i = 0; i < 8; i++)
      unsafe |= flags[(t % Stages) * 8 + i];
    if (unsafe) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          int col = (wc * 2 + j) * 16 + lane % 16;
#pragma unroll
          for (int x = 0; x < 8; x++) {
            int row = (wr * 4 + i) * 16 + lane / 16 * 8 + x;
            float sum = 0;
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
              float av = cur[row * 8 + kk], bv = cur[1024 + col * 8 + kk];
              if constexpr (!Correct) {
                av = tf32(av);
                bv = tf32(bv);
              }
              sum = fmaf(av, bv, sum);
            }
            acc[i][j][x] += sum;
          }
        }
      }
    } else {
      B8 af[4][4] = {}, bf[2][4] = {};
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int x = 0; x < 8; x++) {
          float v = lane < 16 ? cur[((wr * 4 + i) * 16 + lane) * 8 + x] : 0;
          float hi = tf32(v);
          short h, l;
          parts(hi, h, l);
          af[i][0][x] = h;
          af[i][1][x] = l;
          if constexpr (Correct) {
            parts(tf32(v - hi), h, l);
            af[i][2][x] = h;
            af[i][3][x] = l;
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 2; j++) {
#pragma unroll
        for (int x = 0; x < 8; x++) {
          float v =
              lane < 16 ? cur[1024 + ((wc * 2 + j) * 16 + lane) * 8 + x] : 0;
          float hi = tf32(v);
          short h, l;
          parts(hi, h, l);
          bf[j][0][x] = h;
          bf[j][1][x] = l;
          if constexpr (Correct) {
            parts(tf32(v - hi), h, l);
            bf[j][2][x] = h;
            bf[j][3][x] = l;
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          F8 tile = {};
// Low terms first; omitting residual*residual is an explicit TF32x3 model.
#pragma unroll
          for (int aa = 3; aa >= 0; aa--) {
#pragma unroll
            for (int bb = 3; bb >= 0; bb--) {
              if (aa < 2 && bb < 2 || Correct && (aa < 2 || bb < 2))
                tile = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    af[i][aa], bf[j][bb], tile);
            }
          }
          acc[i][j] += tile;
        }
      }
    }
    if (future < tiles)
      publish(lds + (future % Stages) * 2048, flags + (future % Stages) * 8, ar,
              br);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int col = bn + (wc * 2 + j) * 16 + lane % 16;
#pragma unroll
      for (int x = 0; x < 8; x++) {
        int row = bm + (wr * 4 + i) * 16 + lane / 16 * 8 + x;
        if (row < m && col < n)
          c[row * n + col] = acc[i][j][x];
      }
    }
  }
}
template <bool D, int S, bool C>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c, bool swizzle,
            int stride) {
  dim3 grid((b.size(1) + 127) / 128, (a.size(0) + 127) / 128);
  if (swizzle) {
    grid.x = (stride + 127) / 128;
    grid.z = (b.size(1) + stride - 1) / stride;
  }
  constexpr int smem = D ? S * 2048 * 4 : 0;
  tf32_gemm<D, S, C><<<grid, 256, smem>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), a.size(0),
      b.size(1), a.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

namespace {
void check(torch::Tensor a, torch::Tensor b, torch::Tensor c, bool swizzle,
           int stride) {
  TORCH_CHECK(a.is_cuda() && a.device() == b.device() &&
                  a.device() == c.device(),
              "expected one GPU");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2 &&
                  a.size(1) == b.size(0) && c.size(0) == a.size(0) &&
                  c.size(1) == b.size(1),
              "invalid GEMM shapes");
  for (auto t : {a, b, c}) {
    TORCH_CHECK(t.scalar_type() == at::kFloat && t.is_contiguous(),
                "expected contiguous FP32");
    TORCH_CHECK(!t.requires_grad(), "forward only");
    TORCH_CHECK(t.numel() < INT32_MAX, "index range exceeded");
  }
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b), "output aliases input");
  TORCH_CHECK(a.size(0) < INT32_MAX - 128 && a.size(1) < INT32_MAX - 8 &&
                  b.size(1) < INT32_MAX - 128 &&
                  (a.size(0) + 127) / 128 <= 65535,
              "index/grid range exceeded");
  TORCH_CHECK(!swizzle || (stride > 0 && stride % 256 == 0 &&
                           stride < INT32_MAX - 128 &&
                           (b.size(1) + int64_t(stride) - 1) / stride <= 65535),
              "invalid swizzle stride/grid");
}
template <bool D, bool C = true>
void dispatch(torch::Tensor a, torch::Tensor b, torch::Tensor c, int stage,
              bool swizzle, int stride) {
  check(a, b, c, swizzle, stride);
  if (!c.numel())
    return;
  c10::cuda::CUDAGuard guard(a.device());
  switch (stage) {
  case 3:
    launch<D, 3, C>(a, b, c, swizzle, stride);
    break;
  case 4:
    launch<D, 4, C>(a, b, c, swizzle, stride);
    break;
  case 5:
    if constexpr (D) {
      launch<D, 5, C>(a, b, c, swizzle, stride);
      break;
    }
    [[fallthrough]];
  default:
    launch<D, 2, C>(a, b, c, swizzle, stride);
    break;
  }
}
} // namespace
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages(torch::Tensor a, torch::Tensor b,
                                               torch::Tensor c, int stages,
                                               bool swizzle, int stride) {
  dispatch<false>(a, b, c, stages, swizzle, stride);
}
void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem(torch::Tensor a,
                                                     torch::Tensor b,
                                                     torch::Tensor c,
                                                     int stages, bool swizzle,
                                                     int stride) {
  dispatch<true>(a, b, c, stages, swizzle, stride);
}
#ifdef LEETCUDA_TF32_CONTROL
void tf32_strict_static(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                        int stages, bool swizzle, int stride) {
  dispatch<false, false>(a, b, c, stages, swizzle, stride);
}
void tf32_strict_dynamic(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                         int stages, bool swizzle, int stride) {
  dispatch<true, false>(a, b, c, stages, swizzle, stride);
}
__global__ void tf32_round_kernel(const float *a, float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    b[i] = tf32(a[i]);
}
void tf32_round_probe(torch::Tensor a, torch::Tensor b) {
  c10::cuda::CUDAGuard guard(a.device());
  int blocks = (a.numel() + 255) / 256;
  tf32_round_kernel<<<blocks, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                     a.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
