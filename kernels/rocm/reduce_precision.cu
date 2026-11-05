// Explicit precision upgrade for three original BF16-accumulating reducers.
// Keep scalar/x2/x8 packing; compensated FP32 state and deterministic second
// reduction replace BF16 partial sums and unordered global atomic addition.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
namespace {
using B8 = unsigned short __attribute__((ext_vector_type(8)));
using B2 = unsigned short __attribute__((ext_vector_type(2)));
struct Sum {
  float main, residual;
};
__device__ __forceinline__ void add(Sum &a, float b) {
  float s = a.main + b;
  // Compensation formulas contain inf-inf. Preserve IEEE nonfinite sum
  // behavior instead of manufacturing NaN for a single infinity.
  if (!isfinite(a.main) || !isfinite(b) || !isfinite(s)) {
    a.main = s;
    a.residual = 0;
    return;
  }
  float correction =
      fabsf(a.main) >= fabsf(b) ? (a.main - s) + b : (b - s) + a.main;
  a.main = s;
  a.residual += correction;
}
__device__ __forceinline__ Sum join(Sum a, Sum b) {
  add(a, b.main);
  add(a, b.residual);
  return a;
}
__device__ __forceinline__ Sum wave(Sum a) {
#pragma unroll
  for (int shift = 16; shift; shift >>= 1) {
    Sum b = {__shfl_down_sync(0xffffffffULL, a.main, shift),
             __shfl_down_sync(0xffffffffULL, a.residual, shift)};
    if (threadIdx.x % 32 + shift < 32)
      a = join(a, b);
  }
  return a;
}
template <int Pack, int Threads>
__global__ void reduce_blocks(const __nv_bfloat16 *x, Sum *parts, float *y,
                              int n) {
  int tid = threadIdx.x, base = (blockIdx.x * Threads + tid) * Pack;
  Sum acc = {0, 0};
  if constexpr (Pack == 8) {
    if (base + 8 <= n && uintptr_t(x) % 16 == 0) {
      B8 v = *reinterpret_cast<const B8 *>(x + base);
#pragma unroll
      for (int j = 0; j < 8; j++)
        add(acc, __uint_as_float(unsigned(v[j]) << 16));
    } else
      for (int j = 0; j < 8; j++)
        if (base + j < n)
          add(acc, __bfloat162float(x[base + j]));
  } else if constexpr (Pack == 2) {
    if (base + 2 <= n && uintptr_t(x) % 4 == 0) {
      B2 v = *reinterpret_cast<const B2 *>(x + base);
      add(acc, __uint_as_float(unsigned(v[0]) << 16));
      add(acc, __uint_as_float(unsigned(v[1]) << 16));
    } else
      for (int j = 0; j < 2; j++)
        if (base + j < n)
          add(acc, __bfloat162float(x[base + j]));
  } else if (base < n)
    add(acc, __bfloat162float(x[base]));
  __shared__ Sum leaders[Threads / 32];
  acc = wave(acc);
  if (tid % 32 == 0)
    leaders[tid / 32] = acc;
  __syncthreads();
  if (tid < 32) {
    acc = tid < Threads / 32 ? leaders[tid] : Sum{0, 0};
    acc = wave(acc);
    if (tid == 0) {
      if (gridDim.x == 1)
        y[0] = acc.main + acc.residual;
      else
        parts[blockIdx.x] = acc;
    }
  }
}
__global__ void finish(const Sum *parts, float *y, int n) {
  Sum acc = {0, 0};
  int tid = threadIdx.x;
  for (int i = tid; i < n; i += 256)
    acc = join(acc, parts[i]);
  __shared__ Sum leaders[8];
  acc = wave(acc);
  if (tid % 32 == 0)
    leaders[tid / 32] = acc;
  __syncthreads();
  if (tid < 32) {
    acc = tid < 8 ? leaders[tid] : Sum{0, 0};
    acc = wave(acc);
    if (tid == 0)
      y[0] = acc.main + acc.residual;
  }
}
template <int Pack, int Threads>
void launch(torch::Tensor x, torch::Tensor parts, torch::Tensor y, int blocks) {
  reduce_blocks<Pack, Threads><<<blocks, Threads>>>(
      reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
      reinterpret_cast<Sum *>(parts.data_ptr()), y.data_ptr<float>(),
      x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int Pack> torch::Tensor run(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
                  x.is_contiguous(),
              "expected contiguous BF16 GPU input");
  TORCH_CHECK(!x.requires_grad(), "forward only");
  TORCH_CHECK(x.numel() < INT32_MAX - 8192, "index limit");
  c10::cuda::CUDAGuard guard(x.device());
  auto opt = x.options().dtype(at::kFloat);
  auto y = torch::empty({1}, opt);
  if (!x.numel()) {
    y.zero_();
    return y;
  }
  int threads = 1024 / Pack;
  if (x.dim() == 2 && x.size(1) % Pack == 0) {
    int t = x.size(1) / Pack;
    if (t >= 32 && t <= 1024 && (t & (t - 1)) == 0)
      threads = t;
  }
  int blocks = (x.numel() + threads * Pack - 1) / (threads * Pack);
  auto parts = torch::empty({blocks > 1 ? blocks * 2 : 0}, opt);
  switch (threads) {
#define CASE(T)                                                                \
  case T:                                                                      \
    launch<Pack, T>(x, parts, y, blocks);                                      \
    break;
    CASE(32) CASE(64) CASE(128) CASE(256) CASE(512) CASE(1024)
#undef CASE
  }
  if (blocks > 1) {
    finish<<<1, 256>>>(reinterpret_cast<const Sum *>(parts.data_ptr()),
                       y.data_ptr<float>(), blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}
} // namespace
torch::Tensor block_all_reduce_sum_bf16_bf16(torch::Tensor x) {
  return run<1>(x);
}
torch::Tensor block_all_reduce_sum_bf16x2_bf16(torch::Tensor x) {
  return run<2>(x);
}
torch::Tensor block_all_reduce_sum_bf16x8_pack_bf16(torch::Tensor x) {
  return run<8>(x);
}
