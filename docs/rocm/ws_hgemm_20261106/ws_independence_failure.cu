// gfx1201 warp-specialized GEMM: one producer wave, four consumer waves.
// Three reusable LDS slots, independent acquire/release sequence counters.
// Explicit precision upgrade: FP16 inputs/output, FP32 WMMA accumulation.
// B has logical [K,N] shape but stores contiguous [N,K], as upstream.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
namespace {
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
constexpr int BM = 64, BN = 128, BK = 32, Stages = 3, Slot = (BM + BN) * BK;
__device__ __forceinline__ int index(int row, int k) {
  return row * BK + (k ^ ((row & 3) * 8));
}
template <bool Audit>
__device__ __forceinline__ bool wait_flag(unsigned *p, unsigned target,
                                          unsigned *stats) {
  unsigned spins = 0;
  while (__hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) <
         target) {
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
__device__ __forceinline__ void release(unsigned *p, unsigned value) {
  __threadfence_block();
  __syncwarp();
  if (threadIdx.x % 32 == 0)
    __hip_atomic_store(p, value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_WORKGROUP);
}
__device__ __forceinline__ void copy_tile(const _Float16 *a, const _Float16 *b,
                                          _Float16 *lds, int m, int n, int k,
                                          int bm, int bn, int t, int slot) {
  int lane = threadIdx.x % 32;
  for (int mat = 0; mat < 2; mat++) {
    int height = mat ? BN : BM;
    const _Float16 *p = mat ? b : a;
    int origin = mat ? bn : bm, total = mat ? n : m;
    for (int v = lane * 8; v < height * BK; v += 32 * 8) {
      int row = origin + v / BK, col = t * BK + v % BK;
      H8 values = {};
      if (row < total && col + 8 <= k && k % 8 == 0 && uintptr_t(p) % 16 == 0)
        values = *reinterpret_cast<const H8 *>(p + row * k + col);
      else
        for (int x = 0; x < 8; x++)
          if (row < total && col + x < k)
            values[x] = p[row * k + col + x];
      for (int x = 0; x < 8; x++)
        lds[slot * Slot + (mat ? BM * BK : 0) + index(v / BK, v % BK + x)] =
            values[x];
    }
  }
}
__device__ __forceinline__ void consume(const _Float16 *cur, F8 (&acc)[2][4],
                                        int wave) {
  int lane = threadIdx.x % 32, wr = wave / 2, wc = wave % 2;

#pragma unroll
  for (int kk = 0; kk < 2; kk++) {
    H8 af[2], bf[4];
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int x = 0; x < 8; x++)
        af[i][x] = cur[index((wr * 2 + i) * 16 + lane % 16,
                             kk * 16 + lane / 16 * 8 + x)];
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int x = 0; x < 8; x++)
        bf[j][x] = cur[BM * BK + index((wc * 4 + j) * 16 + lane % 16,
                                       kk * 16 + lane / 16 * 8 + x)];
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < 4; j++)
        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            af[i], bf[j], acc[i][j]);
    }
  }
}
template <bool Synchronous, bool Audit, int DelayRole>
__global__ void __launch_bounds__(160)
    ws_pipeline(const _Float16 *a, const _Float16 *b, _Float16 *c, int m, int n,
                int k, unsigned *metrics) {
  __shared__ __align__(16) _Float16 lds[Stages * Slot];
  __shared__ unsigned ready[Stages], done[Stages][4];
  int tid = threadIdx.x, lane = tid % 32;
  unsigned *stats = nullptr;
  if constexpr (Audit)
    stats = metrics + (blockIdx.y * gridDim.x + blockIdx.x) * 8;
  if (tid < Stages)
    ready[tid] = 0;
  if (tid < Stages * 4)
    done[tid / 4][tid % 4] = 0;
  __syncthreads();
  int bm = blockIdx.y * BM, bn = blockIdx.x * BN, tiles = (k + BK - 1) / BK;
  int wave = (tid - 32) / 32, wr = wave / 2, wc = wave % 2;
  F8 acc[2][4] = {};
  if constexpr (Synchronous) {
    for (int t = 0; t < tiles; t++) {
      int slot = t % Stages;
      if (tid < 32)
        copy_tile(a, b, lds, m, n, k, bm, bn, t, slot);
      __syncthreads();
      if (tid >= 32)
        consume(lds + slot * Slot, acc, wave);
      __syncthreads();
    }
  } else if (tid < 32) {
    for (int t = 0; t < tiles; t++) {
      int slot = t % Stages;
      if (t >= Stages)
        for (int w = 0; w < 4; w++)
          if (!wait_flag<Audit>(&done[slot][w], t - Stages + 1, stats))
            return;
      if constexpr (DelayRole == 1)
        for (int d = 0; d < 64; d++)
          __builtin_amdgcn_s_sleep(1);
      copy_tile(a, b, lds, m, n, k, bm, bn, t, slot);
      if constexpr (Audit)
        if (lane == 0) {
          unsigned slowest = t;
          for (int w = 0; w < 4; w++) {
            unsigned latest = 0;
            for (int s = 0; s < Stages; s++)
              latest =
                  max(latest, __hip_atomic_load(&done[s][w], __ATOMIC_ACQUIRE,
                                                __HIP_MEMORY_SCOPE_WORKGROUP));
            slowest = min(slowest, latest);
          }
          stats[0] = t + 1;
          stats[5] = max(stats[5], unsigned(t + 1) - slowest);
        }
      release(&ready[slot], t + 1);
    }
  } else {
    for (int t = 0; t < tiles; t++) {
      int slot = t % Stages;
      if (!wait_flag<Audit>(&ready[slot], t + 1, stats))
        return;
      if constexpr (DelayRole == 2)
        if (wave == 0)
          for (int d = 0; d < 64; d++)
            __builtin_amdgcn_s_sleep(1);
      consume(lds + slot * Slot, acc, wave);
      if constexpr (Audit)
        if (lane == 0)
          stats[1 + wave] = t + 1;
      release(&done[slot][wave], t + 1);
    }
  }
  if (tid >= 32) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int col = bn + (wc * 4 + j) * 16 + lane % 16;
#pragma unroll
        for (int x = 0; x < 8; x++) {
          int row = bm + (wr * 2 + i) * 16 + lane / 16 * 8 + x;
          if (row < m && col < n)
            c[row * n + col] = (_Float16)acc[i][j][x];
        }
      }
    }
  }
}
void check(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() &&
                  c.device() == a.device(),
              "expected one GPU");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf &&
                  c.scalar_type() == at::kHalf,
              "expected FP16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2 &&
                  a.size(1) == b.size(0) && a.size(0) == c.size(0) &&
                  b.size(1) == c.size(1),
              "expected A[M,K], B[K,N], C[M,N]; TN B stores packed [N,K]");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "expected contiguous tensors");
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b), "output aliases input");
  TORCH_CHECK(!a.requires_grad() && !b.requires_grad() && !c.requires_grad(),
              "forward only");
  TORCH_CHECK(a.numel() < INT32_MAX && b.numel() < INT32_MAX &&
                  c.numel() < INT32_MAX && a.size(0) < INT32_MAX - 256 &&
                  a.size(1) < INT32_MAX - 256 && b.size(1) < INT32_MAX - 256,
              "32-bit index range exceeded");
}

template <bool Synchronous, bool Audit, int DelayRole>
void dispatch(torch::Tensor a, torch::Tensor b, torch::Tensor c,
              unsigned *metrics) {
  check(a, b, c);
  if (!c.numel())
    return;
  int m = a.size(0), n = b.size(1), k = a.size(1);
  TORCH_CHECK((m + BM - 1) / BM <= 65535, "grid limit exceeded");
  c10::cuda::CUDAGuard guard(a.device());
  dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  ws_pipeline<Synchronous, Audit, DelayRole><<<grid, 160>>>(
      reinterpret_cast<const _Float16 *>(a.data_ptr()),
      reinterpret_cast<const _Float16 *>(b.data_ptr()),
      reinterpret_cast<_Float16 *>(c.data_ptr()), m, n, k, metrics);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
void ws_hgemm_naive_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  dispatch<false, false, 0>(a, b, c, nullptr);
}
#ifdef LEETCUDA_WS_CONTROL
void ws_synchronous_control(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  dispatch<true, false, 0>(a, b, c, nullptr);
}
void ws_audit(torch::Tensor a, torch::Tensor b, torch::Tensor c,
              torch::Tensor metrics, int delay) {
  check(a, b, c);
  TORCH_CHECK(metrics.device() == a.device() &&
                  metrics.scalar_type() == at::kInt &&
                  metrics.is_contiguous() && !metrics.is_alias_of(a) &&
                  !metrics.is_alias_of(b) && !metrics.is_alias_of(c),
              "invalid metrics");
  TORCH_CHECK(metrics.numel() ==
                  ((a.size(0) + BM - 1) / BM) * ((b.size(1) + BN - 1) / BN) * 8,
              "metrics size");
  auto *p = reinterpret_cast<unsigned *>(metrics.data_ptr());
  if (delay == 1)
    dispatch<false, true, 1>(a, b, c, p);
  else if (delay == 2)
    dispatch<false, true, 2>(a, b, c, p);
  else
    dispatch<false, true, 0>(a, b, c, p);
}
#endif
