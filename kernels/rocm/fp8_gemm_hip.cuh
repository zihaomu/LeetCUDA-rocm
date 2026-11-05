#pragma once
// ROCm port of interview/fp8_gemm.cuh. Quantization algorithms retain the
// original row/column/block granularity. GEMM uses native RDNA4 FP8 WMMA.
#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>

namespace fp8_gemm {
constexpr float kE4M3Max = 448.0f; // e4m3 最大有限值（饱和上界）
constexpr int kScaleBlock = 128;   // per-block 量化粒度（固定 128）

enum class Fp8GemmScaleMode {
  kPerRowPerCol,     // A per-row x B per-col（精度最优，scale 数 M+N）
  kPerRowPerBlock,   // A per-row x B per-block(scale 数 M + N/128)
  kPerBlockPerCol,   // A per-block x B per-col(scale 数 M/128 + N)
  kPerBlockPerBlock, // A per-block x B per-block(attention Q/K
                     // 同构，M/128+N/128)
};

union Vec8BF16 { // 8 x bf16 = 16B, 量化 kernel 的向量化 IO 单元
  uint4 raw;
  hip_bfloat16 elem[8];
};
static_assert(sizeof(Vec8BF16) == 16, "Vec8BF16 must be 128-bit");

// f32 -> e4m3 两条打包；HIP RNE/satfinite 保留原 CUDA 量化语义。
// satfinite 保证 x*inv_δ 舍入越界时饱和到 ±448 而非 NaN.
__device__ __forceinline__ unsigned short cvt_f2_to_e4m3x2(float lo, float hi) {
  return __hip_cvt_float2_to_fp8x2(make_float2(lo, hi), __HIP_SATFINITE,
                                   __HIP_E4M3);
}

// =============================================================================
// Phase 9.2: 量化前处理 kernel(A 行主序保形 / B 转置到 B8T[N,K])
// =============================================================================
// A[M,K] bf16 -> A8[M,K] e4m3 行主序；GEMM 按 (BM,BK) 搬入 LDS。
//
// per-row 版：warp-per-row, 一 warp 负责一整行。amax 经 shfl_xor 蝶形归约，
// 量化与写回同行完成。K 是运行时参数无法在寄存器缓存整行，gmem 读两遍
// （第二遍命中 L1/L2）；ffpa 的 per-block kernel 依赖编译期 kD 才能一遍读。
__global__ void quantize_a_perrow_kernel(const hip_bfloat16 *__restrict__ X,
                                         uint8_t *__restrict__ Y,
                                         float *__restrict__ scale, int M,
                                         int K) {
  constexpr int kVec = 8; // K % 16 == 0 保证 chunk 对齐
  const int row = blockIdx.x * 8 + threadIdx.x / 32; // 8 warps/block
  if (row >= M)
    return;
  const int lane = threadIdx.x % 32;
  const hip_bfloat16 *xr = X + (size_t)row * K;

  float amax = 0.0f; // pass 1: 行 amax
  for (int c = lane * kVec; c < K; c += 32 * kVec) {
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(xr + c);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      amax = fmaxf(amax, fabsf(float(v.elem[e])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) // warp 内蝶形归约
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffULL, amax, off));

  const float s = amax / kE4M3Max;
  const float inv_s = (amax == 0.0f) ? 0.0f : kE4M3Max / amax;
  if (lane == 0)
    scale[row] = s;

  uint8_t *yr = Y + (size_t)row * K; // pass 2: 量化写回
  for (int c = lane * kVec; c < K; c += 32 * kVec) {
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(xr + c);
    uint2 out;
    unsigned short *o2 = reinterpret_cast<unsigned short *>(&out);
#pragma unroll
    for (int e = 0; e < kVec; e += 2)
      o2[e / 2] = cvt_f2_to_e4m3x2(float(v.elem[e]) * inv_s,
                                   float(v.elem[e + 1]) * inv_s);
    *reinterpret_cast<uint2 *>(yr + c) = out; // 8 x e4m3 = 8B
  }
}

// A per-block 版：一个 block 负责连续 128 行(ffpa quantize_fp8_kernel 同构，
// 但 K 运行时化 -> 两遍 gmem 读，第二遍命中 L1/L2).block 级 amax =
// 128 行全体元素的最大绝对值，warp 归约 -> smem -> warp0 二次归约。
__global__ void quantize_a_perblock_kernel(const hip_bfloat16 *__restrict__ X,
                                           uint8_t *__restrict__ Y,
                                           float *__restrict__ scale, int M,
                                           int K) {
  constexpr int kVec = 8;
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  const int row0 = blockIdx.x * kScaleBlock;
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int dv = K / kVec; // 每行 Vec8 chunk 数

  float amax = 0.0f; // pass 1: 128 行块 amax
  for (int i = tid; i < kScaleBlock * dv; i += kThreads) {
    const int r = i / dv, c = (i % dv) * kVec;
    const long row = row0 + r;
    if (row >= M)
      continue;
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(X + row * K + c);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      amax = fmaxf(amax, fabsf(float(v.elem[e])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffULL, amax, off));
  __shared__ float warp_max[kWarps];
  if (lane == 0)
    warp_max[warp] = amax;
  __syncthreads();
  if (warp == 0) {
    amax = (lane < kWarps) ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int off = kWarps / 2; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffULL, amax, off));
    if (lane == 0)
      warp_max[0] = amax;
  }
  __syncthreads();
  amax = warp_max[0];

  const float inv_s = (amax == 0.0f) ? 0.0f : kE4M3Max / amax;
  if (tid == 0)
    scale[blockIdx.x] = amax / kE4M3Max;

  for (int i = tid; i < kScaleBlock * dv; i += kThreads) { // pass 2: 量化
    const int r = i / dv, c = (i % dv) * kVec;
    const long row = row0 + r;
    if (row >= M)
      continue;
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(X + row * K + c);
    uint2 out;
    unsigned short *o2 = reinterpret_cast<unsigned short *>(&out);
#pragma unroll
    for (int e = 0; e < kVec; e += 2)
      o2[e / 2] = cvt_f2_to_e4m3x2(float(v.elem[e]) * inv_s,
                                   float(v.elem[e + 1]) * inv_s);
    *reinterpret_cast<uint2 *>(Y + row * K + c) = out;
  }
}

// B 转置量化：B[K,N] bf16 -> B8T[N,K] e4m3(B^T 行主序，K 内维连续，
// GEMM 按 (BN,BK) 搬入 LDS).一个 block 负责连续 128
// 个 n（即 B8T 的 128 行），K 方向按 64 行 chunk 循环两遍：
//   pass 1: coalesced 读 B 的 [64,128] tile 进 smem, 逐列(n)归约 amax,
//           跨全部 K chunk 累积到 samax[128]
//   pass 2: 重读 tile -> 量化散写到输出朝向 tile_t[n][k] -> warp 串行 n 行，
//           32 lanes 写 32 个连续 k 字节(coalesced 32B store)
// 为什么不直接写 B8T? B 的列方向 stride=N, 直接按列读/写都不 coalesced;
// smem staging 让 gmem 读(B 行)与 gmem 写(B8T 行)都落在连续维度上。
// 越界 load 显式补零；B8T 无需 pad 列，K 尾部由 store guard 裁剪。
// kPerCol: scale[N](B 每列一个，B8T 行主序下即每行一个); 否则 scale[N/128].
template <bool kPerCol>
__global__ void quantize_bt_kernel(const hip_bfloat16 *__restrict__ B,
                                   uint8_t *__restrict__ BT,
                                   float *__restrict__ scale, int K, int N) {
  constexpr int kBNt = 128; // n 方向 tile(= B8T 行数)
  constexpr int kBKt = 64;  // k 方向 chunk(= B 行数)
  constexpr int kVec = 8;
  constexpr int kChunksPerRow = kBNt / kVec; // 16
  constexpr int kChunks = kBKt * kChunksPerRow;
  constexpr int kWarps = 256 / 32;
  const int n0 = blockIdx.x * kBNt;
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;

  __shared__ __align__(16) hip_bfloat16 tile[kBKt][kBNt]; // 源朝向 16KB
  __shared__ uint8_t tile_t[kBNt][kBKt]; // 输出朝向 8KB（平凡类型，
  // 避免 smem 变量动态初始化告警；与 uint8_t 逐位兼容)
  __shared__ float samax[kBNt];

  auto load_tile = [&](int k0) { // B 行 coalesced 读入 smem（越界补零）
    for (int i = tid; i < kChunks; i += 256) {
      const int r = i / kChunksPerRow, c8 = (i % kChunksPerRow) * kVec;
      const int k = k0 + r, n = n0 + c8;
      Vec8BF16 v;
      v.raw = make_uint4(0, 0, 0, 0);
      if (k < K && n < N)
        v = *reinterpret_cast<const Vec8BF16 *>(B + (size_t)k * N + n);
      *reinterpret_cast<uint4 *>(&tile[r][c8]) = v.raw;
    }
  };

  for (int n = tid; n < kBNt; n += 256)
    samax[n] = 0.0f; // tid<128 生效
  __syncthreads();
  for (int k0 = 0; k0 < K; k0 += kBKt) { // pass 1: 逐列 amax
    load_tile(k0);
    __syncthreads();
    for (int n = tid; n < kBNt; n += 256) {
      float a = samax[n];
      for (int r = 0; r < kBKt; ++r)
        a = fmaxf(a, fabsf(float(tile[r][n])));
      samax[n] = a;
    }
    __syncthreads();
  }

  // scale 写出 + samax 原地改存 inv_s(pass 2 直接查表)
  if constexpr (kPerCol) {
    for (int n = tid; n < kBNt; n += 256) {
      if (n0 + n >= N)
        continue;
      const float am = samax[n];
      scale[n0 + n] = am / kE4M3Max;
      samax[n] = (am == 0.0f) ? 0.0f : kE4M3Max / am;
    }
  } else {
    __syncthreads();
    if (tid == 0) { // block 级 amax 归约(128 个列 amax -> 单值)
      float m = 0.0f;
      for (int n = 0; n < kBNt; ++n)
        m = fmaxf(m, samax[n]);
      if (n0 < N)
        scale[n0 / kScaleBlock] = m / kE4M3Max;
      samax[0] = (m == 0.0f) ? 0.0f : kE4M3Max / m;
    }
  }
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += kBKt) { // pass 2: 量化 + 转置写
    load_tile(k0);
    __syncthreads();
    for (int i = tid; i < kChunks; i += 256) { // 散写 tile_t[n][r]
      const int r = i / kChunksPerRow, c8 = (i % kChunksPerRow) * kVec;
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        const int n = c8 + e;
        const float inv_s = kPerCol ? samax[n] : samax[0];
        tile_t[n][r] = __hip_cvt_float_to_fp8(float(tile[r][n]) * inv_s,
                                              __HIP_SATFINITE, __HIP_E4M3);
      }
    }
    __syncthreads();
    for (int n = warp; n < kBNt; n += kWarps) { // 32B coalesced store
      if (n0 + n >= N)
        continue;
      uint8_t *row = BT + (size_t)(n0 + n) * K + k0;
#pragma unroll
      for (int half = 0; half < kBKt / 32; ++half) {
        const int kk = half * 32 + lane;
        if (k0 + kk < K)
          row[kk] = *reinterpret_cast<uint8_t *>(&tile_t[n][kk]);
      }
    }
    __syncthreads(); // tile/tile_t 下一 chunk 可覆写
  }
}

// B 侧量化入口：全链路(fp8_gemm_bf16)与权重离线量化(Phase 9.8)共用同一实现。
inline void fp8_gemm_quantize_b(const hip_bfloat16 *B, uint8_t *b8t, float *sb,
                                int N, int K, bool per_col_b,
                                hipStream_t stream) {
  if (per_col_b)
    quantize_bt_kernel<true>
        <<<(N + 127) / 128, 256, 0, stream>>>(B, b8t, sb, K, N);
  else
    quantize_bt_kernel<false>
        <<<(N + 127) / 128, 256, 0, stream>>>(B, b8t, sb, K, N);
}

// AMD defaults use 64x64x64 tiles. Other supported Traits retain their actual
// tile sizes and stage counts; unsupported resource requests fail at compile
// time.
template <int BM = 64, int BN = 64, int BK = 64, int STAGES = 2>
struct Fp8GemmTraits {
  static_assert(BM % 32 == 0 && BN % 32 == 0 && BK % 64 == 0);
  static_assert(STAGES >= 2 && STAGES <= 4);
  static constexpr int kBM = BM, kBN = BN, kBK = BK, kStages = STAGES;
  static constexpr int kNumWarps = (BM / 32) * (BN / 32);
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kSmemBytes = STAGES * (BM + BN) * BK;
  static_assert(kNumThreads <= 1024 && kSmemBytes <= 65536);
};

using Fp8Fragment = int __attribute__((ext_vector_type(2)));
using AccFragment = float __attribute__((ext_vector_type(8)));

template <int BK> __device__ __forceinline__ int lds_index(int row, int col) {
  return row * BK + (col ^ ((row & 3) * 16));
}

template <typename T>
__device__ void load_gemm_tile(const uint8_t *a, const uint8_t *bt,
                               uint8_t *tile, int m, int n, int k, int kbase,
                               int tid, int threads) {
  for (int pack = tid; pack < (T::kBM + T::kBN) * T::kBK / 16;
       pack += threads) {
    const int row = pack / (T::kBK / 16);
    const int col = (pack % (T::kBK / 16)) * 16;
    const bool from_a = row < T::kBM;
    const int local_row = from_a ? row : row - T::kBM;
    const int global_row =
        local_row + (from_a ? blockIdx.y * T::kBM : blockIdx.x * T::kBN);
    const uint8_t *input = from_a ? a : bt;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (global_row < (from_a ? m : n) && kbase + col < k)
      value = *reinterpret_cast<const uint4 *>(input + size_t(global_row) * k +
                                               kbase + col);
    uint8_t *destination = tile + (from_a ? 0 : T::kBM * T::kBK);
    *reinterpret_cast<uint4 *>(destination +
                               lds_index<T::kBK>(local_row, col)) = value;
  }
}

template <typename T>
__device__ void compute_gemm_tile(const uint8_t *tile,
                                  AccFragment (&acc)[2][2]) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int wm = warp / (T::kBN / 32) * 32;
  const int wn = warp % (T::kBN / 32) * 32;
#pragma unroll
  for (int kk = 0; kk < T::kBK; kk += 16) {
    Fp8Fragment af[2], bf[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int ar = wm + 16 * i + lane % 16;
      const int br = wn + 16 * i + lane % 16;
      const int col = kk + lane / 16 * 8;
      af[i] = *reinterpret_cast<const Fp8Fragment *>(
          tile + lds_index<T::kBK>(ar, col));
      bf[i] = *reinterpret_cast<const Fp8Fragment *>(
          tile + T::kBM * T::kBK + lds_index<T::kBK>(br, col));
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            af[i], bf[j], acc[i][j]);
  }
}

template <bool PER_ROW, bool PER_COL, bool WS, typename T>
__global__ void fp8_gemm_amd_kernel(const uint8_t *a, const uint8_t *bt,
                                    const float *sa, const float *sb,
                                    hip_bfloat16 *c, int m, int n, int k) {
  __shared__ __align__(16) uint8_t ring[T::kSmemBytes];
  constexpr int tile_bytes = (T::kBM + T::kBN) * T::kBK;
  const int tid = threadIdx.x;
  const bool consumer = !WS || tid < T::kNumThreads;
  const bool producer = !WS || !consumer;
  const int load_tid = WS ? tid - T::kNumThreads : tid;
  const int load_threads = WS ? 128 : T::kNumThreads;
  const int tiles = (k + T::kBK - 1) / T::kBK;
  constexpr int ahead = WS ? T::kStages - 1 : T::kStages;
  if (producer)
    for (int t = 0; t < ahead && t < tiles; ++t)
      load_gemm_tile<T>(a, bt, ring + t * tile_bytes, m, n, k, t * T::kBK,
                        load_tid, load_threads);
  __syncthreads();
  AccFragment acc[2][2] = {};
  for (int t = 0; t < tiles; ++t) {
    if (consumer)
      compute_gemm_tile<T>(ring + (t % T::kStages) * tile_bytes, acc);
    if constexpr (!WS)
      __syncthreads(); // All consumers finish before their tile is reused.
    const int next = t + ahead;
    if (producer && next < tiles)
      load_gemm_tile<T>(a, bt, ring + (next % T::kStages) * tile_bytes, m, n, k,
                        next * T::kBK, load_tid, load_threads);
    __syncthreads(); // Producer loads and consumer reads finish before
                     // advancing.
  }
  if (consumer) {
    const int warp = tid / 32, lane = tid % 32;
    const int wm = warp / (T::kBN / 32) * 32;
    const int wn = warp % (T::kBN / 32) * 32;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int row = blockIdx.y * T::kBM + wm + i * 16 + lane / 16 * 8 + v;
          const int col = blockIdx.x * T::kBN + wn + j * 16 + lane % 16;
          if (row < m && col < n)
            c[size_t(row) * n + col] =
                hip_bfloat16(acc[i][j][v] * sa[PER_ROW ? row : row / 128] *
                             sb[PER_COL ? col : col / 128]);
        }
  }
}

inline void check_fp8_launch() {
  auto error = hipGetLastError();
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
inline void check_fp8_shape(int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 16 || n % 8 ||
      int64_t(m) * k > INT32_MAX || int64_t(n) * k > INT32_MAX ||
      int64_t(m) * n > INT32_MAX || m > INT32_MAX - 256 ||
      n > INT32_MAX - 256 || k > INT32_MAX - 256)
    throw std::invalid_argument(
        "FP8 GEMM requires positive 32-bit shapes, K%16=0, N%8=0");
}

template <bool PER_ROW, bool PER_COL, bool WS, typename T = Fp8GemmTraits<>>
inline void fp8_gemm_tma_fwd(const uint8_t *a8, const uint8_t *b8t,
                             const float *sa, const float *sb, hip_bfloat16 *c,
                             int m, int n, int k, hipStream_t stream) {
  // Compatibility name; AMD uses LDS ring buffers, not NVIDIA TMA.
  static_assert(!WS || T::kNumThreads + 128 <= 1024);
  check_fp8_shape(m, n, k);
  const dim3 grid((n + T::kBN - 1) / T::kBN, (m + T::kBM - 1) / T::kBM);
  fp8_gemm_amd_kernel<PER_ROW, PER_COL, WS, T>
      <<<grid, T::kNumThreads + (WS ? 128 : 0), 0, stream>>>(a8, b8t, sa, sb, c,
                                                             m, n, k);
  check_fp8_launch();
}
struct Fp8GemmWorkspace {
  void *buf = nullptr;
  size_t size = 0;
};
struct Fp8GemmActivation {
  void *buf = nullptr;
  size_t size = 0;
};
inline size_t fp8_gemm_align256(size_t x) { return (x + 255) & ~size_t(255); }
inline size_t fp8_gemm_workspace_size(int m, int n, int k) {
  check_fp8_shape(m, n, k);
  return fp8_gemm_align256(size_t(m) * k) + fp8_gemm_align256(size_t(n) * k) +
         fp8_gemm_align256(size_t(m) * sizeof(float)) +
         fp8_gemm_align256(size_t(n) * sizeof(float));
}
inline size_t fp8_gemm_activation_size(int m, int k) {
  check_fp8_shape(m, 8, k);
  return fp8_gemm_align256(size_t(m) * k) +
         fp8_gemm_align256(size_t(m) * sizeof(float));
}
inline void quantize_a(const hip_bfloat16 *a, uint8_t *a8, float *sa, int m,
                       int k, bool per_row, hipStream_t stream) {
  if (per_row) {
    quantize_a_perrow_kernel<<<(m + 7) / 8, 256, 0, stream>>>(a, a8, sa, m, k);
  } else {
    quantize_a_perblock_kernel<<<(m + 127) / 128, 256, 0, stream>>>(a, a8, sa,
                                                                    m, k);
  }
  check_fp8_launch();
}
inline void dispatch_gemm(const uint8_t *a8, const uint8_t *b8t,
                          const float *sa, const float *sb, hip_bfloat16 *c,
                          int m, int n, int k, bool per_row, bool per_col,
                          bool ws, hipStream_t stream) {
#define AMD_FP8_DISPATCH(R, C)                                                 \
  if (ws)                                                                      \
    fp8_gemm_tma_fwd<R, C, true>(a8, b8t, sa, sb, c, m, n, k, stream);         \
  else                                                                         \
    fp8_gemm_tma_fwd<R, C, false>(a8, b8t, sa, sb, c, m, n, k, stream)
  if (per_row && per_col) {
    AMD_FP8_DISPATCH(true, true);
  } else if (per_row) {
    AMD_FP8_DISPATCH(true, false);
  } else if (per_col) {
    AMD_FP8_DISPATCH(false, true);
  } else {
    AMD_FP8_DISPATCH(false, false);
  }
#undef AMD_FP8_DISPATCH
}
inline void fp8_gemm_bf16(const hip_bfloat16 *a, const hip_bfloat16 *b,
                          hip_bfloat16 *c, int m, int n, int k,
                          Fp8GemmScaleMode mode, Fp8GemmWorkspace &ws,
                          hipStream_t stream, bool use_ws = false) {
  const size_t bytes = fp8_gemm_workspace_size(m, n, k);
  if (!ws.buf || ws.size < bytes || uintptr_t(ws.buf) % 256)
    throw std::invalid_argument(
        "FP8 workspace is missing, too small, or not 256-byte aligned");
  if (int(mode) < 0 || int(mode) > 3)
    throw std::invalid_argument("invalid FP8 scale mode");
  auto *a8 = static_cast<uint8_t *>(ws.buf);
  auto *b8t = a8 + fp8_gemm_align256(size_t(m) * k);
  auto *sa = reinterpret_cast<float *>(b8t + fp8_gemm_align256(size_t(n) * k));
  auto *sb =
      reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(sa) +
                                fp8_gemm_align256(size_t(m) * sizeof(float)));
  const bool row = mode == Fp8GemmScaleMode::kPerRowPerCol ||
                   mode == Fp8GemmScaleMode::kPerRowPerBlock;
  const bool col = mode == Fp8GemmScaleMode::kPerRowPerCol ||
                   mode == Fp8GemmScaleMode::kPerBlockPerCol;
  quantize_a(a, a8, sa, m, k, row, stream);
  fp8_gemm_quantize_b(b, b8t, sb, n, k, col, stream);
  check_fp8_launch();
  dispatch_gemm(a8, b8t, sa, sb, c, m, n, k, row, col, use_ws, stream);
}

template <bool ROW, bool COL, bool WS = false, typename T = Fp8GemmTraits<>>
inline void fp8_gemm_bf16_b_offline(const hip_bfloat16 *a, const uint8_t *b8t,
                                    const float *sb, hip_bfloat16 *c, int m,
                                    int n, int k, Fp8GemmActivation &act,
                                    hipStream_t stream) {
  check_fp8_shape(m, n, k);
  if (!act.buf || act.size < fp8_gemm_activation_size(m, k) ||
      uintptr_t(act.buf) % 256)
    throw std::invalid_argument(
        "FP8 activation workspace is missing, too small, or unaligned");
  auto *a8 = static_cast<uint8_t *>(act.buf);
  auto *sa = reinterpret_cast<float *>(a8 + fp8_gemm_align256(size_t(m) * k));
  quantize_a(a, a8, sa, m, k, ROW, stream);
  fp8_gemm_tma_fwd<ROW, COL, WS, T>(a8, b8t, sa, sb, c, m, n, k, stream);
}
} // namespace fp8_gemm
