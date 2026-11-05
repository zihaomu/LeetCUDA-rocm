#pragma once
// NVFP4 storage/quantization, exact E2M1*E4M3 expansion, FP16 AMD WMMA.
// This is not native FP4 matrix arithmetic. Scale bytes retain the CUDA layout.
#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <stdexcept>

namespace fp4_gemm {
constexpr float kE2M1Max = 6.f, kUe4m3Max = 448.f, kFp4ScaleMax = 2688.f;
constexpr int kSFVec = 16, kRowBlock = 128, kSFVecPerBlk = 4;
enum class Fp4GemmScaleMode { kSingleLevel, kLevel1A, kLevel1B, kTwoLevel };
union Vec8BF16 {
  uint4 raw;
  hip_bfloat16 elem[8];
};
static_assert(sizeof(Vec8BF16) == 16);

__host__ __device__ constexpr int sf_byte_offset(int mn, int s, int nb_k) {
  return (mn / 128) * nb_k * 512 + (s / 4) * 512 + (mn % 32) * 16 +
         ((mn / 32) % 4) * 4 + s % 4;
}
__host__ __device__ constexpr size_t sf_buffer_bytes(int mn, int k) {
  return size_t((mn + 127) / 128) * (k / 64) * 512;
}
inline void check_hip(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
inline void check_fp4_shape(int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 64 || n % 8 || m > INT32_MAX - 256 ||
      n > INT32_MAX - 256 || int64_t(m) * k > INT32_MAX ||
      int64_t(n) * k > INT32_MAX || int64_t(m) * n > INT32_MAX ||
      sf_buffer_bytes(m, k) > INT32_MAX - 256 ||
      sf_buffer_bytes(n, k) > INT32_MAX - 256)
    throw std::invalid_argument("FP4 requires positive 32-bit shapes, K%64=0, "
                                "N%8=0 and 32-bit SF indexing");
}
__device__ __forceinline__ uint8_t encode_e2m1(float x) {
  const float a = fabsf(x);
  const unsigned magnitude = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) +
                             (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) +
                             (a > 5.f);
  return uint8_t(magnitude | ((__float_as_uint(x) >> 28) & 8));
}
__device__ __forceinline__ uint8_t cvt_f2_to_e2m1x2(float hi, float lo) {
  return uint8_t((encode_e2m1(hi) << 4) | encode_e2m1(lo));
}
__device__ __forceinline__ uint8_t cvt_f2_to_ue4m3(float x) {
  return __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3);
}
__device__ __forceinline__ float cvt_ue4m3_to_f32(uint8_t code) {
  const unsigned exponent = code >> 3, mantissa = code & 7;
  return exponent ? __uint_as_float(((exponent + 120) << 23) | (mantissa << 20))
                  : float(mantissa) * 0x1p-9f;
}
__device__ __forceinline__ float decode_e2m1(unsigned code) {
  const unsigned m = code & 7;
  const float value = m == 0   ? 0.f
                      : m == 1 ? 0.5f
                      : m == 2 ? 1.f
                      : m == 3 ? 1.5f
                      : m == 4 ? 2.f
                      : m == 5 ? 3.f
                      : m == 6 ? 4.f
                               : 6.f;
  return __uint_as_float(__float_as_uint(value) | ((code & 8) << 28));
}

__global__ void row_amax_fp4_kernel(const hip_bfloat16 *__restrict__ A,
                                    float *__restrict__ sa, int M, int K) {
  constexpr int kVec = 8;
  const int row = blockIdx.x * 8 + threadIdx.x / 32; // 8 warps/block
  if (row >= M)
    return;
  const int lane = threadIdx.x % 32;
  const hip_bfloat16 *ar = A + (size_t)row * K;
  float amax = 0.0f;
  for (int c = lane * kVec; c < K; c += 32 * kVec) {
    Vec8BF16 v = *reinterpret_cast<const Vec8BF16 *>(ar + c);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      amax = fmaxf(amax, fabsf(float(v.elem[e])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffULL, amax, off));
  if (lane == 0)
    sa[row] = amax / kFp4ScaleMax; // 全零行 -> 0（epilogue 相乘即 0）
}

// A[M,K] bf16 -> A4 packed(M,K) + SFA（含 128 行向上取整的补零行）
// kLevel1: 是否先按 sA[m] 归一化（false = 纯 level-2）
template <bool kLevel1>
__global__ void
quantize_a_fp4_kernel(const hip_bfloat16 *__restrict__ A,
                      uint8_t *__restrict__ a4, uint8_t *__restrict__ sfa,
                      const float *__restrict__ sa, int M, int K) {
  const int groups_per_row = K / kSFVec;
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int mpad = (M + kRowBlock - 1) / kRowBlock * kRowBlock;
  if (g >= mpad * groups_per_row)
    return;
  const int m = g / groups_per_row; // 行（可能落在补零区）
  const int grp = g % groups_per_row;
  const int k0 = grp * kSFVec;
  const int nb_k = K / 64;

  if (m >= M) { // 补零行：SF 显式写 0。TMA 只做 OOB 零填充，缓冲区里的
    // 未初始化字节会被当作 SF 乘进 MMA（0 x NaN = NaN），必须清零
    sfa[sf_byte_offset(m, grp, nb_k)] = 0;
    return;
  }

  float inv1 = 1.0f;
  if constexpr (kLevel1) { // 行级归一化：x' = x / sA[m]，sA[m] = rowmax/2688
    const float s = sa[m];
    inv1 = (s > 0.0f) ? (1.0f / s) : 0.0f;
  }

  const hip_bfloat16 *ar = A + (size_t)m * K + k0;
  Vec8BF16 v0 = *reinterpret_cast<const Vec8BF16 *>(ar);
  Vec8BF16 v1 = *reinterpret_cast<const Vec8BF16 *>(ar + 8);
  float x[kSFVec];
  float amax = 0.0f;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float a = float(v0.elem[e]) * inv1;
    const float b = float(v1.elem[e]) * inv1;
    x[e] = a;
    x[e + 8] = b;
    amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
  }

  const unsigned char sf_byte = cvt_f2_to_ue4m3(amax / kE2M1Max);
  const float sf_val = cvt_ue4m3_to_f32(sf_byte);
  // SF 下溢到 0（amax < 6 * 2^-10）-> 整组归零（inv2 = 0），避免 0/0 = NaN
  const float inv2 = (sf_val > 0.0f) ? (1.0f / sf_val) : 0.0f;
  sfa[sf_byte_offset(m, grp, nb_k)] = sf_byte;

  uint2 out;
  unsigned char *o = reinterpret_cast<unsigned char *>(&out);
#pragma unroll
  for (int e = 0; e < 8; ++e) { // 8 个字节 = 16 个 4-bit
    o[e] = cvt_f2_to_e2m1x2(x[2 * e + 1] * inv2, x[2 * e] * inv2);
  }
  *reinterpret_cast<uint2 *>(reinterpret_cast<unsigned char *>(a4) +
                             (size_t)m * (K / 2) + grp * 8) = out;
}

// B: preserve the original split-K column amax and independent
// transpose/quantize tiles. 列 amax（level-1 用）：grid = (ceil(N/128),
// kSplit)，块内 16 条 k-lane x 16 组（每组 8 列）共 256 线程。每线程持 8 列、K
// 方向步进 16，一次 16B 载入 拿满 8 列；warp 内 2 行 x 128 列 = 4 条 128B
// 事务，正好吃满。
__global__ void col_amax_fp4_kernel(const hip_bfloat16 *__restrict__ B,
                                    float *__restrict__ sb, int N, int K) {
  constexpr int kVec = 8;               // 每线程 8 列 = 一次 16B 载入
  constexpr int kKLanes = 16;           // 块内 k 方向 lane 数
  constexpr int kBNt = 128;             // 每块 128 列
  constexpr int kNPerRow = kBNt / kVec; // 16
  const int n0 = blockIdx.x * kBNt;
  const int kBeg = (int)((long long)K * blockIdx.y / gridDim.y);
  const int kEnd = (int)((long long)K * (blockIdx.y + 1) / gridDim.y);
  const int tid = threadIdx.x;
  const int nl = (tid % kNPerRow) * kVec; // 列起点 0..120
  const int kl = tid / kNPerRow;          // k lane 0..15
  __shared__ float part[kKLanes][kBNt];

  float a[kVec];
#pragma unroll
  for (int e = 0; e < kVec; ++e)
    a[e] = 0.0f;
  if (n0 + nl < N) {
    for (int k = kBeg + kl; k < kEnd; k += kKLanes) {
      Vec8BF16 v =
          *reinterpret_cast<const Vec8BF16 *>(B + (size_t)k * N + n0 + nl);
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        a[e] = fmaxf(a[e], fabsf(float(v.elem[e])));
    }
  }
#pragma unroll
  for (int e = 0; e < kVec; ++e)
    part[kl][nl + e] = a[e];
  __syncthreads();
  if (tid < kBNt && n0 + tid < N) {
    float m = 0.0f;
#pragma unroll
    for (int r = 0; r < kKLanes; ++r)
      m = fmaxf(m, part[r][tid]);
    // 先除以 2688 再 atomicMax：除以正常数保序，所以「先除后取 max」与旧版
    // 「先取 max 再除」逐位相同。amax >= 0，IEEE754 非负数的位模式与无符号
    // 整数序一致，可直接用整数 atomicMax（调用方需先把 sb 清零）。
    atomicMax(reinterpret_cast<unsigned *>(sb + n0 + tid),
              __float_as_uint(m / kFp4ScaleMax));
  }
}

// B 转置量化：B[K,N] bf16 -> B4T packed(N,K) + SFB。
// grid = (ceil(N/128), K/64)：一个 block 只做一个 128(n) x 64(k) 的 tile——
// 载入 -> smem 转置 -> 逐 16-K 组量化（组 amax 完全落在块内，无跨块归约）
// -> 8B 合并写 B4T（行 n、K 方向 8 个字节 = SFB 512B 原子里的 4 个字节）。
// 与 fp8 版本同构，区别是 level-2 的 per-16 SF 逐组产出并落到 512B SF 原子。
template <bool kLevel1>
__global__ void quantize_bt_tile_fp4_kernel(const hip_bfloat16 *__restrict__ B,
                                            uint8_t *__restrict__ b4t,
                                            uint8_t *__restrict__ sfb,
                                            const float *__restrict__ sb, int K,
                                            int N) {
  constexpr int kBNt = 128; // n 方向 tile（= B4T 行数）
  constexpr int kBKt = 64;  // k 方向 tile（= B 行数）
  constexpr int kVec = 8;
  constexpr int kChunksPerRow = kBNt / kVec; // 16
  constexpr int kChunks = kBKt * kChunksPerRow;
  const int n0 = blockIdx.x * kBNt;
  const int k0 = blockIdx.y * kBKt;
  const int tid = threadIdx.x;
  const int nb_k = K / 64;

  __shared__ __align__(16) hip_bfloat16 tile[kBKt][kBNt]; // 源朝向 16KB
  // 转置朝向 16.6KB。多出来的 1 列是 padding：写 tile_t[c8+e][r] 时 e 方向
  // 的步长是整行，行宽 64 个字会让 16 个同 r 的线程全撞在同一个 bank 上。
  __shared__ __align__(16) hip_bfloat16 tile_t[kBNt][kBKt + 1];

  for (int i = tid; i < kChunks; i += 256) { // coalesced 载入 128(n) x 64(k)
    const int r = i / kChunksPerRow, c8 = (i % kChunksPerRow) * kVec;
    Vec8BF16 v;
    v.raw = make_uint4(0, 0, 0, 0);
    if (n0 + c8 < N)
      v = *reinterpret_cast<const Vec8BF16 *>(B + (size_t)(k0 + r) * N + n0 +
                                              c8);
    *reinterpret_cast<uint4 *>(&tile[r][c8]) = v.raw;
  }
  __syncthreads();
  for (int i = tid; i < kChunks; i += 256) { // 散写转置朝向的 tile_t
    const int r = i / kChunksPerRow, c8 = (i % kChunksPerRow) * kVec;
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      tile_t[c8 + e][r] = tile[r][c8 + e];
  }
  __syncthreads();
  // 每个 (n, 16-K 组) 一组：128 n x 4 组 = 512 组 / 256 线程 = 2 组/线程
  for (int i = tid; i < kBNt * (kBKt / kSFVec); i += 256) {
    const int n = i / (kBKt / kSFVec), g = i % (kBKt / kSFVec);
    if (n0 + n >= N) { // 补零行：SF 必须显式写 0（与 A 侧同一条理由）
      sfb[sf_byte_offset(n0 + n, (k0 + g * kSFVec) / kSFVec, nb_k)] = 0;
      continue;
    }
    float inv1 = 1.0f;
    if constexpr (kLevel1) { // sb[n] = 列 amax / 2688，由 amax kernel 产出
      const float s = sb[n0 + n];
      inv1 = (s > 0.0f) ? (1.0f / s) : 0.0f;
    }
    float x[kSFVec];
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < kSFVec; ++e) {
      x[e] = float(tile_t[n][g * kSFVec + e]) * inv1;
      amax = fmaxf(amax, fabsf(x[e]));
    }
    const int s = (k0 + g * kSFVec) / kSFVec; // 全局 K 组号
    const unsigned char sf_byte = cvt_f2_to_ue4m3(amax / kE2M1Max);
    const float sf_val = cvt_ue4m3_to_f32(sf_byte);
    // SF 下溢到 0（amax < 6 * 2^-10）-> 整组归零（inv2 = 0），避免 0/0 = NaN
    const float inv2 = (sf_val > 0.0f) ? (1.0f / sf_val) : 0.0f;
    sfb[sf_byte_offset(n0 + n, s, nb_k)] = sf_byte;
    uint2 out;
    unsigned char *o = reinterpret_cast<unsigned char *>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      o[e] = cvt_f2_to_e2m1x2(x[2 * e + 1] * inv2, x[2 * e] * inv2);
    *reinterpret_cast<uint2 *>(reinterpret_cast<unsigned char *>(b4t) +
                               (size_t)(n0 + n) * (K / 2) + k0 / 2 + g * 8) =
        out;
  }
}

// B 侧量化入口：全链路与 B 离线量化共用。level1=false 时列级 sb 无意义
// （inv1 恒为 1），只跑转置量化；level1=true 时先跑列 amax（K 切成 8 份，
// N=4096 时 256 个 CTA）再跑转置量化（N=4096,K=4096 时 2048 个 CTA）。
inline void fp4_gemm_quantize_b(const hip_bfloat16 *B, uint8_t *b4t,
                                uint8_t *sfb, float *sb, int N, int K,
                                bool level1, hipStream_t stream) {
  check_fp4_shape(1, N, K);
  const dim3 grid((N + 127) / 128, K / 64);
  if (level1) {
    check_hip(hipMemsetAsync(sb, 0, (size_t)N * sizeof(float), stream));
    const int kSplit = (K >= 512) ? 8 : 1;
    const dim3 grid_sa((N + 127) / 128, kSplit);
    col_amax_fp4_kernel<<<grid_sa, 256, 0, stream>>>(B, sb, N, K);
    quantize_bt_tile_fp4_kernel<true>
        <<<grid, 256, 0, stream>>>(B, b4t, sfb, sb, K, N);
  } else {
    quantize_bt_tile_fp4_kernel<false>
        <<<grid, 256, 0, stream>>>(B, b4t, sfb, sb, K, N);
  }
  check_hip(hipGetLastError());
}

// Keep BN/stages as real parameters. Choose the largest BM<=128 (multiple of
// 32) that fits the packed ring in RDNA4's 64 KiB LDS. The original wide
// BN=256/stages=6 becomes BM=32; default BN=128/stages=4 retains BM=128.
template <int BN = 128, int STAGES = 4> struct Fp4GemmTraits {
  static_assert(BN >= 128 && BN % 128 == 0 && STAGES >= 2 && STAGES <= 6);
  static constexpr int kBK = 64, kBN = BN, kStages = STAGES;
  static constexpr int available_rows = 65536 / (STAGES * 36) - BN;
  static constexpr int kBM =
      available_rows >= 128 ? 128 : (available_rows / 32) * 32;
  static_assert(kBM >= 32, "FP4 packed pipeline exceeds LDS capacity");
  static constexpr int kNumWarps = (kBM / 32) * (BN / 64);
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kStageBytes = (kBM + BN) * 36;
  static constexpr int kSmemBytes = STAGES * kStageBytes;
  static_assert(kNumThreads <= 1024 && kSmemBytes <= 65536);
};
using HalfFragment = _Float16 __attribute__((ext_vector_type(8)));
using HalfPair = _Float16 __attribute__((ext_vector_type(2)));
using AccFragment = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ unsigned e2m1_half_bits(unsigned code) {
  const unsigned magnitude = code & 7;
  const unsigned value =
      magnitude < 2 ? magnitude * 0x3800u : (magnitude + 28u) << 9;
  return value | ((code & 8) << 12);
}
__device__ __forceinline__ unsigned sf_half_bits(unsigned code) {
  if (code >= 8)
    return (code + 64) << 7;
  const unsigned exponent = 31 - __clz(code | 1u);
  const unsigned value =
      ((exponent + 6) << 10) | ((code << (10 - exponent)) & 1023);
  return code ? value : 0;
}
__device__ __forceinline__ HalfPair expand_pair(unsigned codes, unsigned sf) {
  const unsigned bits =
      e2m1_half_bits(codes) | (e2m1_half_bits(codes >> 4) << 16);
  return __builtin_bit_cast(HalfPair, bits) *
         __builtin_bit_cast(HalfPair, sf | (sf << 16));
}
__device__ __forceinline__ int packed_index(int row, int col) {
  return row * 32 + (col ^ ((row & 7) * 4));
}
template <typename T>
__device__ __forceinline__ void
load_tile(const uint8_t *a, const uint8_t *bt, const uint8_t *sfa,
          const uint8_t *sfb, uint8_t *tile, int m, int n, int k, int base,
          int tid, int threads) {
  constexpr int packed_bytes = (T::kBM + T::kBN) * 32;
  for (int group = tid; group < (T::kBM + T::kBN) * 4; group += threads) {
    const int row = group / 4, g = group % 4;
    const bool from_a = row < T::kBM;
    const int local = from_a ? row : row - T::kBM;
    const int global =
        local + (from_a ? blockIdx.y * T::kBM : blockIdx.x * T::kBN);
    uint2 codes = make_uint2(0, 0);
    uint8_t scale = 0;
    if (global < (from_a ? m : n)) {
      codes = *reinterpret_cast<const uint2 *>(
          (from_a ? a : bt) + size_t(global) * (k / 2) + base / 2 + g * 8);
      scale =
          (from_a ? sfa : sfb)[sf_byte_offset(global, base / 16 + g, k / 64)];
    }
    *reinterpret_cast<unsigned *>(tile + packed_index(row, g * 8)) = codes.x;
    *reinterpret_cast<unsigned *>(tile + packed_index(row, g * 8 + 4)) =
        codes.y;
    tile[packed_bytes + row * 4 + g] = scale;
  }
}
template <typename T>
__device__ __forceinline__ HalfFragment fragment(const uint8_t *tile, int row,
                                                 int kk, int half) {
  const unsigned packed = *reinterpret_cast<const unsigned *>(
      tile + packed_index(row, kk / 2 + half * 4));
  const unsigned sf =
      sf_half_bits(tile[(T::kBM + T::kBN) * 32 + row * 4 + kk / 16]);
  HalfFragment value;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // Every finite E2M1*E4M3 product is exactly representable in FP16.
    const HalfPair product = expand_pair(packed >> (i * 8), sf);
    value[i * 2] = product[0];
    value[i * 2 + 1] = product[1];
  }
  return value;
}
template <typename T>
__device__ __forceinline__ void compute_tile(const uint8_t *tile,
                                             AccFragment (&acc)[2][4]) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int wm = wave / (T::kBN / 64) * 32, wn = wave % (T::kBN / 64) * 64;
#pragma unroll 1
  for (int kk = 0; kk < 64; kk += 16) {
    HalfFragment av[2], bv[4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
      av[i] = fragment<T>(tile, wm + i * 16 + lane % 16, kk, lane / 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      bv[j] =
          fragment<T>(tile, T::kBM + wn + j * 16 + lane % 16, kk, lane / 16);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            av[i], bv[j], acc[i][j]);
  }
}
template <bool L1A, bool L1B, bool WS, typename T>
__global__ void fp4_gemm_amd_kernel(const uint8_t *a, const uint8_t *bt,
                                    const uint8_t *sfa, const uint8_t *sfb,
                                    const float *sa, const float *sb,
                                    hip_bfloat16 *c, int m, int n, int k) {
  __shared__ __align__(16) uint8_t ring[T::kSmemBytes];
  const int tid = threadIdx.x;
  const bool consumer = !WS || tid < T::kNumThreads,
             producer = !WS || !consumer;
  const int lt = WS ? tid - T::kNumThreads : tid,
            nt = WS ? 128 : T::kNumThreads;
  constexpr int ahead = WS ? T::kStages - 1 : T::kStages;
  const int tiles = k / 64;
  if (producer)
    for (int t = 0; t < ahead && t < tiles; ++t)
      load_tile<T>(a, bt, sfa, sfb, ring + t * T::kStageBytes, m, n, k, t * 64,
                   lt, nt);
  __syncthreads();
  AccFragment acc[2][4] = {};
  for (int t = 0; t < tiles; ++t) {
    if (consumer)
      compute_tile<T>(ring + (t % T::kStages) * T::kStageBytes, acc);
    if constexpr (!WS)
      __syncthreads();
    const int next = t + ahead;
    if (producer && next < tiles)
      load_tile<T>(a, bt, sfa, sfb, ring + (next % T::kStages) * T::kStageBytes,
                   m, n, k, next * 64, lt, nt);
    __syncthreads();
  }
  if (consumer) {
    const int wave = tid / 32, lane = tid % 32;
    const int wm = wave / (T::kBN / 64) * 32, wn = wave % (T::kBN / 64) * 64;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int row = blockIdx.y * T::kBM + wm + i * 16 + lane / 16 * 8 + v;
          const int col = blockIdx.x * T::kBN + wn + j * 16 + lane % 16;
          if (row < m && col < n)
            c[size_t(row) * n + col] = hip_bfloat16(
                acc[i][j][v] * (L1A ? sa[row] : 1.f) * (L1B ? sb[col] : 1.f));
        }
  }
}
template <bool A, bool B, bool WS, typename T = Fp4GemmTraits<>>
inline void fp4_gemm_tma_fwd(const uint8_t *a, const uint8_t *bt,
                             const uint8_t *sfa, const uint8_t *sfb,
                             const float *sa, const float *sb, hip_bfloat16 *c,
                             int m, int n, int k, hipStream_t stream) {
  static_assert(T::kNumThreads + (WS ? 128 : 0) <= 1024);
  check_fp4_shape(m, n, k);
  const dim3 grid((n + T::kBN - 1) / T::kBN, (m + T::kBM - 1) / T::kBM);
  fp4_gemm_amd_kernel<A, B, WS, T>
      <<<grid, T::kNumThreads + (WS ? 128 : 0), 0, stream>>>(
          a, bt, sfa, sfb, sa, sb, c, m, n, k);
  check_hip(hipGetLastError());
}
template <typename T = Fp4GemmTraits<>>
inline void dispatch_gemm(const uint8_t *a, const uint8_t *bt,
                          const uint8_t *sfa, const uint8_t *sfb,
                          const float *sa, const float *sb, hip_bfloat16 *c,
                          int m, int n, int k, bool l1a, bool l1b, bool ws,
                          hipStream_t stream) {
#define CALL(A, B)                                                             \
  if (ws)                                                                      \
    fp4_gemm_tma_fwd<A, B, true, T>(a, bt, sfa, sfb, sa, sb, c, m, n, k,       \
                                    stream);                                   \
  else                                                                         \
    fp4_gemm_tma_fwd<A, B, false, T>(a, bt, sfa, sfb, sa, sb, c, m, n, k,      \
                                     stream)
  if (l1a && l1b) {
    CALL(true, true);
  } else if (l1a) {
    CALL(true, false);
  } else if (l1b) {
    CALL(false, true);
  } else {
    CALL(false, false);
  }
#undef CALL
}
inline size_t fp4_gemm_align256(size_t size) {
  return (size + 255) & ~size_t(255);
}
struct Fp4GemmWorkspace {
  void *buf = nullptr;
  size_t size = 0;
};
struct Fp4GemmActivation {
  void *buf = nullptr;
  size_t size = 0;
};
inline size_t fp4_gemm_workspace_size(int m, int n, int k) {
  check_fp4_shape(m, n, k);
  return fp4_gemm_align256(size_t(m) * (k / 2)) +
         fp4_gemm_align256(size_t(n) * (k / 2)) +
         fp4_gemm_align256(sf_buffer_bytes(m, k)) +
         fp4_gemm_align256(sf_buffer_bytes(n, k)) +
         fp4_gemm_align256(size_t(m) * 4) + fp4_gemm_align256(size_t(n) * 4);
}
inline size_t fp4_gemm_activation_size(int m, int k) {
  check_fp4_shape(m, 8, k);
  return fp4_gemm_align256(size_t(m) * (k / 2)) +
         fp4_gemm_align256(sf_buffer_bytes(m, k)) +
         fp4_gemm_align256(size_t(m) * 4);
}
inline void quantize_a(const hip_bfloat16 *a, uint8_t *a4, uint8_t *sfa,
                       float *sa, int m, int k, bool l1, hipStream_t stream) {
  check_fp4_shape(m, 8, k);
  const int groups = ((m + 127) / 128 * 128) * (k / 16);
  if (l1) {
    row_amax_fp4_kernel<<<(m + 7) / 8, 256, 0, stream>>>(a, sa, m, k);
    quantize_a_fp4_kernel<true>
        <<<(groups + 255) / 256, 256, 0, stream>>>(a, a4, sfa, sa, m, k);
  } else {
    quantize_a_fp4_kernel<false>
        <<<(groups + 255) / 256, 256, 0, stream>>>(a, a4, sfa, sa, m, k);
  }
  check_hip(hipGetLastError());
}
template <typename T = Fp4GemmTraits<>>
inline void fp4_gemm_bf16(const hip_bfloat16 *a, const hip_bfloat16 *b,
                          hip_bfloat16 *c, int m, int n, int k,
                          Fp4GemmScaleMode mode, Fp4GemmWorkspace &ws,
                          hipStream_t stream, bool use_ws = false) {
  const auto bytes = fp4_gemm_workspace_size(m, n, k);
  if (!ws.buf || ws.size < bytes || uintptr_t(ws.buf) % 256)
    throw std::invalid_argument(
        "FP4 workspace missing, too small, or unaligned");
  if (int(mode) < 0 || int(mode) > 3)
    throw std::invalid_argument("invalid FP4 mode");
  auto *a4 = static_cast<uint8_t *>(ws.buf);
  auto *bt = a4 + fp4_gemm_align256(size_t(m) * (k / 2));
  auto *sfa = bt + fp4_gemm_align256(size_t(n) * (k / 2));
  auto *sfb = sfa + fp4_gemm_align256(sf_buffer_bytes(m, k));
  auto *sa =
      reinterpret_cast<float *>(sfb + fp4_gemm_align256(sf_buffer_bytes(n, k)));
  auto *sb = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(sa) +
                                       fp4_gemm_align256(size_t(m) * 4));
  const bool l1a = int(mode) & 1, l1b = int(mode) & 2;
  quantize_a(a, a4, sfa, sa, m, k, l1a, stream);
  fp4_gemm_quantize_b(b, bt, sfb, sb, n, k, l1b, stream);
  check_hip(hipGetLastError());
  dispatch_gemm<T>(a4, bt, sfa, sfb, sa, sb, c, m, n, k, l1a, l1b, use_ws,
                   stream);
}
template <bool A, bool B, bool WS = false, typename T = Fp4GemmTraits<>>
inline void fp4_gemm_bf16_b_offline(const hip_bfloat16 *a, const uint8_t *bt,
                                    const uint8_t *sfb, const float *sb,
                                    hip_bfloat16 *c, int m, int n, int k,
                                    Fp4GemmActivation &act,
                                    hipStream_t stream) {
  check_fp4_shape(m, n, k);
  if (!act.buf || act.size < fp4_gemm_activation_size(m, k) ||
      uintptr_t(act.buf) % 256)
    throw std::invalid_argument(
        "FP4 activation workspace missing, too small, or unaligned");
  auto *a4 = static_cast<uint8_t *>(act.buf);
  auto *sfa = a4 + fp4_gemm_align256(size_t(m) * (k / 2));
  auto *sa =
      reinterpret_cast<float *>(sfa + fp4_gemm_align256(sf_buffer_bytes(m, k)));
  quantize_a(a, a4, sfa, sa, m, k, A, stream);
  fp4_gemm_tma_fwd<A, B, WS, T>(a4, bt, sfa, sfb, sa, sb, c, m, n, k, stream);
}
inline bool fp4_gemm_use_wide_tile(int m, int n, int k) {
  return m >= 4096 && n >= 4096 && k >= 4096;
}
} // namespace fp4_gemm
