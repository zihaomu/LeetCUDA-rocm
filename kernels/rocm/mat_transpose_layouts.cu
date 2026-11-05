// AMD implementations of the eleven CuTe-labelled transpose interfaces.
// Preserve register/LDS, row/column copy, vectorization and swizzle variants;
// no dependency on NVIDIA CuTe. The original CUDA implementations remain intact.
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <torch/extension.h>

namespace {

void check_transpose(const torch::Tensor &x, const torch::Tensor &y,
                     bool vectorized) {
  TORCH_CHECK(x.is_cuda() && y.device() == x.device(),
              "transpose inputs must be on the same GPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
              "transpose requires FP32 tensors");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && y.size(0) == x.size(1) &&
                  y.size(1) == x.size(0),
              "expected x[M,N] and y[N,M]");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
              "transpose requires contiguous tensors");
  TORCH_CHECK(!x.is_alias_of(y), "transpose output must not alias input");
  TORCH_CHECK(!x.requires_grad() && !y.requires_grad(), "forward-only interface");
  TORCH_CHECK(x.numel() <= std::numeric_limits<int32_t>::max() &&
                  x.size(0) <= std::numeric_limits<int32_t>::max() - 128 &&
                  x.size(1) <= std::numeric_limits<int32_t>::max() - 128,
              "transpose exceeds 32-bit indexing including tile padding");
  if (vectorized) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0,
                "vectorized transpose requires 16-byte-aligned storage");
  }
}

template <bool COLUMN_LOAD>
__global__ void transpose_register(const float *x, float *y, int m, int n) {
  const int r = blockIdx.x * 16 +
                (COLUMN_LOAD ? threadIdx.x % 16 : threadIdx.x / 16);
  const int c = blockIdx.y * 16 +
                (COLUMN_LOAD ? threadIdx.x / 16 : threadIdx.x % 16);
  if (r < m && c < n)
    y[c * m + r] = x[r * n + c];
}

template <int BM, int BN, int SWIZZLE>
__device__ __forceinline__ int lds_offset(int r, int c) {
  // Swizzle<4,0,4> for original scalar/vector copies; optimized path stores
  // the transposed 128x8 tile with Swizzle<2,3,2> before vectorized output.
  int offset = SWIZZLE == 2 ? c * BM + r : r * BN + c;
  if constexpr (SWIZZLE == 1)
    offset ^= (offset >> 4) & 15;
  if constexpr (SWIZZLE == 2)
    offset ^= (offset >> 2) & 24;
  return offset;
}

template <bool COLUMN_LOAD, int SWIZZLE>
__global__ void transpose_scalar_lds(const float *x, float *y, int m, int n) {
  __shared__ float tile[16 * 16];
  const int tid = threadIdx.x;
  const int r = COLUMN_LOAD ? tid % 16 : tid / 16;
  const int c = COLUMN_LOAD ? tid / 16 : tid % 16;
  const int gr = blockIdx.x * 16 + r;
  const int gc = blockIdx.y * 16 + c;
  tile[lds_offset<16, 16, SWIZZLE>(r, c)] =
      gr < m && gc < n ? x[gr * n + gc] : 0.0f;
  __syncthreads();
  const int sr = COLUMN_LOAD ? tid / 16 : tid % 16;
  const int sc = COLUMN_LOAD ? tid % 16 : tid / 16;
  const int yr = blockIdx.x * 16 + sr;
  const int yc = blockIdx.y * 16 + sc;
  if (yr < m && yc < n)
    y[yc * m + yr] = tile[lds_offset<16, 16, SWIZZLE>(sr, sc)];
}

template <int BM, int BN, bool ROW_VECTOR, int SWIZZLE>
__global__ void transpose_vector_lds(const float *x, float *y, int m, int n) {
  __shared__ float tile[BM * BN];
  const int tid = threadIdx.x;
  const int r = ROW_VECTOR ? tid / (BN / 4) : 4 * (tid / BN);
  const int c = ROW_VECTOR ? 4 * (tid % (BN / 4)) : tid % BN;
  const int gr = blockIdx.x * BM + r;
  const int gc = blockIdx.y * BN + c;
  float values[4];
  if (ROW_VECTOR && gr < m && gc + 3 < n && (gr * n + gc) % 4 == 0) {
    const float4 packed = *reinterpret_cast<const float4 *>(x + gr * n + gc);
    values[0] = packed.x;
    values[1] = packed.y;
    values[2] = packed.z;
    values[3] = packed.w;
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int rr = gr + (ROW_VECTOR ? 0 : j);
      const int cc = gc + (ROW_VECTOR ? j : 0);
      values[j] = rr < m && cc < n ? x[rr * n + cc] : 0.0f;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
    tile[lds_offset<BM, BN, SWIZZLE>(r + (ROW_VECTOR ? 0 : j),
                                    c + (ROW_VECTOR ? j : 0))] = values[j];
  // Required across wavefronts, including the optimized register->LDS path.
  __syncthreads();
  constexpr bool VECTOR_STORE = !ROW_VECTOR || SWIZZLE == 2;
  const int sr = VECTOR_STORE ? 4 * (tid % (BM / 4)) : tid % BM;
  const int sc = VECTOR_STORE ? tid / (BM / 4) : 4 * (tid / BM);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    values[j] = tile[lds_offset<BM, BN, SWIZZLE>(
        sr + (VECTOR_STORE ? j : 0), sc + (VECTOR_STORE ? 0 : j))];
  const int yr = blockIdx.x * BM + sr;
  const int yc = blockIdx.y * BN + sc;
  if (VECTOR_STORE && yr + 3 < m && yc < n && (yc * m + yr) % 4 == 0) {
    *reinterpret_cast<float4 *>(y + yc * m + yr) =
        make_float4(values[0], values[1], values[2], values[3]);
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int rr = yr + (VECTOR_STORE ? j : 0);
      const int cc = yc + (VECTOR_STORE ? 0 : j);
      if (rr < m && cc < n)
        y[cc * m + rr] = values[j];
    }
  }
}

template <bool COLUMN_LOAD, int SWIZZLE, bool REGISTER_ONLY>
void scalar_launch(torch::Tensor x, torch::Tensor y) {
  check_transpose(x, y, false);
  if (x.numel() == 0)
    return;
  const c10::cuda::CUDAGuard guard(x.device());
  const int m = x.size(0), n = x.size(1);
  const dim3 grid((m + 15) / 16, (n + 15) / 16);
  if constexpr (REGISTER_ONLY)
    transpose_register<COLUMN_LOAD><<<grid, 256>>>(x.data_ptr<float>(),
                                                  y.data_ptr<float>(), m, n);
  else
    transpose_scalar_lds<COLUMN_LOAD, SWIZZLE><<<grid, 256>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), m, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BM, int BN, bool ROW_VECTOR, int SWIZZLE>
void vector_launch(torch::Tensor x, torch::Tensor y) {
  check_transpose(x, y, true);
  if (x.numel() == 0)
    return;
  const c10::cuda::CUDAGuard guard(x.device());
  const int m = x.size(0), n = x.size(1);
  const dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
  transpose_vector_lds<BM, BN, ROW_VECTOR, SWIZZLE><<<grid, BM * BN / 4>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), m, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void mat_transpose_cute_col2row_reg(torch::Tensor x, torch::Tensor y) {
  scalar_launch<false, 0, true>(x, y);
}
void mat_transpose_cute_row2col_reg(torch::Tensor x, torch::Tensor y) {
  scalar_launch<true, 0, true>(x, y);
}
void mat_transpose_cute_col_smem(torch::Tensor x, torch::Tensor y) {
  scalar_launch<true, 0, false>(x, y);
}
void mat_transpose_cute_row_smem(torch::Tensor x, torch::Tensor y) {
  scalar_launch<false, 0, false>(x, y);
}
void mat_transpose_cute_col_smem_swizzled(torch::Tensor x, torch::Tensor y) {
  scalar_launch<true, 1, false>(x, y);
}
void mat_transpose_cute_row_smem_swizzled(torch::Tensor x, torch::Tensor y) {
  scalar_launch<false, 1, false>(x, y);
}
void mat_transpose_cute_row_cvectorized(torch::Tensor x, torch::Tensor y) {
  vector_launch<64, 16, false, 0>(x, y);
}
void mat_transpose_cute_row_cvectorized_swizzled(torch::Tensor x,
                                              torch::Tensor y) {
  vector_launch<64, 16, false, 1>(x, y);
}
void mat_transpose_cute_row_rvectorized(torch::Tensor x, torch::Tensor y) {
  vector_launch<16, 64, true, 0>(x, y);
}
void mat_transpose_cute_row_rvectorized_swizzled(torch::Tensor x,
                                              torch::Tensor y) {
  vector_launch<16, 64, true, 1>(x, y);
}
void mat_transpose_cute_row_rvectorized_swizzled_optimized(torch::Tensor x,
                                                        torch::Tensor y) {
  vector_launch<8, 128, true, 2>(x, y);
}
