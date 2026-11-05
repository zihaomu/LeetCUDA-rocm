// Original cuBLAS entrypoints backed by hipBLAS/rocBLAS on AMD.
// HIPBLAS_COMPUTE_16F permits at least FP16 precision, matching the original
// non-pedantic compute mode. It does not promise per-step FP16 rounding.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <limits>
#include <map>
#include <torch/extension.h>

namespace {

void check_blas(hipblasStatus_t status) {
  TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, "hipBLAS status ", int(status));
}

struct Handles {
  std::map<int, hipblasHandle_t> by_device;
  ~Handles() {
    // No exceptions during process/thread teardown.
    for (auto entry : by_device)
      hipblasDestroy(entry.second);
  }
};
thread_local Handles handles;

hipblasHandle_t current_handle() {
  int device;
  C10_CUDA_CHECK(hipGetDevice(&device));
  auto found = handles.by_device.find(device);
  if (found == handles.by_device.end()) {
    hipblasHandle_t handle = nullptr;
    check_blas(hipblasCreate(&handle));
    found = handles.by_device.emplace(device, handle).first;
  }
  check_blas(hipblasSetStream(found->second, at::cuda::getCurrentCUDAStream()));
  check_blas(hipblasSetPointerMode(found->second, HIPBLAS_POINTER_MODE_HOST));
  return found->second;
}

void check_gemm(const torch::Tensor &a, const torch::Tensor &b,
                const torch::Tensor &c, at::ScalarType dtype) {
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() &&
                  c.device() == a.device(),
              "GEMM requires tensors on the same GPU");
  TORCH_CHECK(a.scalar_type() == dtype && b.scalar_type() == dtype &&
                  c.scalar_type() == dtype,
              "GEMM dtype mismatch");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2 &&
                  a.size(1) == b.size(0) && c.size(0) == a.size(0) &&
                  c.size(1) == b.size(1),
              "expected a[M,K], b[K,N], c[M,N]");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "GEMM requires contiguous storage");
  TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b),
              "GEMM output aliases input");
  TORCH_CHECK(!a.requires_grad() && !b.requires_grad() && !c.requires_grad(),
              "forward-only interface");
  for (const auto &tensor : {a, b, c}) {
    TORCH_CHECK(tensor.numel() <= std::numeric_limits<int>::max() &&
                    tensor.size(0) <= std::numeric_limits<int>::max() &&
                    tensor.size(1) <= std::numeric_limits<int>::max(),
                "GEMM exceeds 32-bit interface dimensions");
  }
}

template <bool HALF, bool TRANSPOSE_B>
void gemm(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check_gemm(a, b, c, HALF ? at::kHalf : at::kFloat);
  const c10::cuda::CUDAGuard guard(a.device());
  const int m = a.size(0), k = a.size(1), n = b.size(1);
  if (m == 0 || n == 0)
    return;
  if (k == 0) {
    c.zero_();
    return;
  }
  const float alpha_f = 1.0f, beta_f = 0.0f;
  const at::Half alpha_h = 1.0f, beta_h = 0.0f;
  const void *alpha = HALF ? static_cast<const void *>(&alpha_h) : &alpha_f;
  const void *beta = HALF ? static_cast<const void *>(&beta_h) : &beta_f;
  const auto dtype = HALF ? HIP_R_16F : HIP_R_32F;
  check_blas(hipblasGemmEx(
      current_handle(), TRANSPOSE_B ? HIPBLAS_OP_T : HIPBLAS_OP_N, HIPBLAS_OP_N,
      n, m, k, alpha, b.data_ptr(), dtype, TRANSPOSE_B ? k : n, a.data_ptr(),
      dtype, k, beta, c.data_ptr(), dtype, n,
      HALF ? HIPBLAS_COMPUTE_16F : HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

} // namespace

void sgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  gemm<false, false>(a, b, c);
}
void hgemm_cublas_tensor_op_nn(torch::Tensor a, torch::Tensor b,
                               torch::Tensor c) {
  gemm<true, false>(a, b, c);
}
void hgemm_cublas_tensor_op_tn(torch::Tensor a, torch::Tensor b,
                               torch::Tensor c) {
  // Preserve the original ABI: b has shape [K,N], but its raw storage is [N,K].
  gemm<true, true>(a, b, c);
}
void init_cublas_handle() { current_handle(); }
void destroy_cublas_handle() {
  int device;
  C10_CUDA_CHECK(hipGetDevice(&device));
  auto found = handles.by_device.find(device);
  if (found != handles.by_device.end()) {
    check_blas(hipblasDestroy(found->second));
    handles.by_device.erase(found);
  }
}

// TF32x3 on a backend without native TF32: explicitly rounded high/residual
// FP32 storage followed by three hipBLAS products. Exceptional exponents use
// full input values in the high component. No ordinary-SGEMM function alias.
namespace {
__device__ float tf32_rna(float x) {
  unsigned u = __float_as_uint(x);
  if ((u & 0x7fffffff) >= 0x7f800000)
    return x;
  return __uint_as_float((u + 0x1000u) & 0xffffe000u);
}
__global__ void tf32_split(const float *input, float *hi, float *lo, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = input[i], absx = fabsf(x);
    if (absx != 0 && !(absx >= 0x1p-96f && absx <= 0x1p112f)) {
      hi[i] = x;
      lo[i] = 0;
    } else {
      float h = tf32_rna(x);
      hi[i] = h;
      lo[i] = tf32_rna(x - h);
    }
  }
}
} // namespace
void sgemm_cublas_tf32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  check_gemm(a, b, c, at::kFloat);
  c10::cuda::CUDAGuard guard(a.device());
  int m = a.size(0), n = b.size(1), k = a.size(1);
  if (m == 0 || n == 0)
    return;
  if (k == 0) {
    c.zero_();
    return;
  }
  auto ah = torch::empty_like(a), al = torch::empty_like(a),
       bh = torch::empty_like(b), bl = torch::empty_like(b);
  int ablocks = (a.numel() + 255) / 256, bblocks = (b.numel() + 255) / 256;
  tf32_split<<<ablocks, 256>>>(a.data_ptr<float>(), ah.data_ptr<float>(),
                               al.data_ptr<float>(), a.numel());
  tf32_split<<<bblocks, 256>>>(b.data_ptr<float>(), bh.data_ptr<float>(),
                               bl.data_ptr<float>(), b.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const float one = 1, zero = 0;
  for (int part = 0; part < 3; part++) {
    auto left = part == 0 ? al : ah;
    auto right = part == 1 ? bl : bh;
    check_blas(hipblasGemmEx(current_handle(), HIPBLAS_OP_N, HIPBLAS_OP_N, n, m,
                             k, &one, right.data_ptr(), HIP_R_32F, n,
                             left.data_ptr(), HIP_R_32F, k,
                             part == 0 ? &zero : &one, c.data_ptr(), HIP_R_32F,
                             n, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  }
}
