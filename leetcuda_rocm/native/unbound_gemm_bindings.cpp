#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {
using torch::Tensor;
bool shared(Tensor a, Tensor b) {
  return a.storage().unsafeGetStorageImpl() ==
         b.storage().unsafeGetStorageImpl();
}
void tensor(Tensor x, Tensor a, at::ScalarType dtype) {
  TORCH_CHECK(x.is_cuda() && x.device() == a.device() &&
                  x.scalar_type() == dtype,
              "unbound GEMM tensor dtype/device mismatch");
  TORCH_CHECK(x.is_contiguous() && !x.requires_grad() && x.numel() <= INT_MAX,
              "unbound GEMM requires contiguous inference tensors within int32 "
              "indexing");
}
template <int Kind, int Stage, bool F32, bool Audit>
void call(Tensor a, Tensor b, Tensor c, bool swizzle, int stride,
          unsigned *stats) {
  C10_CUDA_CHECK(
      (leetcuda_interview_hgemm::launch<Kind, Stage, F32, Audit, Audit>(
          reinterpret_cast<const _Float16 *>(a.data_ptr()),
          reinterpret_cast<const _Float16 *>(b.data_ptr()),
          reinterpret_cast<_Float16 *>(c.data_ptr()), a.size(0), b.size(1),
          a.size(1), swizzle, at::cuda::getCurrentCUDAStream(), stats,
          stride)));
}
template <int Kind, bool F32, bool Audit>
void dispatch(Tensor a, Tensor b, Tensor c, int stages, bool swizzle,
              int stride, unsigned *stats) {
  switch (stages) {
  case 2:
    call<Kind, 2, F32, Audit>(a, b, c, swizzle, stride, stats);
    break;
  case 3:
    call<Kind, 3, F32, Audit>(a, b, c, swizzle, stride, stats);
    break;
  case 4:
    call<Kind, 4, F32, Audit>(a, b, c, swizzle, stride, stats);
    break;
  case 5:
    call<Kind, 5, F32, Audit>(a, b, c, swizzle, stride, stats);
    break;
  default:
    TORCH_CHECK(false, "invalid resolved stage");
  }
}
void forward(Tensor a, Tensor b, Tensor c, int stages, bool swizzle, int stride,
             int kind, bool fp32, c10::optional<Tensor> audit) {
  TORCH_CHECK(a.is_cuda(), "unbound GEMM requires GPU tensors");
  const c10::cuda::CUDAGuard guard(a.device());
  for (auto x : {a, b, c})
    tensor(x, a, at::kHalf);
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2 && c.dim() == 2 && a.size(1) == b.size(0) &&
          c.size(0) == a.size(0) && c.size(1) == b.size(1),
      "unbound GEMM expects A[M,K], B[K,N] storing physical BT[N,K], C[M,N]");
  TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0 &&
                  a.size(0) <= 65535 * 64 && a.size(1) <= INT_MAX - 128 &&
                  b.size(1) <= INT_MAX - 128,
              "unbound GEMM dimensions exceed qualified launch range");
  TORCH_CHECK(kind == 3 || kind == 5, "unbound GEMM unsupported schedule");
  TORCH_CHECK(!swizzle || (stride > 0 && 1 + (b.size(1) - 1) / stride <= 65535),
              "unbound GEMM invalid swizzle stride/grid");
  TORCH_CHECK(!shared(a, c) && !shared(b, c),
              "unbound GEMM output aliases input storage");
  if (stages < 2 || stages > 5)
    stages = kind == 3 ? 3 : 2; // upstream fallback contract
  unsigned *stats = nullptr;
  if (audit.has_value()) {
    auto x = *audit;
    tensor(x, a, at::kInt);
    int z = swizzle ? 1 + (b.size(1) - 1) / stride : 1;
    int gx = ((b.size(1) + 127) / 128 + z - 1) / z, gy = (a.size(0) + 63) / 64;
    TORCH_CHECK(kind == 3 && fp32 && x.numel() == int64_t(gx) * gy * z * 16,
                "pipeline audit requires WS FP32 and exact metrics size");
    for (auto y : {a, b, c})
      TORCH_CHECK(!shared(x, y), "audit aliases tensor");
    stats = reinterpret_cast<unsigned *>(x.data_ptr());
    dispatch<3, true, true>(a, b, c, stages, swizzle, stride, stats);
  } else if (kind == 3) {
    if (fp32)
      dispatch<3, true, false>(a, b, c, stages, swizzle, stride, stats);
    else
      dispatch<3, false, false>(a, b, c, stages, swizzle, stride, stats);
  } else {
    if (fp32)
      dispatch<5, true, false>(a, b, c, stages, swizzle, stride, stats);
    else
      dispatch<5, false, false>(a, b, c, stages, swizzle, stride, stats);
  }
}
void wgmma_f32(Tensor a, Tensor b, Tensor c, int s, bool sw, int stride) {
  forward(a, b, c, s, sw, stride, 3, true, c10::nullopt);
}
// Explicit FP32 accumulator upgrade for the original FP16-state interface.
void wgmma_f16(Tensor a, Tensor b, Tensor c, int s, bool sw, int stride) {
  forward(a, b, c, s, sw, stride, 3, true, c10::nullopt);
}
void tn_swizzle(Tensor a, Tensor b, Tensor c, int s, bool sw, int stride) {
  forward(a, b, c, s, sw, stride, 5, true, c10::nullopt);
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hgemm_wgmma_m64n128k16_stages_tma_ws_tn", &wgmma_f32);
  m.def("hgemm_wgmma_m64n128k16_f16acc_stages_tma_ws_tn", &wgmma_f16);
  m.def("hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_swizzle",
        &tn_swizzle);
  m.def("_diagnostic", &forward);
}
