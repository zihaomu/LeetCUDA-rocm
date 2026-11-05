// Test-only binding to the original CUDA pipeline (or explicit E5M2 variant).
#include "fp8_reference.cuh"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {
using torch::Tensor;
using FP8 = fp8_gemm::Fp8GemmTraits<>::Element;
const __nv_bfloat16 *bf(Tensor x) {
  return reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
}
void quant_a(Tensor x, Tensor y, Tensor s, bool row) {
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int m = x.size(0), k = x.size(1);
  auto *yp = reinterpret_cast<FP8 *>(y.data_ptr());
  if (row)
    fp8_gemm::quantize_a_perrow_kernel<<<(m + 7) / 8, 256, 0, stream>>>(
        bf(x), yp, s.data_ptr<float>(), m, k);
  else
    fp8_gemm::quantize_a_perblock_kernel<<<(m + 127) / 128, 256, 0, stream>>>(
        bf(x), yp, s.data_ptr<float>(), m, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void quant_b(Tensor x, Tensor y, Tensor s, bool col) {
  c10::cuda::CUDAGuard guard(x.device());
  fp8_gemm::fp8_gemm_quantize_b(bf(x), reinterpret_cast<FP8 *>(y.data_ptr()),
                                s.data_ptr<float>(), x.size(1), x.size(0), col,
                                at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <bool R, bool C, bool WS>
void quantized_t(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor c) {
  fp8_gemm::fp8_gemm_tma_fwd<R, C, WS>(
      reinterpret_cast<FP8 *>(a.data_ptr()),
      reinterpret_cast<FP8 *>(b.data_ptr()), sa.data_ptr<float>(),
      sb.data_ptr<float>(),
      reinterpret_cast<cutlass::bfloat16_t *>(c.data_ptr()), a.size(0),
      b.size(0), a.size(1), at::cuda::getCurrentCUDAStream());
}
void quantized(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor c, bool row,
               bool col, bool ws) {
  c10::cuda::CUDAGuard guard(a.device());
#define CALL(R, C)                                                             \
  if (ws)                                                                      \
    quantized_t<R, C, true>(a, b, sa, sb, c);                                  \
  else                                                                         \
    quantized_t<R, C, false>(a, b, sa, sb, c)
  if (row && col) {
    CALL(true, true);
  } else if (row) {
    CALL(true, false);
  } else if (col) {
    CALL(false, true);
  } else {
    CALL(false, false);
  }
#undef CALL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void forward(Tensor a, Tensor b, Tensor c, Tensor w, int mode, bool ws) {
  c10::cuda::CUDAGuard guard(a.device());
  fp8_gemm::Fp8GemmWorkspace workspace{w.data_ptr(), size_t(w.numel())};
  fp8_gemm::fp8_gemm_bf16(
      bf(a), bf(b), reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), a.size(0),
      b.size(1), a.size(1), static_cast<fp8_gemm::Fp8GemmScaleMode>(mode),
      workspace, at::cuda::getCurrentCUDAStream(), ws);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <bool R, bool C, bool WS>
void offline_t(Tensor a, Tensor b, Tensor sb, Tensor c, Tensor w) {
  fp8_gemm::Fp8GemmActivation act{w.data_ptr(), size_t(w.numel())};
  fp8_gemm::fp8_gemm_bf16_b_offline<R, C, WS>(
      bf(a), reinterpret_cast<FP8 *>(b.data_ptr()), sb.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), a.size(0), b.size(0),
      a.size(1), act, at::cuda::getCurrentCUDAStream());
}
void offline(Tensor a, Tensor b, Tensor sb, Tensor c, Tensor w, bool row,
             bool col, bool ws) {
  c10::cuda::CUDAGuard guard(a.device());
#define CALL(R, C)                                                             \
  if (ws)                                                                      \
    offline_t<R, C, true>(a, b, sb, c, w);                                     \
  else                                                                         \
    offline_t<R, C, false>(a, b, sb, c, w)
  if (row && col) {
    CALL(true, true);
  } else if (row) {
    CALL(true, false);
  } else if (col) {
    CALL(false, true);
  } else {
    CALL(false, false);
  }
#undef CALL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_a", &quant_a);
  m.def("quantize_b", &quant_b);
  m.def("quantized", &quantized);
  m.def("forward", &forward);
  m.def("offline", &offline);
  m.def("workspace_size", &fp8_gemm::fp8_gemm_workspace_size);
  m.def("activation_size", &fp8_gemm::fp8_gemm_activation_size);
}
