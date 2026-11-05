// Test-only binding to original NVFP4 kernels. No algorithm substitutions.
#include "fp4_reference.cuh"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {
using torch::Tensor;
using Default = fp4_gemm::Fp4GemmTraits<>;
using Wide = fp4_gemm::Fp4GemmTraits<256, 6>;
template <class T> T *ptr(Tensor x) {
  return reinterpret_cast<T *>(x.data_ptr());
}
void quant_a(Tensor x, Tensor y, Tensor sf, Tensor s, bool l1) {
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int m = x.size(0), k = x.size(1);
  if (l1)
    fp4_gemm::row_amax_fp4_kernel<<<(m + 7) / 8, 256, 0, stream>>>(
        ptr<__nv_bfloat16>(x), s.data_ptr<float>(), m, k);
  int blocks = (((m + 127) / 128 * 128) * (k / 16) + 255) / 256;
#define CALL(L1)                                                               \
  fp4_gemm::quantize_a_fp4_kernel<L1><<<blocks, 256, 0, stream>>>(             \
      ptr<__nv_bfloat16>(x), ptr<cutlass::float_e2m1_t>(y),                    \
      ptr<cutlass::float_ue4m3_t>(sf), s.data_ptr<float>(), m, k)
  if (l1) {
    CALL(true);
  } else {
    CALL(false);
  }
#undef CALL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void quant_b(Tensor x, Tensor y, Tensor sf, Tensor s, bool l1) {
  c10::cuda::CUDAGuard guard(x.device());
  fp4_gemm::fp4_gemm_quantize_b(
      ptr<__nv_bfloat16>(x), ptr<cutlass::float_e2m1_t>(y),
      ptr<cutlass::float_ue4m3_t>(sf), s.data_ptr<float>(), x.size(1),
      x.size(0), l1, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <bool A, bool B, bool WS, class T>
void quantized_t(Tensor a, Tensor b, Tensor af, Tensor bf, Tensor sa, Tensor sb,
                 Tensor out) {
  fp4_gemm::fp4_gemm_tma_fwd<A, B, WS, T>(
      ptr<cutlass::float_e2m1_t>(a), ptr<cutlass::float_e2m1_t>(b),
      ptr<cutlass::float_ue4m3_t>(af), ptr<cutlass::float_ue4m3_t>(bf),
      sa.data_ptr<float>(), sb.data_ptr<float>(), ptr<cutlass::bfloat16_t>(out),
      a.size(0), b.size(0), a.size(1) * 2, at::cuda::getCurrentCUDAStream());
}
template <class T>
void quantized_dispatch(Tensor a, Tensor b, Tensor af, Tensor bf, Tensor sa,
                        Tensor sb, Tensor out, bool l1a, bool l1b, bool ws) {
#define CALL(A, B)                                                             \
  if (ws)                                                                      \
    quantized_t<A, B, true, T>(a, b, af, bf, sa, sb, out);                     \
  else                                                                         \
    quantized_t<A, B, false, T>(a, b, af, bf, sa, sb, out)
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
void quantized(Tensor a, Tensor b, Tensor af, Tensor bf, Tensor sa, Tensor sb,
               Tensor out, bool l1a, bool l1b, bool ws, bool wide) {
  c10::cuda::CUDAGuard guard(a.device());
  if (wide)
    quantized_dispatch<Wide>(a, b, af, bf, sa, sb, out, l1a, l1b, ws);
  else
    quantized_dispatch<Default>(a, b, af, bf, sa, sb, out, l1a, l1b, ws);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void forward(Tensor a, Tensor b, Tensor c, Tensor w, int mode, bool ws,
             bool wide) {
  c10::cuda::CUDAGuard guard(a.device());
  fp4_gemm::Fp4GemmWorkspace workspace{w.data_ptr(), size_t(w.numel())};
#define CALL(T)                                                                \
  fp4_gemm::fp4_gemm_bf16<T>(ptr<__nv_bfloat16>(a), ptr<__nv_bfloat16>(b),     \
                             ptr<__nv_bfloat16>(c), a.size(0), b.size(1),      \
                             a.size(1),                                        \
                             static_cast<fp4_gemm::Fp4GemmScaleMode>(mode),    \
                             workspace, at::cuda::getCurrentCUDAStream(), ws)
  if (wide) {
    CALL(Wide);
  } else {
    CALL(Default);
  }
#undef CALL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <bool A, bool B, bool WS, class T>
void offline_t(Tensor a, Tensor b, Tensor bf, Tensor sb, Tensor c, Tensor w) {
  fp4_gemm::Fp4GemmActivation act{w.data_ptr(), size_t(w.numel())};
  fp4_gemm::fp4_gemm_bf16_b_offline<A, B, WS, T>(
      ptr<__nv_bfloat16>(a), ptr<cutlass::float_e2m1_t>(b),
      ptr<cutlass::float_ue4m3_t>(bf), sb.data_ptr<float>(),
      ptr<__nv_bfloat16>(c), a.size(0), b.size(0), a.size(1), act,
      at::cuda::getCurrentCUDAStream());
}
template <class T>
void offline_dispatch(Tensor a, Tensor b, Tensor bf, Tensor sb, Tensor c,
                      Tensor w, bool l1a, bool l1b, bool ws) {
#define CALL(A, B)                                                             \
  if (ws)                                                                      \
    offline_t<A, B, true, T>(a, b, bf, sb, c, w);                              \
  else                                                                         \
    offline_t<A, B, false, T>(a, b, bf, sb, c, w)
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
void offline(Tensor a, Tensor b, Tensor bf, Tensor sb, Tensor c, Tensor w,
             bool l1a, bool l1b, bool ws, bool wide) {
  c10::cuda::CUDAGuard guard(a.device());
  if (wide)
    offline_dispatch<Wide>(a, b, bf, sb, c, w, l1a, l1b, ws);
  else
    offline_dispatch<Default>(a, b, bf, sb, c, w, l1a, l1b, ws);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_a", &quant_a);
  m.def("quantize_b", &quant_b);
  m.def("quantized", &quantized);
  m.def("forward", &forward);
  m.def("offline", &offline);
  m.def("workspace_size", &fp4_gemm::fp4_gemm_workspace_size);
  m.def("activation_size", &fp4_gemm::fp4_gemm_activation_size);
  m.def("sf_size", &fp4_gemm::sf_buffer_bytes);
}
