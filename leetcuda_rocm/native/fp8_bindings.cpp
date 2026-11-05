#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {
using torch::Tensor;
void common(std::initializer_list<Tensor> tensors) {
  const auto device = tensors.begin()->device();
  for (const auto &t : tensors) {
    TORCH_CHECK(t.is_cuda() && t.device() == device,
                "all tensors must be on the same GPU");
    TORCH_CHECK(t.is_contiguous(), "FP8 pipeline requires contiguous tensors");
    TORCH_CHECK(!t.requires_grad(), "forward-only interface");
    TORCH_CHECK(t.numel() <= INT32_MAX, "tensor exceeds 32-bit indexing");
    TORCH_CHECK(uintptr_t(t.data_ptr()) % 16 == 0,
                "FP8 pipeline requires 16-byte storage alignment");
  }
}
void matrix(Tensor t, at::ScalarType dtype) {
  TORCH_CHECK(t.dim() == 2 && t.scalar_type() == dtype,
              "matrix rank or dtype mismatch");
}
void scales(Tensor s, int rows, bool individual) {
  TORCH_CHECK(s.dim() == 1 && s.scalar_type() == at::kFloat &&
                  s.numel() == (individual ? rows : (rows + 127) / 128),
              "invalid FP8 scales");
}
void no_alias(Tensor out, std::initializer_list<Tensor> inputs) {
  for (const auto &t : inputs)
    TORCH_CHECK(!out.is_alias_of(t), "output/workspace aliases input");
}
void check_out(Tensor c, int m, int n) {
  matrix(c, at::kBFloat16);
  TORCH_CHECK(c.size(0) == m && c.size(1) == n, "output shape mismatch");
}
void quant_a(Tensor x, Tensor y, Tensor s, bool row) {
  common({x, y, s});
  matrix(x, at::kBFloat16);
  matrix(y, at::kByte);
  const int m = x.size(0), k = x.size(1);
  fp8_gemm::check_fp8_shape(m, 8, k);
  TORCH_CHECK(y.sizes() == x.sizes(), "quantized A shape mismatch");
  scales(s, m, row);
  no_alias(y, {x, s});
  no_alias(s, {x});
  const c10::cuda::CUDAGuard guard(x.device());
  fp8_gemm::quantize_a(reinterpret_cast<const hip_bfloat16 *>(x.data_ptr()),
                       y.data_ptr<uint8_t>(), s.data_ptr<float>(), m, k, row,
                       at::cuda::getCurrentCUDAStream());
}
void quant_b(Tensor x, Tensor y, Tensor s, bool col) {
  common({x, y, s});
  matrix(x, at::kBFloat16);
  matrix(y, at::kByte);
  const int k = x.size(0), n = x.size(1);
  fp8_gemm::check_fp8_shape(1, n, k);
  TORCH_CHECK(y.size(0) == n && y.size(1) == k, "quantized B must be [N,K]");
  scales(s, n, col);
  no_alias(y, {x, s});
  no_alias(s, {x});
  const c10::cuda::CUDAGuard guard(x.device());
  fp8_gemm::fp8_gemm_quantize_b(
      reinterpret_cast<const hip_bfloat16 *>(x.data_ptr()),
      y.data_ptr<uint8_t>(), s.data_ptr<float>(), n, k, col,
      at::cuda::getCurrentCUDAStream());
  fp8_gemm::check_fp8_launch();
}
void quantized(Tensor a, Tensor bt, Tensor sa, Tensor sb, Tensor c, bool row,
               bool col, bool ws) {
  common({a, bt, sa, sb, c});
  matrix(a, at::kByte);
  matrix(bt, at::kByte);
  const int m = a.size(0), k = a.size(1), n = bt.size(0);
  fp8_gemm::check_fp8_shape(m, n, k);
  TORCH_CHECK(bt.size(1) == k, "quantized K mismatch");
  scales(sa, m, row);
  scales(sb, n, col);
  check_out(c, m, n);
  no_alias(c, {a, bt, sa, sb});
  const c10::cuda::CUDAGuard guard(a.device());
  fp8_gemm::dispatch_gemm(a.data_ptr<uint8_t>(), bt.data_ptr<uint8_t>(),
                          sa.data_ptr<float>(), sb.data_ptr<float>(),
                          reinterpret_cast<hip_bfloat16 *>(c.data_ptr()), m, n,
                          k, row, col, ws, at::cuda::getCurrentCUDAStream());
}
void forward(Tensor a, Tensor b, Tensor c, Tensor workspace, int mode,
             bool use_ws) {
  common({a, b, c, workspace});
  matrix(a, at::kBFloat16);
  matrix(b, at::kBFloat16);
  const int m = a.size(0), k = a.size(1), n = b.size(1);
  TORCH_CHECK(b.size(0) == k, "K mismatch");
  check_out(c, m, n);
  TORCH_CHECK(workspace.dim() == 1 && workspace.scalar_type() == at::kByte,
              "workspace must be a byte vector");
  no_alias(c, {a, b, workspace});
  no_alias(workspace, {a, b});
  const c10::cuda::CUDAGuard guard(a.device());
  fp8_gemm::Fp8GemmWorkspace ws{workspace.data_ptr(),
                                size_t(workspace.numel())};
  fp8_gemm::fp8_gemm_bf16(reinterpret_cast<const hip_bfloat16 *>(a.data_ptr()),
                          reinterpret_cast<const hip_bfloat16 *>(b.data_ptr()),
                          reinterpret_cast<hip_bfloat16 *>(c.data_ptr()), m, n,
                          k, static_cast<fp8_gemm::Fp8GemmScaleMode>(mode), ws,
                          at::cuda::getCurrentCUDAStream(), use_ws);
}
void offline(Tensor a, Tensor bt, Tensor sb, Tensor c, Tensor workspace,
             bool row, bool col, bool use_ws) {
  common({a, bt, sb, c, workspace});
  matrix(a, at::kBFloat16);
  matrix(bt, at::kByte);
  const int m = a.size(0), k = a.size(1), n = bt.size(0);
  TORCH_CHECK(bt.size(1) == k, "K mismatch");
  scales(sb, n, col);
  check_out(c, m, n);
  TORCH_CHECK(workspace.dim() == 1 && workspace.scalar_type() == at::kByte,
              "workspace must be a byte vector");
  no_alias(c, {a, bt, sb, workspace});
  no_alias(workspace, {a, bt, sb});
  const c10::cuda::CUDAGuard guard(a.device());
  fp8_gemm::Fp8GemmActivation act{workspace.data_ptr(),
                                  size_t(workspace.numel())};
  const auto *pa = reinterpret_cast<const hip_bfloat16 *>(a.data_ptr());
  auto *pc = reinterpret_cast<hip_bfloat16 *>(c.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
#define CALL(R, C)                                                             \
  if (use_ws)                                                                  \
    fp8_gemm::fp8_gemm_bf16_b_offline<R, C, true>(pa, bt.data_ptr<uint8_t>(),  \
                                                  sb.data_ptr<float>(), pc, m, \
                                                  n, k, act, stream);          \
  else                                                                         \
    fp8_gemm::fp8_gemm_bf16_b_offline<R, C, false>(pa, bt.data_ptr<uint8_t>(), \
                                                   sb.data_ptr<float>(), pc,   \
                                                   m, n, k, act, stream)
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
