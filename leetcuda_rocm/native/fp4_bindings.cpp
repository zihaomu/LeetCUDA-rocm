#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {
using torch::Tensor;
using Default = fp4_gemm::Fp4GemmTraits<>;
using Wide = fp4_gemm::Fp4GemmTraits<256, 6>;
void common(std::initializer_list<Tensor> tensors) {
  auto device = tensors.begin()->device();
  for (const auto &t : tensors) {
    TORCH_CHECK(t.is_cuda() && t.device() == device,
                "all tensors must be on the same GPU");
    TORCH_CHECK(t.is_contiguous(), "FP4 requires contiguous tensors");
    TORCH_CHECK(!t.requires_grad(), "forward-only interface");
    TORCH_CHECK(t.numel() <= INT32_MAX, "tensor exceeds 32-bit indexing");
    TORCH_CHECK(uintptr_t(t.data_ptr()) % 16 == 0,
                "FP4 requires 16-byte storage alignment");
  }
}
void matrix(Tensor t, at::ScalarType type) {
  TORCH_CHECK(t.dim() == 2 && t.scalar_type() == type,
              "matrix rank or dtype mismatch");
}
void scales(Tensor sf, Tensor s, int rows, int k) {
  TORCH_CHECK(sf.dim() == 1 && sf.scalar_type() == at::kByte &&
                  sf.numel() == fp4_gemm::sf_buffer_bytes(rows, k),
              "invalid canonical FP4 block scales");
  TORCH_CHECK(s.dim() == 1 && s.scalar_type() == at::kFloat &&
                  s.numel() == rows,
              "invalid level-1 scales");
}
void no_alias(Tensor output, std::initializer_list<Tensor> inputs) {
  for (const auto &t : inputs)
    TORCH_CHECK(!output.is_alias_of(t), "output/workspace aliases input");
}
void check_output(Tensor c, int m, int n) {
  matrix(c, at::kBFloat16);
  TORCH_CHECK(c.size(0) == m && c.size(1) == n, "output shape mismatch");
}
void quant_a(Tensor x, Tensor y, Tensor sf, Tensor s, bool l1) {
  common({x, y, sf, s});
  matrix(x, at::kBFloat16);
  matrix(y, at::kByte);
  const int m = x.size(0), k = x.size(1);
  fp4_gemm::check_fp4_shape(m, 8, k);
  scales(sf, s, m, k);
  TORCH_CHECK(y.size(0) == m && y.size(1) == k / 2, "packed A must be [M,K/2]");
  no_alias(y, {x, sf, s});
  no_alias(sf, {x, s});
  no_alias(s, {x});
  const c10::cuda::CUDAGuard guard(x.device());
  fp4_gemm::quantize_a(reinterpret_cast<const hip_bfloat16 *>(x.data_ptr()),
                       y.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(),
                       s.data_ptr<float>(), m, k, l1,
                       at::cuda::getCurrentCUDAStream());
}
void quant_b(Tensor x, Tensor y, Tensor sf, Tensor s, bool l1) {
  common({x, y, sf, s});
  matrix(x, at::kBFloat16);
  matrix(y, at::kByte);
  const int k = x.size(0), n = x.size(1);
  fp4_gemm::check_fp4_shape(1, n, k);
  scales(sf, s, n, k);
  TORCH_CHECK(y.size(0) == n && y.size(1) == k / 2, "packed B must be [N,K/2]");
  no_alias(y, {x, sf, s});
  no_alias(sf, {x, s});
  no_alias(s, {x});
  const c10::cuda::CUDAGuard guard(x.device());
  fp4_gemm::fp4_gemm_quantize_b(
      reinterpret_cast<const hip_bfloat16 *>(x.data_ptr()),
      y.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(), s.data_ptr<float>(), n, k,
      l1, at::cuda::getCurrentCUDAStream());
  fp4_gemm::check_hip(hipGetLastError());
}
void quantized(Tensor a, Tensor bt, Tensor sfa, Tensor sfb, Tensor sa,
               Tensor sb, Tensor c, bool l1a, bool l1b, bool ws, bool wide) {
  common({a, bt, sfa, sfb, sa, sb, c});
  matrix(a, at::kByte);
  matrix(bt, at::kByte);
  TORCH_CHECK(a.size(1) <= INT32_MAX / 2, "packed K exceeds 32-bit indexing");
  const int m = a.size(0), n = bt.size(0), k = a.size(1) * 2;
  fp4_gemm::check_fp4_shape(m, n, k);
  TORCH_CHECK(bt.size(1) == k / 2, "packed K mismatch");
  scales(sfa, sa, m, k);
  scales(sfb, sb, n, k);
  check_output(c, m, n);
  no_alias(c, {a, bt, sfa, sfb, sa, sb});
  const c10::cuda::CUDAGuard guard(a.device());
#define GEMM(T)                                                                \
  fp4_gemm::dispatch_gemm<T>(                                                  \
      a.data_ptr<uint8_t>(), bt.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(),  \
      sfb.data_ptr<uint8_t>(), sa.data_ptr<float>(), sb.data_ptr<float>(),     \
      reinterpret_cast<hip_bfloat16 *>(c.data_ptr()), m, n, k, l1a, l1b, ws,   \
      at::cuda::getCurrentCUDAStream())
  if (wide) {
    GEMM(Wide);
  } else {
    GEMM(Default);
  }
#undef GEMM
}
void forward(Tensor a, Tensor b, Tensor c, Tensor workspace, int mode, bool ws,
             bool wide) {
  common({a, b, c, workspace});
  matrix(a, at::kBFloat16);
  matrix(b, at::kBFloat16);
  const int m = a.size(0), k = a.size(1), n = b.size(1);
  TORCH_CHECK(b.size(0) == k, "K mismatch");
  check_output(c, m, n);
  TORCH_CHECK(workspace.dim() == 1 && workspace.scalar_type() == at::kByte,
              "workspace must be a byte vector");
  no_alias(c, {a, b, workspace});
  no_alias(workspace, {a, b});
  const c10::cuda::CUDAGuard guard(a.device());
  fp4_gemm::Fp4GemmWorkspace buffer{workspace.data_ptr(),
                                    size_t(workspace.numel())};
#define FORWARD(T)                                                             \
  fp4_gemm::fp4_gemm_bf16<T>(                                                  \
      reinterpret_cast<const hip_bfloat16 *>(a.data_ptr()),                    \
      reinterpret_cast<const hip_bfloat16 *>(b.data_ptr()),                    \
      reinterpret_cast<hip_bfloat16 *>(c.data_ptr()), m, n, k,                 \
      static_cast<fp4_gemm::Fp4GemmScaleMode>(mode), buffer,                   \
      at::cuda::getCurrentCUDAStream(), ws)
  if (wide) {
    FORWARD(Wide);
  } else {
    FORWARD(Default);
  }
#undef FORWARD
}
template <typename T>
void offline_impl(Tensor a, Tensor bt, Tensor sfb, Tensor sb, Tensor c,
                  Tensor workspace, bool l1a, bool l1b, bool ws) {
  const int m = a.size(0), k = a.size(1), n = bt.size(0);
  fp4_gemm::Fp4GemmActivation buffer{workspace.data_ptr(),
                                     size_t(workspace.numel())};
  auto pa = reinterpret_cast<const hip_bfloat16 *>(a.data_ptr());
  auto pc = reinterpret_cast<hip_bfloat16 *>(c.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
#define CALL(A, B)                                                             \
  if (ws)                                                                      \
    fp4_gemm::fp4_gemm_bf16_b_offline<A, B, true, T>(                          \
        pa, bt.data_ptr<uint8_t>(), sfb.data_ptr<uint8_t>(),                   \
        sb.data_ptr<float>(), pc, m, n, k, buffer, stream);                    \
  else                                                                         \
    fp4_gemm::fp4_gemm_bf16_b_offline<A, B, false, T>(                         \
        pa, bt.data_ptr<uint8_t>(), sfb.data_ptr<uint8_t>(),                   \
        sb.data_ptr<float>(), pc, m, n, k, buffer, stream)
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
void offline(Tensor a, Tensor bt, Tensor sfb, Tensor sb, Tensor c,
             Tensor workspace, bool l1a, bool l1b, bool ws, bool wide) {
  common({a, bt, sfb, sb, c, workspace});
  matrix(a, at::kBFloat16);
  matrix(bt, at::kByte);
  const int m = a.size(0), k = a.size(1), n = bt.size(0);
  fp4_gemm::check_fp4_shape(m, n, k);
  scales(sfb, sb, n, k);
  check_output(c, m, n);
  TORCH_CHECK(bt.size(1) == k / 2, "packed K mismatch");
  TORCH_CHECK(workspace.dim() == 1 && workspace.scalar_type() == at::kByte,
              "workspace must be a byte vector");
  no_alias(c, {a, bt, sfb, sb, workspace});
  no_alias(workspace, {a, bt, sfb, sb});
  const c10::cuda::CUDAGuard guard(a.device());
  if (wide)
    offline_impl<Wide>(a, bt, sfb, sb, c, workspace, l1a, l1b, ws);
  else
    offline_impl<Default>(a, bt, sfb, sb, c, workspace, l1a, l1b, ws);
}
__global__ void decode_pairs_kernel(const uint8_t *codes, const uint8_t *scales,
                                    float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto pair =
        fp4_gemm::expand_pair(codes[i], fp4_gemm::sf_half_bits(scales[i]));
    out[2 * i] = float(pair[0]);
    out[2 * i + 1] = float(pair[1]);
  }
}
void decode_pairs(Tensor codes, Tensor sf, Tensor out) {
  common({codes, sf, out});
  TORCH_CHECK(
      codes.dim() == 1 && sf.dim() == 1 && out.dim() == 1 &&
          codes.scalar_type() == at::kByte && sf.scalar_type() == at::kByte &&
          out.scalar_type() == at::kFloat && codes.numel() == sf.numel() &&
          out.numel() == codes.numel() * 2 && codes.numel() > 0,
      "invalid decode check buffers");
  no_alias(out, {codes, sf});
  const c10::cuda::CUDAGuard guard(codes.device());
  hipLaunchKernelGGL(decode_pairs_kernel, dim3((codes.numel() + 255) / 256),
                     dim3(256), 0, at::cuda::getCurrentCUDAStream(),
                     codes.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(),
                     out.data_ptr<float>(), int(codes.numel()));
  fp4_gemm::check_hip(hipGetLastError());
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
  m.def("decode_pairs", &decode_pairs);
  m.def("configuration", [](bool wide) {
    if (wide)
      return std::vector<int>{Wide::kBM,         Wide::kBN,
                              Wide::kBK,         Wide::kStages,
                              Wide::kNumThreads, Wide::kSmemBytes};
    return std::vector<int>{Default::kBM,         Default::kBN,
                            Default::kBK,         Default::kStages,
                            Default::kNumThreads, Default::kSmemBytes};
  });
}
