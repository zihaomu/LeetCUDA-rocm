#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {
using torch::Tensor;
void tensor(Tensor x, Tensor q, at::ScalarType dtype) {
  TORCH_CHECK(x.is_cuda() && x.device() == q.device() &&
                  x.scalar_type() == dtype,
              "FFPA tensor dtype/device mismatch");
  TORCH_CHECK(x.is_contiguous() && !x.requires_grad(),
              "FFPA requires contiguous inference tensors");
  TORCH_CHECK(x.numel() <= INT_MAX, "FFPA index range exceeded");
}
bool shared(Tensor a, Tensor b) {
  return a.numel() && b.numel() &&
         a.storage().unsafeGetStorageImpl() ==
             b.storage().unsafeGetStorageImpl();
}
void forward(Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse,
             c10::optional<Tensor> audit, int kind, int sq, int sv, int delay) {
  TORCH_CHECK(q.is_cuda(), "FFPA requires GPU tensors");
  const c10::cuda::CUDAGuard guard(q.device());
  for (auto x : {q, k, v, o})
    tensor(x, q, at::kHalf);
  tensor(lse, q, at::kFloat);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes() &&
                  o.sizes() == q.sizes(),
              "FFPA shape mismatch");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) &&
                  q.size(3) == k.size(3),
              "FFPA batch/head/D mismatch");
  TORCH_CHECK(lse.sizes() == q.sizes().slice(0, 3), "FFPA LSE shape mismatch");
  TORCH_CHECK(q.size(0) * q.size(1) <= 65535 && q.size(2) <= INT_MAX - 16 &&
                  k.size(2) <= INT_MAX - 64,
              "FFPA grid range exceeded");
  for (auto x : {q, k, v})
    TORCH_CHECK(!shared(o, x) && !shared(lse, x), "FFPA output aliases input");
  TORCH_CHECK(!shared(o, lse), "FFPA outputs alias");
  TORCH_CHECK(kind >= 0 && kind <= 2 && sq >= 1 && sq <= 3 && sv >= 1 &&
                  sv <= 3 && delay >= 0 && delay <= 2,
              "FFPA invalid variant/stages/delay");
  int d = q.size(3);
  TORCH_CHECK((d >= 64 && d <= 512 && d % 64 == 0) || d == 1024,
              "FFPA unsupported head dimension");
  unsigned *stats = nullptr;
  if (audit.has_value()) {
    auto a = *audit;
    tensor(a, q, at::kInt);
    TORCH_CHECK(a.numel() ==
                    q.size(0) * q.size(1) * ((q.size(2) + 15) / 16) * 8,
                "FFPA audit size mismatch");
    for (auto x : {q, k, v, o, lse})
      TORCH_CHECK(!shared(a, x), "FFPA audit aliases tensor");
    stats = reinterpret_cast<unsigned *>(a.data_ptr());
  }
  C10_CUDA_CHECK(leetcuda_ffpa::forward(
      reinterpret_cast<const _Float16 *>(q.data_ptr()),
      reinterpret_cast<const _Float16 *>(k.data_ptr()),
      reinterpret_cast<const _Float16 *>(v.data_ptr()),
      reinterpret_cast<_Float16 *>(o.data_ptr()), lse.data_ptr<float>(),
      q.size(0) * q.size(1), q.size(2), k.size(2), d, kind, sq, sv,
      at::cuda::getCurrentCUDAStream(), stats, delay));
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
