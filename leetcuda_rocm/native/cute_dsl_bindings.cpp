#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
using torch::Tensor;
void forward(Tensor a, Tensor b, Tensor o, int kind,
             c10::optional<Tensor> metrics) {
  TORCH_CHECK(a.is_cuda(), "GPU required");
  c10::cuda::CUDAGuard guard(a.device());
  for (auto t : {a, b, o})
    TORCH_CHECK(t.device() == a.device() && t.scalar_type() == at::kFloat &&
                    t.dim() == 2 && t.is_contiguous() && !t.requires_grad() &&
                    t.numel() <= INT_MAX,
                "DSL GEMM expects contiguous FP32 rank2 inference tensors");
  TORCH_CHECK(kind >= 0 && kind < 3, "DSL variant");
  int m = a.size(0), k = a.size(1), n = kind == 0 ? b.size(1) : b.size(0);
  TORCH_CHECK(m > 0 && n > 0 && k > 0 && k <= INT_MAX - 8 &&
                  m <= INT_MAX - 128 && n <= INT_MAX - 128 &&
                  (kind == 0 ? b.size(0) : b.size(1)) == k && o.size(0) == m &&
                  o.size(1) == n,
              "DSL shape");
  auto aliases = [](Tensor x, Tensor y) {
    return x.storage().unsafeGetStorageImpl() ==
           y.storage().unsafeGetStorageImpl();
  };
  TORCH_CHECK(!aliases(o, a) && !aliases(o, b), "DSL output alias");
  int gm = (m + 127) / 128, gn = (n + 127) / 128;
  unsigned *stats = nullptr;
  if (metrics.has_value()) {
    auto t = *metrics;
    TORCH_CHECK(kind > 0 && t.device() == a.device() &&
                    t.scalar_type() == at::kInt && t.is_contiguous() &&
                    t.numel() == int64_t(gm) * gn * 8 && !aliases(t, a) &&
                    !aliases(t, b) && !aliases(t, o),
                "DSL audit buffer");
    stats = reinterpret_cast<unsigned *>(t.data_ptr());
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (kind == 0) {
    using leetcuda_cute_dsl::naive;
    naive<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(32, 32), 0, stream>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), o.data_ptr<float>(), m, n,
            k);
  } else if (kind == 1) {
    leetcuda_cute_dsl::tiled<false><<<dim3(gm, gn), 256, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), o.data_ptr<float>(), m, n, k,
        stats);
  } else {
    leetcuda_cute_dsl::tiled<true><<<dim3(gm, gn), 256, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), o.data_ptr<float>(), m, n, k,
        stats);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
