#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
using torch::Tensor;
template <int Kind, int AP, int BP> void launch(Tensor a, Tensor b, Tensor o) {
  constexpr int bm = Kind == 1 || Kind == 3 ? 128 : 16,
                bn = Kind == 1 || Kind == 3 ? 128 : 8,
                threads = Kind == 1 || Kind == 3 ? 256 : 32;
  int m = a.size(0), k = a.size(1), n = b.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  leetcuda_swizzle_examples::gemm<Kind, AP, BP>
      <<<dim3((n + bn - 1) / bn, (m + bm - 1) / bm), threads, 0, stream>>>(
          reinterpret_cast<const _Float16 *>(a.data_ptr()),
          reinterpret_cast<const _Float16 *>(b.data_ptr()),
          reinterpret_cast<_Float16 *>(o.data_ptr()), m, n, k);
}
void forward(Tensor a, Tensor b, Tensor o, int kind, int ap, int bp) {
  TORCH_CHECK(a.is_cuda(), "GPU required");
  c10::cuda::CUDAGuard guard(a.device());
  for (auto t : {a, b, o})
    TORCH_CHECK(t.device() == a.device() && t.scalar_type() == at::kHalf &&
                    t.dim() == 2 && t.is_contiguous() && !t.requires_grad() &&
                    t.numel() <= INT_MAX,
                "swizzle GEMM expects contiguous FP16 rank2 inference tensors");
  int m = a.size(0), k = a.size(1), n = b.size(1);
  TORCH_CHECK(m > 0 && n > 0 && k > 0 && m <= INT_MAX - 128 &&
                  n <= INT_MAX - 128 && k <= INT_MAX - 16 && b.size(0) == k &&
                  o.size(0) == m && o.size(1) == n,
              "swizzle GEMM shape");
  TORCH_CHECK(o.storage().unsafeGetStorageImpl() !=
                      a.storage().unsafeGetStorageImpl() &&
                  o.storage().unsafeGetStorageImpl() !=
                      b.storage().unsafeGetStorageImpl(),
              "swizzle GEMM alias");
  TORCH_CHECK(kind >= 0 && kind < 5 && (ap == 0 || ap == 8) &&
                  (bp == 0 || bp == 8) &&
                  ((kind == 1 || kind == 3) || (!ap && !bp)),
              "swizzle GEMM kind/pad");
#define PAD(K)                                                                 \
  if (ap == 8) {                                                               \
    if (bp == 8)                                                               \
      launch<K, 8, 8>(a, b, o);                                                \
    else                                                                       \
      launch<K, 8, 0>(a, b, o);                                                \
  } else {                                                                     \
    if (bp == 8)                                                               \
      launch<K, 0, 8>(a, b, o);                                                \
    else                                                                       \
      launch<K, 0, 0>(a, b, o);                                                \
  }
  if (kind == 0) {
    launch<0, 0, 0>(a, b, o);
  } else if (kind == 1) {
    PAD(1);
  } else if (kind == 2) {
    launch<2, 0, 0>(a, b, o);
  } else if (kind == 3) {
    PAD(3);
  } else {
    launch<4, 0, 0>(a, b, o);
  }
#undef PAD
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> layout(int rows) {
  TORCH_CHECK(rows == 16 || rows == 128, "layout rows");
  std::vector<int> r;
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < 16; j++)
      r.push_back(i * 16 + leetcuda_swizzle_examples::xor_column(i, j));
  return r;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("layout", &layout);
}
