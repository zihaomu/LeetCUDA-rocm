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
void tensor(Tensor x, Tensor q, at::ScalarType dtype) {
  TORCH_CHECK(x.is_cuda() && x.device() == q.device() &&
                  x.scalar_type() == dtype,
              "interview attention dtype/device mismatch");
  TORCH_CHECK(x.is_contiguous() && !x.requires_grad() && x.numel() <= INT_MAX,
              "interview attention requires contiguous inference tensors "
              "within int32 indexing");
}
template <int Kind, int D>
void call(Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, int sk, int sv,
          int padmask, bool causal, float scale, unsigned *stats, int delay,
          int ctas) {
  C10_CUDA_CHECK((leetcuda_interview_fa::launch<Kind, D>(
      reinterpret_cast<const _Float16 *>(q.data_ptr()),
      reinterpret_cast<const _Float16 *>(k.data_ptr()),
      reinterpret_cast<const _Float16 *>(v.data_ptr()),
      reinterpret_cast<_Float16 *>(o.data_ptr()), lse.data_ptr<float>(),
      q.size(0), q.size(1), k.size(1), q.size(2), k.size(2), sk, sv, padmask,
      causal, scale, at::cuda::getCurrentCUDAStream(), stats, delay, ctas)));
}
void forward(Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, int kind,
             int sk, int sv, int padmask, bool causal, float scale, int ctas,
             c10::optional<Tensor> audit, int delay) {
  TORCH_CHECK(q.is_cuda(), "interview attention requires GPU tensors");
  const c10::cuda::CUDAGuard guard(q.device());
  for (auto t : {q, k, v, o})
    tensor(t, q, at::kHalf);
  tensor(lse, q, at::kFloat);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes() &&
                  o.sizes() == q.sizes() &&
                  lse.sizes() == q.sizes().slice(0, 3),
              "interview attention shape mismatch");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3) &&
                  q.size(0) > 0 && q.size(1) > 0 && k.size(1) > 0 &&
                  q.size(1) % k.size(1) == 0 && q.size(2) > 0 && k.size(2) > 0,
              "interview attention batch/head/length mismatch");
  TORCH_CHECK(kind >= 0 && kind <= 7 && kind != 6,
              "interview attention invalid compute variant");
  int d = q.size(3);
  TORCH_CHECK(d == 64 || d == 128 || (kind == 0 && d == 32) ||
                  (kind == 7 && d == 96),
              "interview attention unsupported D");
  TORCH_CHECK(kind == 7 || (q.sizes() == k.sizes() && !causal && ctas == 0),
              "GQA, distinct lengths, causal and persistent CTA count require "
              "persist_d");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 && delay >= 0 && delay <= 2 &&
                  ctas >= 0,
              "interview attention invalid scale/delay/grid");
  TORCH_CHECK(sk >= 1 && sk <= (kind == 7 ? 2 : 4) && sv >= 1 &&
                  sv <= ((kind == 1 || kind == 4 || kind == 7) ? 2 : 1),
              "interview attention invalid K/V stage depth");
  TORCH_CHECK(padmask >= 0 && padmask <= 7 && (kind == 0 || padmask == 0),
              "padding applies only to mma variant");
  TORCH_CHECK(q.size(2) <= INT_MAX - 64 && k.size(2) <= INT_MAX - 64,
              "interview attention length range exceeded");
  for (auto t : {q, k, v})
    TORCH_CHECK(!shared(o, t) && !shared(lse, t),
                "interview attention output aliases input");
  TORCH_CHECK(!shared(o, lse), "interview attention outputs alias");
  int br = (kind == 0 || kind == 3) ? 64 : 32;
  int64_t total = q.size(0) * q.size(1) * ((q.size(2) + br - 1) / br);
  TORCH_CHECK(total <= INT_MAX, "interview attention grid range exceeded");
  int grid = total;
  if (kind == 7 && (!causal || ctas > 0)) {
    if (ctas == 0)
      C10_CUDA_CHECK(hipDeviceGetAttribute(
          &ctas, hipDeviceAttributeMultiprocessorCount, q.get_device()));
    grid = std::min(grid, ctas);
  }
  unsigned *stats = nullptr;
  if (audit.has_value()) {
    auto a = *audit;
    tensor(a, q, at::kInt);
    TORCH_CHECK(a.numel() == int64_t(grid) * 16,
                "interview attention metrics size mismatch");
    for (auto t : {q, k, v, o, lse})
      TORCH_CHECK(!shared(a, t), "interview attention metrics alias");
    stats = reinterpret_cast<unsigned *>(a.data_ptr());
  }
#define CALL(K, D)                                                             \
  call<K, D>(q, k, v, o, lse, sk, sv, padmask, causal, scale, stats, delay,    \
             ctas)
#define CASE(K)                                                                \
  case K:                                                                      \
    if (d == 64) {                                                             \
      CALL(K, 64);                                                             \
    } else {                                                                   \
      CALL(K, 128);                                                            \
    }                                                                          \
    break
  switch (kind) {
  case 0:
    if (d == 32) {
      CALL(0, 32);
    } else if (d == 64) {
      CALL(0, 64);
    } else {
      CALL(0, 128);
    }
    break;
    CASE(1);
    CASE(2);
    CASE(3);
    CASE(4);
    CASE(5);
  case 7:
    if (d == 64) {
      CALL(7, 64);
    } else if (d == 96) {
      CALL(7, 96);
    } else {
      CALL(7, 128);
    }
    break;
  }
#undef CASE
#undef CALL
}
void copy(Tensor q, Tensor o) {
  TORCH_CHECK(q.is_cuda(), "interview copy requires GPU tensors");
  const c10::cuda::CUDAGuard guard(q.device());
  tensor(q, q, at::kHalf);
  tensor(o, q, at::kHalf);
  TORCH_CHECK(q.dim() == 2 && q.sizes() == o.sizes() && q.size(0) > 0 &&
                  q.size(0) <= INT_MAX - 64 &&
                  (q.size(1) == 64 || q.size(1) == 128),
              "interview copy expects [rows,D64/128]");
  TORCH_CHECK(!shared(q, o), "interview copy output aliases input");
  auto stream = at::cuda::getCurrentCUDAStream();
  int grid = (q.size(0) + 63) / 64;
  if (q.size(1) == 64)
    leetcuda_interview_fa::copy_kernel<64><<<grid, 128, 0, stream>>>(
        reinterpret_cast<const _Float16 *>(q.data_ptr()),
        reinterpret_cast<_Float16 *>(o.data_ptr()), q.size(0));
  else
    leetcuda_interview_fa::copy_kernel<128><<<grid, 128, 0, stream>>>(
        reinterpret_cast<const _Float16 *>(q.data_ptr()),
        reinterpret_cast<_Float16 *>(o.data_ptr()), q.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int Kind, int D, int Rows> std::vector<int> coords(bool pad) {
  std::vector<int> result;
  for (int r = 0; r < Rows; r++)
    for (int c = 0; c < D; c++)
      result.push_back(
          leetcuda_interview_fa::Config<Kind, D>::template index<Rows>(r, c,
                                                                       pad));
  return result;
}
template <int Kind, int D> std::vector<int> coords_rows(int rows, bool pad) {
  if (rows == 16)
    return coords<Kind, D, 16>(pad);
  if (rows == 32)
    return coords<Kind, D, 32>(pad);
  if (rows == 64)
    return coords<Kind, D, 64>(pad);
  TORCH_CHECK(false, "unsupported layout rows");
}
std::vector<int> layout(int kind, int d, int rows, bool pad) {
  if (kind == 0) {
    if (d == 32)
      return coords_rows<0, 32>(rows, pad);
    if (d == 64)
      return coords_rows<0, 64>(rows, pad);
    if (d == 128)
      return coords_rows<0, 128>(rows, pad);
  }
  if (kind >= 1 && kind <= 7 && !pad) {
    if (d == 64)
      return coords_rows<3, 64>(rows, false);
    if (d == 96)
      return coords_rows<7, 96>(rows, false);
    if (d == 128)
      return coords_rows<3, 128>(rows, false);
  }
  TORCH_CHECK(false, "unsupported layout kind/D/padding");
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("copy", &copy);
  m.def("layout", &layout);
}
