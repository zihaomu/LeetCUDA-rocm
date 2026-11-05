#include "bench_runtime.hpp"
#if defined(LEETCUDA_ROCM)
#include "interview/hgemm.cuh"
namespace hg = leetcuda_interview_hgemm;
#else
#include <cublas_v2.h>
#include <cuda_fp16.h>
inline cudaStream_t leetcuda_test_stream;
#include "interview_hgemm_reference.cuh"
#endif
using namespace leetcuda_bench;

// Both backends use this standalone CPU oracle and poisoned graph check.
template <class F>
void halves(Harness &h, const std::string &id, const std::vector<double> &ref,
            F fn) {
  Buffer<half> out(ref.size() + 16);
  std::vector<half> initial(out.n, __float2half(NAN));
  for (int i = 0; i < 8; i++)
    initial[i] = initial[out.n - 1 - i] = __float2half(91);
  out.upload(initial);
  auto verify = [&](const std::string &key, double ms = -1.) {
    auto full = out.download();
    bool canary = true;
    for (int i = 0; i < 8; i++)
      canary &= __half2float(full[i]) == 91 &&
                __half2float(full[out.n - 1 - i]) == 91;
    std::vector<float> values(ref.size());
    for (size_t i = 0; i < values.size(); i++)
      values[i] = __half2float(full[i + 8]);
    h.record(key, values, ref, .01, 0, canary, ms);
  };
  fn(out.p + 8);
  check(GPU(GetLastError)());
  check(GPU(StreamSynchronize)(h.stream));
  verify(id, h.timed([&] { fn(out.p + 8); }));
  GPU(Graph_t) graph;
  GPU(GraphExec_t) exec;
  check(GPU(StreamBeginCapture)(h.stream, GPU(StreamCaptureModeGlobal)));
  fn(out.p + 8);
  check(GPU(StreamEndCapture)(h.stream, &graph));
  size_t nodes = 0;
  check(GPU(GraphGetNodes)(graph, nullptr, &nodes));
  if (!nodes)
    throw std::runtime_error("empty graph");
  check(GPU(GraphInstantiateWithFlags)(&exec, graph, 0));
  out.upload(initial);
  check(GPU(GraphLaunch)(exec, h.stream));
  check(GPU(StreamSynchronize)(h.stream));
  verify(id + "/graph");
  check(GPU(GraphExecDestroy)(exec));
  check(GPU(GraphDestroy)(graph));
}
#if !defined(LEETCUDA_ROCM)
struct CudaContext {
  cublasHandle_t blas{};
  CUtensorMap *ta = nullptr, *tb = nullptr;
  CudaContext(half *a, half *b, int m, int n, int k, cudaStream_t stream) {
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublas initialization");
    ta = allocate_and_create_tensor_map(a, m / 128, k / 64);
    tb = allocate_and_create_tensor_map(b, n / 128, k / 64);
  }
  ~CudaContext() {
    if (ta)
      cudaFree(ta);
    if (tb)
      cudaFree(tb);
    cublasDestroy(blas);
  }
};
template <int Kind, int S, bool Swizzle>
void cuda_launch(half *a, half *b, half *c, int m, int n, int k,
                 cudaStream_t stream, CudaContext &ctx) {
  int z = Swizzle ? (n + 2047) / 2048 : 1;
  dim3 grid(((n + 127) / 128 + z - 1) / z, m / 128, z);
  if constexpr (Kind == 0) {
    constexpr int bytes = S * 256 * 16 * 2;
    auto kernel = hgemm_mma_stages_tn<16, 8, 16, 2, 4, 4, 4, S, Swizzle>;
    check(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<grid, 256, bytes, stream>>>(a, b, c, m, n, k);
  } else if constexpr (Kind == 1) {
    constexpr int bytes = S * 256 * 64 * 2;
    auto kernel =
        hgemm_mma_stages_tn_swizzle<16, 8, 16, 2, 4, 4, 4, 4, S, Swizzle>;
    check(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<grid, 256, bytes, stream>>>(a, b, c, m, n, k);
  } else if constexpr (Kind == 2) {
    leetcuda_test_stream = stream;
    launch_hgemm_mma_stages_tn_cute<half, S, Swizzle, LEETCUDA_REFERENCE_F32>(
        a, b, c, m, n, k);
  } else if constexpr (Kind == 3) {
    // SM120 cannot execute Hopper WGMMA. This is explicitly a mathematical
    // GPU reference, never described as execution of the original WGMMA code.
    float alpha = 1, beta = 0;
    auto status =
        cublasGemmEx(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b,
                     CUDA_R_16F, k, a, CUDA_R_16F, k, &beta, c, CUDA_R_16F, n,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublas FP32 reference");
  } else {
    constexpr int bytes = S * 256 * 64 * 2;
    auto kernel =
        hgemm_tma_mma_ws_tn<16, 8, 16, 2, 2, 4, 8, 4, S, 256, Swizzle>;
    check(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<grid, 256, bytes, stream>>>(m, n, k, c, ctx.ta, ctx.tb);
  }
}
#endif
std::vector<double> oracle(const std::vector<half> &a,
                           const std::vector<half> &b, int m, int n, int k,
                           bool half_state = false) {
  std::vector<double> result(m * n, 0);
  for (int row = 0; row < m; row++)
    for (int col = 0; col < n; col++) {
      double sum = 0;
      for (int start = 0; start < k; start += 16) {
        for (int kk = start; kk < std::min(k, start + 16); kk++)
          sum += double(__half2float(a[row * k + kk])) *
                 __half2float(b[col * k + kk]);
        if (half_state)
          sum = __half2float(__float2half(float(sum)));
      }
      result[row * n + col] = __half2float(__float2half(float(sum)));
    }
  return result;
}
template <int Kind, int S, bool F32 = true>
void matrix(Harness &h, const std::vector<float> &fixture, int m, int n, int k,
            int seed, bool swizzle = false,
            const std::string &prefix = "shared", bool state = false,
            bool unaligned = false) {
  std::vector<half> a(m * k), b(n * k);
  for (int i = 0; i < m * k; i++)
    a[i] = __float2half(
        state ? 1.f : fixture[(i + seed * 8192) % fixture.size()] * .5f);
  for (int i = 0; i < n * k; i++)
    b[i] = __float2half(state ? (i % k < 16 ? .0625f : 0x1p-15f)
                              : fixture[(fixture.size() - 1 -
                                         (i + seed * 8192) % fixture.size())] *
                                    .5f);
  auto ref = oracle(a, b, m, n, k, !F32);
  Buffer<half> da(a.size() + int(unaligned)), db(b.size() + int(unaligned));
  auto aa = a, bb = b;
  if (unaligned) {
    aa.insert(aa.begin(), __float2half(91));
    bb.insert(bb.begin(), __float2half(91));
  }
  da.upload(aa);
  db.upload(bb);
  half *ap = da.p + int(unaligned), *bp = db.p + int(unaligned);
  std::string id = prefix + "/kind" + std::to_string(Kind) + "/s" +
                   std::to_string(S) + "/f32" + std::to_string(F32) + "/m" +
                   std::to_string(m) + "/n" + std::to_string(n) + "/k" +
                   std::to_string(k) + "/seed" + std::to_string(seed) + "/sw" +
                   std::to_string(swizzle);
#if !defined(LEETCUDA_ROCM)
  CudaContext context(ap, bp, m, n, k, h.stream);
#endif
  halves(h, id, ref, [&](half *out) {
#if defined(LEETCUDA_ROCM)
    check(hg::launch<Kind, S, F32>(
        reinterpret_cast<_Float16 *>(ap), reinterpret_cast<_Float16 *>(bp),
        reinterpret_cast<_Float16 *>(out), m, n, k, swizzle, h.stream));
#else
    if(swizzle)cuda_launch<Kind,S,true>(ap,bp,out,m,n,k,h.stream,context);
    else cuda_launch<Kind,S,false>(ap,bp,out,m,n,k,h.stream,context);
#endif
  });
}
template <int Kind, int S>
void shared(Harness &h, const std::vector<float> &f) {
  for (int seed = 0; seed < 2; seed++)
    for (auto shape : {std::vector<int>{128, 256, 64},
                       {128, 256, 192},
                       {256, 256, 512},
                       {128, 256, 1024},
                       {128, 256, 2048},
                       {128, 2304, 64}}) {
      if constexpr (Kind == 1 && S == 3)
        shape[2] = std::max(shape[2], 128);
      matrix<Kind, S>(h, f, shape[0], shape[1], shape[2], seed);
      if (shape[1] > 2048)
        matrix<Kind, S>(h, f, shape[0], shape[1], shape[2], seed, true);
    }
}
#if defined(LEETCUDA_ROCM)
template <int Kind, int S> void extra(Harness &h, const std::vector<float> &f) {
  using T = hg::Config<Kind, S>;
  std::vector<int> used(T::BM * T::BK, 0), usedb(T::BN * T::BK, 0);
  bool valid = true;
  for (int r = 0; r < T::BM; r++)
    for (int k = 0; k < T::BK; k++) {
      int i = T::index(r, k);
      if (i < 0 || i >= int(used.size()))
        valid = false;
      else
        used[i]++;
    }
  for (int r = 0; r < T::BN; r++)
    for (int k = 0; k < T::BK; k++) {
      int i = T::index(r, k);
      if (i < 0 || i >= int(usedb.size()))
        valid = false;
      else
        usedb[i]++;
    }
  for (int v : used)
    valid &= v == 1;
  for (int v : usedb)
    valid &= v == 1;
  h.record("hip/layout/kind" + std::to_string(Kind) + "/s" + std::to_string(S),
           std::vector<int>{int(valid)}, std::vector<double>{1}, 0, 0);
  matrix<Kind, S>(h, f, 65, 137, 80, 0, false, "hip/tail");
  matrix<Kind, S>(h, f, 1, 3, 17, 0, false, "hip/unaligned", false, true);
  matrix<Kind, S>(h, f, 16, 16, 1024, 0, false, "hip/fp32_state", true);
  matrix<Kind, S, false>(h, f, 16, 16, 1024, 0, false, "hip/fp16_state", true);
}
template <int Kind, int S> void audit(Harness &h) {
  using T = hg::Config<Kind, S>;
  int m = T::BM, n = T::BN, k = T::BK * 12;
  Buffer<_Float16> a(m * k), b(n * k), c(m * n);
  Buffer<unsigned> stats(16);
  a.upload(std::vector<_Float16>(a.n, 1));
  b.upload(std::vector<_Float16>(b.n, 1));
  stats.upload(std::vector<unsigned>(16, 0));
  check(hg::launch<Kind, S, true, true, true>(a.p, b.p, c.p, m, n, k, false,
                                              h.stream, stats.p));
  check(hipStreamSynchronize(h.stream));
  auto values = c.download();
  bool valid = std::all_of(values.begin(), values.end(),
                           [&](auto v) { return float(v) == k; });
  auto actual = stats.download();
  std::vector<double> ref(16, 0);
  ref[0] = 12;
  for (int w = 0; w < 4; w++)
    ref[w + 1] = 12;
  ref[5] = S;
  for (int p = 0; p < T::Producers; p++)
    ref[8 + p] = 12;
  h.record("hip/pipeline/kind" + std::to_string(Kind) + "/s" +
               std::to_string(S),
           actual, ref, 0, 0, valid);
}
#endif
int main(int argc, char **argv) {
  try {
    if (argc < 4)
      throw std::runtime_error("device fixtures outputs [--performance]");
    check(GPU(SetDevice)(std::stoi(argv[1])));
    auto fixture = read_fixture(argv[2]);
    Harness h(argv[3], argc > 4 && std::string(argv[4]) == "--performance");
    shared<0, 2>(h, fixture);
    shared<0, 3>(h, fixture);
    shared<0, 4>(h, fixture);
    shared<1, 2>(h, fixture);
    shared<1, 3>(h, fixture);
    shared<2, 2>(h, fixture);
    shared<2, 3>(h, fixture);
    shared<3, 2>(h, fixture);
    shared<3, 3>(h, fixture);
    shared<3, 4>(h, fixture);
    shared<4, 2>(h, fixture);
    shared<4, 3>(h, fixture);
#if defined(LEETCUDA_ROCM)
#define EXTRA(K)                                                               \
  extra<K, 2>(h, fixture);                                                     \
  extra<K, 3>(h, fixture);                                                     \
  extra<K, 4>(h, fixture)
    EXTRA(0);
    EXTRA(1);
    EXTRA(2);
    EXTRA(3);
    EXTRA(4);
#undef EXTRA
    audit<3, 2>(h);
    audit<3, 3>(h);
    audit<3, 4>(h);
    audit<4, 2>(h);
    audit<4, 3>(h);
    audit<4, 4>(h);
#endif
    return h.passed ? 0 : 1;
  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
