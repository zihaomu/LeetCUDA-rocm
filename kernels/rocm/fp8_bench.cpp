// Standalone HIP qualification of the original interview FP8 header API.
#include "interview/fp8_gemm.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <hipblas/hipblas.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace fp8_gemm;
using BF16 = hip_bfloat16;

void check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
void check_blas(hipblasStatus_t e) {
  if (e != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("hipBLAS status " + std::to_string(e));
}
template <typename T> struct Buffer {
  T *p;
  size_t count;
  explicit Buffer(size_t n) : count(n) { check(hipMalloc(&p, n * sizeof(T))); }
  ~Buffer() { (void)hipFree(p); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void upload(const std::vector<T> &v) {
    check(hipMemcpy(p, v.data(), count * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> v(count);
    check(hipMemcpy(v.data(), p, count * sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};

// Independent scalar definition: no HIP FP8 conversion or production reducer.
float decode(uint8_t code) {
  const int x = code & 127;
#if defined(LEETCUDA_FP8_E5M2)
  const float v = x < 4 ? std::ldexp(float(x), -16)
                        : std::ldexp(float(4 + x % 4), x / 4 - 17);
#else
  const float v = x < 8 ? std::ldexp(float(x), -9)
                        : std::ldexp(float(8 + x % 8), x / 8 - 10);
#endif
  return (code & 128) ? -v : v;
}
uint8_t encode(float x) {
  const float magnitude = std::min(std::abs(x), kFP8Max);
#if defined(LEETCUDA_FP8_E5M2)
  int lo = 0, hi = 123;
#else
  int lo = 0, hi = 126;
#endif
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    const float edge = (decode(mid) + decode(mid + 1)) / 2;
    if (magnitude > edge || (magnitude == edge && (mid & 1)))
      lo = mid + 1;
    else
      hi = mid;
  }
  return uint8_t(lo | (std::signbit(x) ? 128 : 0));
}
struct Quantized {
  std::vector<uint8_t> code;
  std::vector<float> scale;
};
Quantized quantize(const std::vector<BF16> &input, int rows, int k,
                   bool individual, bool transpose) {
  Quantized q{std::vector<uint8_t>(size_t(rows) * k), {}};
  const int group = individual ? 1 : 128;
  auto read = [&](int r, int c) {
    return float(input[transpose ? size_t(c) * rows + r : size_t(r) * k + c]);
  };
  for (int start = 0; start < rows; start += group) {
    float amax = 0;
    for (int r = start; r < std::min(start + group, rows); ++r)
      for (int c = 0; c < k; ++c)
        amax = std::max(amax, std::abs(read(r, c)));
    q.scale.push_back(amax / kFP8Max);
    const float inv = amax == 0 ? 0 : kFP8Max / amax;
    for (int r = start; r < std::min(start + group, rows); ++r)
      for (int c = 0; c < k; ++c)
        q.code[size_t(r) * k + c] = encode(read(r, c) * inv);
  }
  return q;
}
double validate(const std::vector<BF16> &out, const Quantized &a,
                const Quantized &bt, int m, int n, int k, bool row, bool col,
                bool sampled) {
  const size_t count = sampled ? 64 : out.size();
  double maximum = 0;
  for (size_t i = 0; i < count; ++i) {
    const size_t index = sampled ? i * (out.size() - 1) / (count - 1) : i;
    const int r = index / n, c = index % n;
    double sum = 0;
    for (int j = 0; j < k; ++j)
      sum += double(decode(a.code[size_t(r) * k + j])) *
             decode(bt.code[size_t(c) * k + j]);
    const double expected = float(BF16(
        float(sum * a.scale[row ? r : r / 128] * bt.scale[col ? c : c / 128])));
    const double error = std::abs(float(out[index]) - expected);
    maximum = std::max(maximum, error);
    if (!std::isfinite(float(out[index])) ||
        error > std::ldexp(1., -10) + std::ldexp(1., -7) * std::abs(expected))
      throw std::runtime_error("FP64 quantized reference mismatch at " +
                               std::to_string(index));
  }
  return maximum;
}

template <typename T = Fp8GemmTraits<>>
void offline(bool row, bool col, bool ws, const BF16 *a, const uint8_t *bt,
             const float *sb, BF16 *c, int m, int n, int k,
             Fp8GemmActivation &act, hipStream_t stream) {
#define CALL(R, C)                                                             \
  if (ws)                                                                      \
    fp8_gemm_bf16_b_offline<R, C, true, T>(a, bt, sb, c, m, n, k, act,         \
                                           stream);                            \
  else                                                                         \
    fp8_gemm_bf16_b_offline<R, C, false, T>(a, bt, sb, c, m, n, k, act, stream)
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

struct Graph {
  hipGraph_t graph;
  hipGraphExec_t executable;
  hipEvent_t start, stop;
  hipStream_t stream;
  explicit Graph(const std::function<void()> &fn, hipStream_t s) : stream(s) {
    for (int i = 0; i < 5; ++i)
      fn();
    check(hipStreamSynchronize(stream));
    check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int i = 0; i < 32; ++i)
      fn();
    check(hipStreamEndCapture(stream, &graph));
    check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    check(hipEventCreate(&start));
    check(hipEventCreate(&stop));
  }
  ~Graph() {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    (void)hipGraphExecDestroy(executable);
    (void)hipGraphDestroy(graph);
  }
  float measure() {
    check(hipGraphLaunch(executable, stream));
    check(hipEventRecord(start, stream));
    check(hipGraphLaunch(executable, stream));
    check(hipEventRecord(stop, stream));
    check(hipEventSynchronize(stop));
    float probe;
    check(hipEventElapsedTime(&probe, start, stop));
    const int repeats =
        std::max(1, int(std::ceil(40.0f / std::max(probe, 1e-4f))));
    check(hipEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i)
      check(hipGraphLaunch(executable, stream));
    check(hipEventRecord(stop, stream));
    check(hipEventSynchronize(stop));
    float ms;
    check(hipEventElapsedTime(&ms, start, stop));
    return ms / (32 * repeats);
  }
};

void run_shape(int m, int n, int k, bool bench, hipStream_t stream,
               hipblasHandle_t handle) {
  std::mt19937 gen(20261103);
  std::uniform_real_distribution<float> dist(-0.5, 0.5);
  std::vector<BF16> ha(size_t(m) * k), hb(size_t(k) * n);
  for (auto &v : ha)
    v = BF16(dist(gen));
  for (auto &v : hb)
    v = BF16(dist(gen));
  Buffer<BF16> a(ha.size()), b(hb.size()), c(size_t(m) * n), ref(size_t(m) * n);
  a.upload(ha);
  b.upload(hb);
  Buffer<uint8_t> workspace(fp8_gemm_workspace_size(m, n, k));
  Buffer<uint8_t> activation(fp8_gemm_activation_size(m, k));
  Buffer<uint8_t> a8(size_t(m) * k), bt8(size_t(n) * k);
  Buffer<float> sa(m), sb(n);
  Fp8GemmWorkspace ws{workspace.p, workspace.count};
  Fp8GemmActivation act{activation.p, activation.count};
  const float alpha = 1, beta = 0;
  auto baseline = [&]() {
    check_blas(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                             &alpha, b.p, HIPBLAS_R_16B, n, a.p, HIPBLAS_R_16B,
                             k, &beta, ref.p, HIPBLAS_R_16B, n,
                             HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  };
  baseline();
  check(hipStreamSynchronize(stream));
  const auto baseline_values = ref.download();
  const size_t baseline_count = bench ? 64 : baseline_values.size();
  double baseline_error = 0;
  for (size_t i = 0; i < baseline_count; ++i) {
    const size_t index = bench ? i * (baseline_values.size() - 1) / 63 : i;
    const int r = index / n, col = index % n;
    double expected = 0;
    for (int j = 0; j < k; ++j)
      expected +=
          double(float(ha[size_t(r) * k + j])) * float(hb[size_t(j) * n + col]);
    const double error = std::abs(float(baseline_values[index]) - expected);
    baseline_error = std::max(baseline_error, error);
    if (!std::isfinite(float(baseline_values[index])) ||
        error > std::ldexp(1., -10) + std::ldexp(1., -7) * std::abs(expected))
      throw std::runtime_error(
          "hipBLAS baseline differs from CPU FP64 reference");
  }
  std::cout << std::defaultfloat << std::setprecision(10)
            << "{\"kind\":\"baseline_check\",\"shape_mnk\":[" << m << ',' << n
            << ',' << k << "],\"checked_outputs\":" << baseline_count
            << ",\"max_abs_error\":" << baseline_error << ",\"passed\":true}\n";
  for (int mode = 0; mode < 4; ++mode) {
    const bool row = mode < 2, col = mode % 2 == 0;
    const Quantized qa = quantize(ha, m, k, row, false),
                    qb = quantize(hb, n, k, col, true);
    quantize_a(a.p, a8.p, sa.p, m, k, row, stream);
    fp8_gemm_quantize_b(b.p, bt8.p, sb.p, n, k, col, stream);
    check(hipStreamSynchronize(stream));
    if (a8.download() != qa.code || bt8.download() != qb.code)
      throw std::runtime_error(
          "quantization differs from independent CPU codes");
    std::array<float, 5> preparation_samples = {};
    float preparation_ms = 0;
    if (bench) {
      Graph preparation(
          [&]() { fp8_gemm_quantize_b(b.p, bt8.p, sb.p, n, k, col, stream); },
          stream);
      for (int sample = 0; sample < 5; ++sample)
        preparation_samples[sample] = preparation.measure();
      auto ordered = preparation_samples;
      std::sort(ordered.begin(), ordered.end());
      preparation_ms = ordered[2];
    }
    for (bool use_ws : {false, true}) {
      std::array<std::function<void()>, 3> functions = {
          [&]() {
            dispatch_gemm(a8.p, bt8.p, sa.p, sb.p, c.p, m, n, k, row, col,
                          use_ws, stream);
          },
          [&]() {
            fp8_gemm_bf16(a.p, b.p, c.p, m, n, k, Fp8GemmScaleMode(mode), ws,
                          stream, use_ws);
          },
          [&]() {
            offline(row, col, use_ws, a.p, bt8.p, sb.p, c.p, m, n, k, act,
                    stream);
          }};
      const char *names[] = {"quantized", "full", "offline_b"};
      for (int path = 0; path < 3; ++path) {
        functions[path]();
        check(hipStreamSynchronize(stream));
        const auto result = c.download();
        const double error = validate(result, qa, qb, m, n, k, row, col, bench);
        std::cout << std::defaultfloat << std::setprecision(10)
                  << "{\"kind\":\"check\",\"shape_mnk\":[" << m << ',' << n
                  << ',' << k << "],\"mode\":" << mode
                  << ",\"use_ws\":" << use_ws << ",\"path\":\"" << names[path]
                  << "\",\"checked_outputs\":" << (bench ? 64 : m * n)
                  << ",\"max_abs_error\":" << error << ",\"passed\":true}\n";
        if (bench) {
          Graph ours(functions[path], stream), base(baseline, stream);
          check(hipMemsetAsync(c.p, 0xff, c.count * sizeof(BF16), stream));
          check(hipGraphLaunch(ours.executable, stream));
          check(hipStreamSynchronize(stream));
          const auto replay = c.download();
          if (!std::equal(result.begin(), result.end(), replay.begin(),
                          [](BF16 x, BF16 y) { return float(x) == float(y); }))
            throw std::runtime_error("FP8 graph replay failed");
          std::array<float, 5> ot, bt;
          for (int t = 0; t < 5; ++t) {
            if (t % 2) {
              bt[t] = base.measure();
              ot[t] = ours.measure();
            } else {
              ot[t] = ours.measure();
              bt[t] = base.measure();
            }
          }
          // A linked library can change iostream flags. Keep sub-us precision.
          std::cout << std::defaultfloat << std::setprecision(10)
                    << "{\"kind\":\"performance\",\"graph_replay_verified\":"
                       "true,\"shape_mnk\":["
                    << m << ',' << n << ',' << k
                    << "],\"offline_b_preparation_ms\":" << preparation_ms
                    << ",\"offline_b_preparation_samples_ms\":[";
          for (int sample = 0; sample < 5; ++sample)
            std::cout << (sample ? "," : "") << preparation_samples[sample];
          std::cout << "],\"mode\":" << mode << ",\"use_ws\":" << use_ws
                    << ",\"path\":\"" << names[path]
                    << "\",\"ours_samples_ms\":[";
          for (int t = 0; t < 5; ++t)
            std::cout << (t ? "," : "") << ot[t];
          std::cout << "],\"baseline_samples_ms\":[";
          for (int t = 0; t < 5; ++t)
            std::cout << (t ? "," : "") << bt[t];
          std::sort(ot.begin(), ot.end());
          std::sort(bt.begin(), bt.end());
          std::cout << "],\"ours_ms\":" << ot[2] << ",\"baseline_ms\":" << bt[2]
                    << ",\"baseline_over_ours\":" << bt[2] / ot[2] << "}\n";
        }
      }
      if (!bench && mode == 0) {
        // Non-default tile sizes and ring depths must affect real computation.
        for (int variant = 0; variant < 3; ++variant) {
          if (variant == 0)
            offline<Fp8GemmTraits<32, 64, 128, 2>>(
                row, col, use_ws, a.p, bt8.p, sb.p, c.p, m, n, k, act, stream);
          if (variant == 1)
            offline<Fp8GemmTraits<64, 32, 64, 3>>(
                row, col, use_ws, a.p, bt8.p, sb.p, c.p, m, n, k, act, stream);
          if (variant == 2)
            offline<Fp8GemmTraits<64, 64, 64, 4>>(
                row, col, use_ws, a.p, bt8.p, sb.p, c.p, m, n, k, act, stream);
          check(hipStreamSynchronize(stream));
          validate(c.download(), qa, qb, m, n, k, row, col, false);
          std::cout << "{\"kind\":\"tile_check\",\"variant\":" << variant
                    << ",\"use_ws\":" << use_ws << ",\"shape_mnk\":[" << m
                    << ',' << n << ',' << k << "],\"passed\":true}\n";
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  try {
    check(hipSetDevice(argc > 1 ? std::stoi(argv[1]) : 0));
    hipStream_t stream;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipblasHandle_t handle;
    check_blas(hipblasCreate(&handle));
    check_blas(hipblasSetStream(handle, stream));
    for (auto shape :
         {std::array<int, 3>{1, 8, 16}, {129, 136, 272}, {33, 72, 576}})
      run_shape(shape[0], shape[1], shape[2], false, stream, handle);
    if (argc > 2 && std::string(argv[2]) == "--performance")
      for (int n : {512, 2048})
        run_shape(n, n, n, true, stream, handle);
    check_blas(hipblasDestroy(handle));
    check(hipStreamDestroy(stream));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
