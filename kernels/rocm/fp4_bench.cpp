// Standalone validation of the original FP4 header's AMD branch.
#include "interview/fp4_gemm.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <hipblas/hipblas.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
using namespace fp4_gemm;
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

float decode8(uint8_t code) {
  const int x = code & 127;
  return x < 8 ? std::ldexp(float(x), -9)
               : std::ldexp(float(8 + x % 8), x / 8 - 10);
}
uint8_t encode8(float x) {
  const float value = std::min(std::abs(x), 448.f);
  int lo = 0, hi = 126;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    const float edge = (decode8(mid) + decode8(mid + 1)) / 2;
    if (value > edge || (value == edge && (mid & 1)))
      lo = mid + 1;
    else
      hi = mid;
  }
  return uint8_t(lo);
}
constexpr float values4[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
uint8_t encode4(float x) {
  const float value = std::min(std::abs(x), 6.f);
  int lo = 0;
  while (lo < 7) {
    const float edge = (values4[lo] + values4[lo + 1]) / 2;
    if (value < edge || (value == edge && !(lo & 1)))
      break;
    ++lo;
  }
  return uint8_t(lo | (std::signbit(x) ? 8 : 0));
}
struct Quantized {
  std::vector<uint8_t> code, sf;
  std::vector<float> scale, decoded;
};
Quantized quantize(const std::vector<BF16> &input, int rows, int k, bool level1,
                   bool transpose) {
  Quantized q{std::vector<uint8_t>(size_t(rows) * (k / 2)),
              std::vector<uint8_t>(sf_buffer_bytes(rows, k), 0),
              std::vector<float>(rows, 1),
              std::vector<float>(size_t(rows) * k)};
  auto read = [&](int row, int col) {
    return float(
        input[transpose ? size_t(col) * rows + row : size_t(row) * k + col]);
  };
  for (int row = 0; row < rows; ++row) {
    if (level1) {
      float amax = 0;
      for (int col = 0; col < k; ++col)
        amax = std::max(amax, std::abs(read(row, col)));
      q.scale[row] = amax / 2688.f;
    }
    const float inv1 = q.scale[row] > 0 ? 1.f / q.scale[row] : 0.f;
    for (int group = 0; group < k / 16; ++group) {
      float values[16], amax = 0;
      for (int i = 0; i < 16; ++i) {
        values[i] = read(row, group * 16 + i) * inv1;
        amax = std::max(amax, std::abs(values[i]));
      }
      const auto sf = encode8(amax / 6.f);
      const float scale = decode8(sf), inv2 = scale > 0 ? 1.f / scale : 0.f;
      const size_t offset =
          ((((row / 128) * (k / 64) + group / 4) * 32 + row % 32) * 4 +
           (row % 128) / 32) *
              4 +
          group % 4;
      q.sf[offset] = sf;
      for (int i = 0; i < 16; ++i) {
        const uint8_t code = encode4(values[i] * inv2);
        q.code[size_t(row) * (k / 2) + group * 8 + i / 2] |= code
                                                             << ((i % 2) * 4);
        q.decoded[size_t(row) * k + group * 16 + i] =
            (code & 8 ? -1.f : 1.f) * values4[code & 7] * scale;
      }
    }
  }
  return q;
}
double validate(const std::vector<BF16> &actual, const Quantized &a,
                const Quantized &bt, int m, int n, int k, bool sampled) {
  const size_t count = sampled ? 64 : actual.size();
  double maximum = 0;
  for (size_t i = 0; i < count; ++i) {
    const size_t index = sampled ? i * (actual.size() - 1) / 63 : i;
    const int row = index / n, col = index % n;
    double expected = 0;
    for (int j = 0; j < k; ++j)
      expected += double(a.decoded[size_t(row) * k + j]) *
                  bt.decoded[size_t(col) * k + j];
    expected = float(BF16(float(expected * a.scale[row] * bt.scale[col])));
    const double error = std::abs(float(actual[index]) - expected);
    maximum = std::max(maximum, error);
    if (!std::isfinite(float(actual[index])) ||
        error > std::ldexp(1., -10) + std::ldexp(1., -7) * std::abs(expected))
      throw std::runtime_error(
          "FP4 differs from independent quantized FP64 reference");
  }
  return maximum;
}
template <typename T = Fp4GemmTraits<>>
void offline(bool l1a, bool l1b, bool ws, const BF16 *a, const uint8_t *bt,
             const uint8_t *sfb, const float *sb, BF16 *c, int m, int n, int k,
             Fp4GemmActivation &act, hipStream_t stream) {
#define CALL(A, B)                                                             \
  if (ws)                                                                      \
    fp4_gemm_bf16_b_offline<A, B, true, T>(a, bt, sfb, sb, c, m, n, k, act,    \
                                           stream);                            \
  else                                                                         \
    fp4_gemm_bf16_b_offline<A, B, false, T>(a, bt, sfb, sb, c, m, n, k, act,   \
                                            stream)
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

struct Graph {
  hipGraph_t graph;
  hipGraphExec_t executable;
  hipEvent_t start, stop;
  hipStream_t stream;
  int replays = 8;
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
    for (int i = 0; i < 8; ++i)
      check(hipGraphLaunch(executable, stream));
    const float estimate = measure();
    replays = std::clamp(int(std::ceil(40.f / (estimate * 32))), 1, 4096);
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
    for (int i = 0; i < replays; ++i)
      check(hipGraphLaunch(executable, stream));
    check(hipEventRecord(stop, stream));
    check(hipEventSynchronize(stop));
    float ms;
    check(hipEventElapsedTime(&ms, start, stop));
    return ms / (32 * replays);
  }
};

void run_shape(int m, int n, int k, bool bench, hipStream_t stream,
               hipblasHandle_t handle) {
  std::mt19937 gen(20261103);
  std::uniform_real_distribution<float> dist(-.5, .5);
  std::vector<BF16> ha(size_t(m) * k), hb(size_t(k) * n);
  for (auto &v : ha)
    v = BF16(dist(gen));
  for (auto &v : hb)
    v = BF16(dist(gen));
  Buffer<BF16> a(ha.size()), b(hb.size()), c(size_t(m) * n), ref(size_t(m) * n);
  a.upload(ha);
  b.upload(hb);
  Buffer<uint8_t> workspace(fp4_gemm_workspace_size(m, n, k)),
      activation(fp4_gemm_activation_size(m, k));
  Buffer<uint8_t> a4(size_t(m) * (k / 2)), bt4(size_t(n) * (k / 2)),
      sfa(sf_buffer_bytes(m, k)), sfb(sf_buffer_bytes(n, k));
  Buffer<float> sa(m), sb(n);
  Fp4GemmWorkspace ws{workspace.p, workspace.count};
  Fp4GemmActivation act{activation.p, activation.count};
  const float alpha = 1, beta = 0;
  auto baseline = [&]() {
    check_blas(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                             &alpha, b.p, HIPBLAS_R_16B, n, a.p, HIPBLAS_R_16B,
                             k, &beta, ref.p, HIPBLAS_R_16B, n,
                             HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  };
  baseline();
  check(hipStreamSynchronize(stream));
  const auto rv = ref.download();
  const size_t count = bench ? 64 : rv.size();
  for (size_t i = 0; i < count; ++i) {
    const size_t idx = bench ? i * (rv.size() - 1) / 63 : i;
    const int r = idx / n, col = idx % n;
    double expected = 0;
    for (int j = 0; j < k; ++j)
      expected +=
          double(float(ha[size_t(r) * k + j])) * float(hb[size_t(j) * n + col]);
    if (!std::isfinite(float(rv[idx])) ||
        std::abs(float(rv[idx]) - expected) >
            std::ldexp(1., -10) + std::ldexp(1., -7) * std::abs(expected))
      throw std::runtime_error(
          "hipBLAS baseline differs from CPU FP64 reference");
  }
  std::cout << "{\"kind\":\"baseline_check\",\"shape_mnk\":[" << m << ',' << n
            << ',' << k << "],\"checked_outputs\":" << count
            << ",\"passed\":true}\n";
  using Default = Fp4GemmTraits<>;
  using Wide = Fp4GemmTraits<256, 6>;
  for (int mode = 0; mode < 4; ++mode) {
    const bool l1a = mode & 1, l1b = mode & 2;
    const Quantized qa = quantize(ha, m, k, l1a, false),
                    qb = quantize(hb, n, k, l1b, true);
    quantize_a(a.p, a4.p, sfa.p, sa.p, m, k, l1a, stream);
    fp4_gemm_quantize_b(b.p, bt4.p, sfb.p, sb.p, n, k, l1b, stream);
    check(hipStreamSynchronize(stream));
    if (a4.download() != qa.code || bt4.download() != qb.code ||
        sfa.download() != qa.sf || sfb.download() != qb.sf)
      throw std::runtime_error(
          "FP4 bytes or canonical scales differ from CPU reference");
    for (bool wide : {false, true})
      for (bool use_ws : {false, true}) {
        std::array<std::function<void()>, 3> functions = {
            [&]() {
              if (wide)
                dispatch_gemm<Wide>(a4.p, bt4.p, sfa.p, sfb.p, sa.p, sb.p, c.p,
                                    m, n, k, l1a, l1b, use_ws, stream);
              else
                dispatch_gemm<Default>(a4.p, bt4.p, sfa.p, sfb.p, sa.p, sb.p,
                                       c.p, m, n, k, l1a, l1b, use_ws, stream);
            },
            [&]() {
              if (wide)
                fp4_gemm_bf16<Wide>(a.p, b.p, c.p, m, n, k,
                                    Fp4GemmScaleMode(mode), ws, stream, use_ws);
              else
                fp4_gemm_bf16<Default>(a.p, b.p, c.p, m, n, k,
                                       Fp4GemmScaleMode(mode), ws, stream,
                                       use_ws);
            },
            [&]() {
              if (wide)
                offline<Wide>(l1a, l1b, use_ws, a.p, bt4.p, sfb.p, sb.p, c.p, m,
                              n, k, act, stream);
              else
                offline<Default>(l1a, l1b, use_ws, a.p, bt4.p, sfb.p, sb.p, c.p,
                                 m, n, k, act, stream);
            }};
        const char *names[] = {"quantized", "full", "offline_b"};
        for (int path = 0; path < 3; ++path) {
          functions[path]();
          check(hipStreamSynchronize(stream));
          const double error = validate(c.download(), qa, qb, m, n, k, bench);
          std::cout << std::defaultfloat << std::setprecision(10)
                    << "{\"kind\":\"check\",\"shape_mnk\":[" << m << ',' << n
                    << ',' << k << "],\"mode\":" << mode << ",\"wide\":" << wide
                    << ",\"use_ws\":" << use_ws << ",\"path\":\"" << names[path]
                    << "\",\"checked_outputs\":" << (bench ? 64 : m * n)
                    << ",\"max_abs_error\":" << error << ",\"passed\":true}\n";
          if (bench) {
            Graph ours(functions[path], stream), base(baseline, stream);
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
            std::cout << std::defaultfloat << std::setprecision(10)
                      << "{\"kind\":\"performance\",\"shape_mnk\":[" << m << ','
                      << n << ',' << k << "],\"mode\":" << mode
                      << ",\"wide\":" << wide << ",\"use_ws\":" << use_ws
                      << ",\"path\":\"" << names[path]
                      << "\",\"ours_samples_ms\":[";
            for (int t = 0; t < 5; ++t)
              std::cout << (t ? "," : "") << ot[t];
            std::cout << "],\"baseline_samples_ms\":[";
            for (int t = 0; t < 5; ++t)
              std::cout << (t ? "," : "") << bt[t];
            std::sort(ot.begin(), ot.end());
            std::sort(bt.begin(), bt.end());
            std::cout << "],\"calls_per_graph\":32,\"ours_replays\":"
                      << ours.replays
                      << ",\"baseline_replays\":" << base.replays
                      << ",\"ours_ms\":" << ot[2]
                      << ",\"baseline_ms\":" << bt[2]
                      << ",\"baseline_over_ours\":" << bt[2] / ot[2] << "}\n";
          }
        }
        if (!bench && mode == 3 && !wide)
          for (int variant = 0; variant < 3; ++variant) {
            if (variant == 0)
              offline<Fp4GemmTraits<128, 2>>(true, true, use_ws, a.p, bt4.p,
                                             sfb.p, sb.p, c.p, m, n, k, act,
                                             stream);
            if (variant == 1)
              offline<Fp4GemmTraits<128, 3>>(true, true, use_ws, a.p, bt4.p,
                                             sfb.p, sb.p, c.p, m, n, k, act,
                                             stream);
            if (variant == 2)
              offline<Fp4GemmTraits<256, 4>>(true, true, use_ws, a.p, bt4.p,
                                             sfb.p, sb.p, c.p, m, n, k, act,
                                             stream);
            check(hipStreamSynchronize(stream));
            validate(c.download(), qa, qb, m, n, k, false);
            std::cout << "{\"kind\":\"stage_check\",\"variant\":" << variant
                      << ",\"use_ws\":" << use_ws << ",\"shape_mnk\":[" << m
                      << ',' << n << ',' << k << "],\"passed\":true}\n";
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
         {std::array<int, 3>{1, 8, 64}, {129, 136, 320}, {33, 72, 576}})
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
