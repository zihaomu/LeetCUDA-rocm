#include "bench_runtime.hpp"
#include "interview/sgemm.cuh"
#include <bit>
using namespace leetcuda_bench;

// Independent arithmetic oracle: round magnitude on the exponent's TF32
// grid, ties away from zero. No GPU bit-mask conversion code is reused.
float cpu_tf32(float value) {
  if (value == 0 || !std::isfinite(value))
    return value;
  int exponent = 0;
  std::frexp(std::abs(double(value)), &exponent);
  int shift = std::max(exponent - 11, -136);
  double rounded = std::ldexp(
      std::floor(std::ldexp(std::abs(double(value)), -shift) + .5), shift);
  return float(std::copysign(rounded, value));
}
std::vector<double> product(const std::vector<float> &a,
                            const std::vector<float> &b, int m, int n, int k) {
  std::vector<double> c(m * n, 0);
  for (int r = 0; r < m; r++)
    for (int kk = 0; kk < k; kk++)
      for (int col = 0; col < n; col++)
        c[r * n + col] += double(a[r * k + kk]) * b[kk * n + col];
  return c;
}
void conversion(Harness &h, const std::vector<float> &x,
                const std::string &id) {
  Buffer<float> dx(x.size());
  dx.upload(x);
  std::vector<double> ref;
  for (float v : x)
    ref.push_back(cpu_tf32(v));
  h.floats(
      id, ref,
      [&](float *out) {
        f32x4_tf32x4_kernel<<<(x.size() + 1023) / 1024, 256, 0, h.stream>>>(
            dx.p, out, x.size());
      },
      0, 0);
  // Separate in-place contract and sign/zero bit check.
  f32x4_tf32x4_kernel<<<(x.size() + 1023) / 1024, 256, 0, h.stream>>>(
      dx.p, dx.p, x.size());
  check(GPU(GetLastError)());
  check(GPU(StreamSynchronize)(h.stream));
  auto values = dx.download();
  std::vector<uint32_t> bits;
  std::vector<double> ref_bits;
  for (size_t i = 0; i < x.size(); i++) {
    bits.push_back(std::bit_cast<uint32_t>(values[i]));
    ref_bits.push_back(std::bit_cast<uint32_t>(cpu_tf32(x[i])));
  }
  h.record(id + "/inplace_bits", bits, ref_bits, 0, 0);
}
void matrix(Harness &h, const std::vector<float> &fixture, int m, int n, int k,
            int seed, bool tf32) {
  auto a = slice(fixture, seed * 8192, m * k);
  auto b = slice(fixture, seed * 8192, k * n);
  // Distinct A/B contents from the frozen fixture, with deterministic B reversal.
  std::reverse(b.begin(), b.end());
  Buffer<float> da(a.size()), db(b.size());
  da.upload(a);
  db.upload(b);
  std::string id = "/m" + std::to_string(m) + "/n" + std::to_string(n) + "/k" +
                   std::to_string(k) + "/seed" + std::to_string(seed);
  if (tf32) {
    f32x4_tf32x4_kernel<<<(a.size() + 1023) / 1024, 256, 0, h.stream>>>(
        da.p, da.p, a.size());
    f32x4_tf32x4_kernel<<<(b.size() + 1023) / 1024, 256, 0, h.stream>>>(
        db.p, db.p, b.size());
    for (float &x : a)
      x = cpu_tf32(x);
    for (float &x : b)
      x = cpu_tf32(x);
    check(GPU(StreamSynchronize)(h.stream));
    h.record("tf32_input_a" + id, da.download(),
             std::vector<double>(a.begin(), a.end()), 0, 0);
    h.record("tf32_input_b" + id, db.download(),
             std::vector<double>(b.begin(), b.end()), 0, 0);
    auto ref = product(a, b, m, n, k);
    h.floats(
        "sgemm_tf32" + id, ref,
        [&](float *out) {
          constexpr int bytes = 2 * (128 * 12 + 8 * 132) * sizeof(float);
          sgemm_tf32<><<<dim3(n / 128, m / 128), 256, bytes, h.stream>>>(
              da.p, db.p, out, m, n, k);
        },
        2e-4, 2e-5);
  } else {
    auto ref = product(a, b, m, n, k);
    h.floats(
        "sgemm" + id, ref,
        [&](float *out) {
          sgemm<<<dim3(n / 32, m / 32), dim3(32, 32), 0, h.stream>>>(
              da.p, db.p, out, m, n, k);
        },
        2e-4, 2e-5);
    if (m % 128 == 0 && n % 128 == 0)
      h.floats(
          "sgemm_vec4" + id, ref,
          [&](float *out) {
            sgemm_vec4<<<dim3(n / 128, m / 128), dim3(32, 32), 0, h.stream>>>(
                da.p, db.p, out, m, n, k);
          },
          2e-4, 2e-5);
  }
}
int main(int argc, char **argv) {
  try {
    if (argc < 4)
      throw std::runtime_error("device fixture output [--performance]");
    check(GPU(SetDevice)(std::stoi(argv[1])));
    auto fixture = read_fixture(argv[2]);
    Harness h(argv[3], argc > 4 && std::string(argv[4]) == "--performance");
    for (int seed = 0; seed < 3; seed++) {
      for (int n : {4, 260, 8192})
        conversion(h, slice(fixture, seed * 8192, n),
                   "conversion/n" + std::to_string(n) + "/seed" +
                       std::to_string(seed));
      for (auto shape : {std::vector<int>{32, 32, 32},
                         {64, 96, 64},
                         {128, 128, 128},
                         {256, 128, 256},
                         {128, 256, 512},
                         {128, 128, 1024},
                         {256, 256, 64}})
        matrix(h, fixture, shape[0], shape[1], shape[2], seed, false);
      for (auto shape : {std::vector<int>{128, 128, 8},
                         {128, 128, 16},
                         {128, 128, 24},
                         {128, 128, 32},
                         {256, 128, 64},
                         {128, 256, 128},
                         {128, 128, 512},
                         {128, 128, 1024},
                         {256, 256, 32}})
        matrix(h, fixture, shape[0], shape[1], shape[2], seed, true);
    }
    std::vector<float> edges;
    for (int sign : {0, 1})
      for (unsigned exponent = 0; exponent < 254; exponent++)
        for (unsigned mantissa : {0u, 0xfffu, 0x1000u, 0x1001u, 0x1fffu,
                                  0x2000u, 0x7fefffu, 0x7ff000u})
          edges.push_back(std::bit_cast<float>((unsigned(sign) << 31) |
                                               (exponent << 23) | mantissa));
    conversion(h, edges, "conversion/exponent_boundaries_ties");
    return h.passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
