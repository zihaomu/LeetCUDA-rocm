#include "bench_runtime.hpp"
#include "interview/sgemv.cuh"
using namespace leetcuda_bench;

template <int K>
void rows(Harness &h, const std::vector<float> &data, int offset, int seed) {
  constexpr int Rows = 7, N = Rows * K;
  auto x = slice(data, offset, N);
  Buffer<float> dx(N);
  dx.upload(x);
  std::string suffix =
      "/k" + std::to_string(K) + "/seed" + std::to_string(seed);
  std::vector<double> soft(N), rms(N), norm(N);
  constexpr float g = .75f, b = -.125f;
  for (int row = 0; row < Rows; row++) {
    double sum = 0, square = 0, expsum = 0;
    for (int c = 0; c < K; c++) {
      double value = x[row * K + c];
      sum += value;
      square += value * value;
      expsum += std::exp(value);
    }
    double mean = sum / K, variance = 0;
    for (int c = 0; c < K; c++) {
      double delta = x[row * K + c] - mean;
      variance += delta * delta;
    }
    for (int c = 0; c < K; c++) {
      double value = x[row * K + c];
      soft[row * K + c] = std::exp(value) / expsum;
      rms[row * K + c] = value / std::sqrt(square / K + 1e-5) * g;
      norm[row * K + c] =
          (value - mean) / std::sqrt(variance / K + 1e-5) * g + b;
    }
  }
  h.floats("softmax_per_token" + suffix, soft, [&](float *o) {
    softmax_per_token<K><<<Rows, K, 0, h.stream>>>(dx.p, o, N);
  });
  h.floats("safe_softmax_per_token" + suffix, soft, [&](float *o) {
    safe_softmax_per_token<K><<<Rows, K, 0, h.stream>>>(dx.p, o, N);
  });
  h.floats("online_safe_softmax_per_token" + suffix, soft, [&](float *o) {
    online_safe_softmax_per_token<K><<<Rows, K, 0, h.stream>>>(dx.p, o, N);
  });
  h.floats("rms_norm" + suffix, rms, [&](float *o) {
    rms_norm<K><<<Rows, K, 0, h.stream>>>(dx.p, o, g, Rows, K);
  });
  h.floats("rms_norm_vec4" + suffix, rms, [&](float *o) {
    rms_norm_vec4<K / 4><<<Rows, K / 4, 0, h.stream>>>(dx.p, o, g, Rows, K);
  });
  h.floats("layer_norm" + suffix, norm, [&](float *o) {
    layer_norm<K><<<Rows, K, 0, h.stream>>>(dx.p, o, g, b, Rows, K);
  });
  h.floats("layer_norm_vec4" + suffix, norm, [&](float *o) {
    layer_norm_vec4<K / 4>
        <<<Rows, K / 4, 0, h.stream>>>(dx.p, o, g, b, Rows, K);
  });
}
void vectors(Harness &h, const std::vector<float> &data, int n, int offset,
             int seed) {
  auto a = slice(data, offset, n), b = slice(data, offset + 32768, n);
  Buffer<float> da(n), db(n);
  da.upload(a);
  db.upload(b);
  std::vector<double> rel(n), add(n);
  double sum = 0, dot_ref = 0;
  for (int i = 0; i < n; i++) {
    rel[i] = std::max(0., double(a[i]));
    add[i] = double(a[i]) + double(b[i]);
    sum += a[i];
    dot_ref += double(a[i]) * b[i];
  }
  std::string s = "/n" + std::to_string(n) + "/seed" + std::to_string(seed);
  h.floats(
      "block_reduce_all" + s, {sum},
      [&](float *o) {
        check(GPU(MemsetAsync)(o, 0, sizeof(float), h.stream));
        block_reduce_all<256>
            <<<(n + 255) / 256, 256, 0, h.stream>>>(da.p, o, n);
      },
      2e-4, 2e-5);
  h.floats(
      "dot" + s, {dot_ref},
      [&](float *o) {
        check(GPU(MemsetAsync)(o, 0, sizeof(float), h.stream));
        dot<256><<<(n + 255) / 256, 256, 0, h.stream>>>(da.p, db.p, o, n);
      },
      2e-4, 2e-5);
  h.floats(
      "dot_vec4" + s, {dot_ref},
      [&](float *o) {
        check(GPU(MemsetAsync)(o, 0, sizeof(float), h.stream));
        dot_vec4<64><<<n / 256, 64, 0, h.stream>>>(da.p, db.p, o, n);
      },
      2e-4, 2e-5);
  h.floats(
      "relu" + s, rel,
      [&](float *o) {
        relu<<<(n + 255) / 256, 256, 0, h.stream>>>(da.p, o, n);
      },
      0, 0);
  h.floats(
      "relu_vec4" + s, rel,
      [&](float *o) {
        relu_vec4<<<(n + 1023) / 1024, 256, 0, h.stream>>>(da.p, o, n);
      },
      0, 0);
  h.floats(
      "elementwise_add" + s, add,
      [&](float *o) {
        elementwise_add<<<(n + 255) / 256, 256, 0, h.stream>>>(da.p, db.p, o,
                                                               n);
      },
      0, 0);
  h.floats(
      "elementwise_add_vec4" + s, add,
      [&](float *o) {
        elementwise_add_vec4<<<(n + 1023) / 1024, 256, 0, h.stream>>>(
            da.p, db.p, o, n);
      },
      0, 0);
  std::vector<int> bins(n), zeros(97, 0);
  std::vector<double> counts(97, 0);
  for (int i = 0; i < n; i++) {
    bins[i] = (int(std::ldexp(a[i], 15)) + 32768) % 97;
    counts[bins[i]]++;
  }
  Buffer<int> di(n), dh(97);
  di.upload(bins);
  dh.upload(zeros);
  histogram<<<(n + 255) / 256, 256, 0, h.stream>>>(di.p, dh.p, n);
  check(GPU(GetLastError)());
  check(GPU(StreamSynchronize)(h.stream));
  h.record("histogram" + s, dh.download(), counts, 0, 0, true, h.timed([&] {
    check(GPU(MemsetAsync)(dh.p, 0, 97 * sizeof(int), h.stream));
    histogram<<<(n + 255) / 256, 256, 0, h.stream>>>(di.p, dh.p, n);
  }));
}
void layouts(Harness &h, const std::vector<float> &data, int scale, int offset,
             int seed) {
  int m = 64 * scale, n = 16 * scale;
  auto x = slice(data, offset, m * n);
  Buffer<float> dx(x.size());
  dx.upload(x);
  std::vector<double> trans(x.size());
  for (int r = 0; r < m; r++)
    for (int c = 0; c < n; c++)
      trans[c * m + r] = x[r * n + c];
  std::string s =
      "/scale" + std::to_string(scale) + "/seed" + std::to_string(seed);
  h.floats(
      "mat_transpose" + s, trans,
      [&](float *o) {
        mat_transpose<<<dim3(n / 16, m / 16), dim3(16, 16), 0, h.stream>>>(
            dx.p, o, m, n);
      },
      0, 0);
  h.floats(
      "mat_transpose_padded" + s, trans,
      [&](float *o) {
        mat_transpose_padded<<<dim3(n / 16, m / 64), dim3(16, 16), 0,
                               h.stream>>>(dx.p, o, m, n);
      },
      0, 0);
  int seq = 16, pairs = 64 * scale, count = seq * pairs * 2;
  auto xr = slice(data, offset, count);
  Buffer<float> dr(count);
  dr.upload(xr);
  std::vector<double> rot(count);
  for (int row = 0; row < seq; row++)
    for (int c = 0; c < pairs; c++) {
      double theta = row / std::pow(10000., double(c) / pairs),
             co = std::cos(theta), si = std::sin(theta);
      int i = (row * pairs + c) * 2;
      rot[i] = xr[i] * co - xr[i + 1] * si;
      rot[i + 1] = xr[i] * si + xr[i + 1] * co;
    }
  h.floats("rope" + s, rot, [&](float *o) {
    rope<<<seq * pairs / 256, 256, 0, h.stream>>>(dr.p, o, seq, pairs);
  });
  int tokens = 17, heads = 3, d = 64 * scale, total = tokens * heads * d;
  auto prefix = slice(data, offset, total),
       suffix = slice(data, offset + 32768, total),
       pl = slice(data, offset + 65536, tokens * heads),
       sl = slice(data, offset + 65536 + 256, tokens * heads);
  // Each one-sided empty partition is part of the original infinity sentinel
  // contract.
  pl[0] = -INFINITY;
  sl[1] = INFINITY;
  Buffer<float> dp(total), ds(total), dpl(pl.size()), dsl(sl.size());
  dp.upload(prefix);
  ds.upload(suffix);
  dpl.upload(pl);
  dsl.upload(sl);
  std::vector<double> merged(total);
  for (int t = 0; t < tokens; t++)
    for (int head = 0; head < heads; head++) {
      double p =
          std::isinf(pl[head * tokens + t]) ? -INFINITY : pl[head * tokens + t];
      double s2 =
          std::isinf(sl[head * tokens + t]) ? -INFINITY : sl[head * tokens + t];
      double maximum = std::max(p, s2), a = std::exp(p - maximum),
             b = std::exp(s2 - maximum);
      for (int c = 0; c < d; c++) {
        int i = (t * heads + head) * d + c;
        merged[i] = (prefix[i] * a + suffix[i] * b) / (a + b);
      }
    }
  h.floats("merge_attn_states" + s, merged, [&](float *o) {
    merge_attn_states<<<(total / 4 + 127) / 128, 128, 0, h.stream>>>(
        o, dp.p, dpl.p, ds.p, dsl.p, tokens, heads, d);
  });
}
void gemv(Harness &h, const std::vector<float> &data, int scale, int offset,
          int seed) {
  int m = 16 * scale;
  for (int kind = 0; kind < 3; kind++) {
    int k = kind == 0 ? 32 * scale : kind == 1 ? 128 * scale : 16;
    auto a = slice(data, offset, m * k), x = slice(data, offset + 32768, k);
    Buffer<float> da(a.size()), dx(x.size());
    da.upload(a);
    dx.upload(x);
    std::vector<double> y(m);
    for (int row = 0; row < m; row++)
      for (int c = 0; c < k; c++)
        y[row] += double(a[row * k + c]) * x[c];
    std::string name = kind == 0   ? "sgemv_k32"
                       : kind == 1 ? "sgemv_k128"
                                   : "sgemv_k16";
    h.floats(
        name + "/scale" + std::to_string(scale) + "/seed" +
            std::to_string(seed),
        y,
        [&](float *o) {
          if (kind == 0)
            sgemv_k32<<<(m + 3) / 4, dim3(32, 4), 0, h.stream>>>(da.p, dx.p, o,
                                                                 m, k);
          else if (kind == 1)
            sgemv_k128<<<(m + 3) / 4, dim3(32, 4), 0, h.stream>>>(da.p, dx.p, o,
                                                                  m, k);
          else
            sgemv_k16<2><<<(m + 7) / 8, dim3(32, 4), 0, h.stream>>>(da.p, dx.p,
                                                                    o, m, k);
        },
        2e-4, 2e-5);
  }
}
int main(int argc, char **argv) {
  try {
    if (argc < 4)
      throw std::runtime_error(
          "usage: bench DEVICE FIXTURE OUTPUT [--performance]");
    check(GPU(SetDevice)(std::atoi(argv[1])));
    auto fixture = read_fixture(argv[2]);
    Harness h(argv[3], argc > 4);
    for (int seed = 0; seed < 2; seed++) {
      int offset = seed * 8192;
      for (int n : {256, 1024, 8192})
        vectors(h, fixture, n, offset, seed);
      rows<128>(h, fixture, offset, seed);
      rows<256>(h, fixture, offset, seed);
      rows<1024>(h, fixture, offset, seed);
      for (int scale : {1, 2, 3}) {
        layouts(h, fixture, scale, offset, seed);
        gemv(h, fixture, scale, offset, seed);
      }
    }
    return h.passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
