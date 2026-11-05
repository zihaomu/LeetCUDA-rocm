#include "bench_runtime.hpp"
#include "interview/flash_attn.cuh"
using namespace leetcuda_bench;
namespace fa = leetcuda_interview_fa;

template <int Kind, int D>
void test(Harness &h, const std::vector<float> &fixture, int n, int sk,
          int sv) {
  std::vector<_Float16> q(n * D), k(n * D), v(n * D);
  for (int i = 0; i < n * D; i++) {
    q[i] = _Float16(fixture[i % fixture.size()] * .5f);
    k[i] = _Float16(fixture[(i + 8192) % fixture.size()] * .5f);
    v[i] = _Float16(fixture[(fixture.size() - 1 - i % fixture.size())] * .5f);
  }
  std::vector<double> expected(n * D), expected_lse(n);
  float scale = 1.f / std::sqrt(float(D));
  for (int r = 0; r < n; r++) {
    std::vector<double> logits(n);
    double mx = -INFINITY, sum = 0;
    for (int s = 0; s < n; s++) {
      double dot = 0;
      for (int d = 0; d < D; d++) {
        float x = float(q[r * D + d]);
        if constexpr (Kind == 7)
          x = float(_Float16(x * (scale * 1.4426950408889634f)));
        dot += double(x) * float(k[s * D + d]);
      }
      logits[s] = dot * (Kind == 7 ? std::log(2.) : double(scale));
      mx = std::max(mx, logits[s]);
    }
    for (double &x : logits) {
      x = std::exp(x - mx);
      sum += x;
    }
    expected_lse[r] = mx + std::log(sum);
    for (int d = 0; d < D; d++)
      for (int s = 0; s < n; s++)
        expected[r * D + d] += logits[s] / sum * float(v[s * D + d]);
  }
  Buffer<_Float16> dq(q.size()), dk(k.size()), dv(v.size()), out(q.size() + 16);
  Buffer<float> lse(n + 16);
  dq.upload(q);
  dk.upload(k);
  dv.upload(v);
  std::vector<_Float16> poison(out.n, _Float16(NAN));
  std::vector<float> lp(lse.n, NAN);
  for (int i = 0; i < 8; i++) {
    poison[i] = poison[out.n - 1 - i] = _Float16(91);
    lp[i] = lp[lse.n - 1 - i] = 91;
  }
  auto launch = [&] {
    check(fa::launch<Kind, D>(dq.p, dk.p, dv.p, out.p + 8, lse.p + 8, 1, 1, 1,
                              n, n, sk, sv, Kind == 0 ? 7 : 0, false, scale,
                              h.stream, nullptr, 0, Kind == 7 ? 1 : 0));
  };
  std::string id = "kind" + std::to_string(Kind) + "/d" + std::to_string(D) +
                   "/n" + std::to_string(n) + "/s" + std::to_string(sk) + "v" +
                   std::to_string(sv);
  auto verify = [&](std::string suffix) {
    auto o = out.download();
    auto l = lse.download();
    bool canary = true;
    for (int i = 0; i < 8; i++)
      canary &= float(o[i]) == 91 && float(o[out.n - 1 - i]) == 91 &&
                l[i] == 91 && l[lse.n - 1 - i] == 91;
    std::vector<float> values(q.size());
    for (size_t i = 0; i < values.size(); i++)
      values[i] = float(o[i + 8]);
    h.record(id + suffix, values, expected, .003, .003, canary);
    h.record(id + suffix + "/lse",
             std::vector<float>(l.begin() + 8, l.end() - 8), expected_lse, 2e-5,
             2e-5, canary);
  };
  out.upload(poison);
  lse.upload(lp);
  launch();
  check(hipStreamSynchronize(h.stream));
  verify("");
  hipGraph_t graph;
  hipGraphExec_t exec;
  check(hipStreamBeginCapture(h.stream, hipStreamCaptureModeGlobal));
  launch();
  check(hipStreamEndCapture(h.stream, &graph));
  check(hipGraphInstantiateWithFlags(&exec, graph, 0));
  out.upload(poison);
  lse.upload(lp);
  check(hipGraphLaunch(exec, h.stream));
  check(hipStreamSynchronize(h.stream));
  verify("/graph");
  check(hipGraphExecDestroy(exec));
  check(hipGraphDestroy(graph));
}
template <int Kind, int D> void suite(Harness &h, const std::vector<float> &f) {
  for (int n : {65, 128})
    for (int sk = 1; sk <= (Kind == 7 ? 2 : 4); sk++)
      for (int sv = 1; sv <= ((Kind == 1 || Kind == 4 || Kind == 7) ? 2 : 1);
           sv++)
        test<Kind, D>(h, f, n, sk, sv);
}
template <int D> void copy(Harness &h) {
  for (int n : {17, 64, 137}) {
    Buffer<_Float16> q(n * D), o(n * D + 16);
    std::vector<_Float16> x(q.n), p(o.n, _Float16(91));
    std::vector<double> ref(q.n);
    for (size_t i = 0; i < x.size(); i++)
      ref[i] = float(x[i] = _Float16(int(i % 977) - 488));
    q.upload(x);
    o.upload(p);
    fa::copy_kernel<D><<<(n + 63) / 64, 128, 0, h.stream>>>(q.p, o.p + 8, n);
    check(hipGetLastError());
    check(hipStreamSynchronize(h.stream));
    auto values = o.download();
    bool canary = true;
    for (int i = 0; i < 8; i++)
      canary &= float(values[i]) == 91 && float(values[o.n - 1 - i]) == 91;
    std::vector<float> result(q.n);
    for (size_t i = 0; i < q.n; i++)
      result[i] = float(values[i + 8]);
    h.record("copy/d" + std::to_string(D) + "/n" + std::to_string(n), result,
             ref, 0, 0, canary);
  }
}
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: interview_attention_bench fixture.f32 outputs.f64");
    Harness h(argv[2], false);
    auto f = read_fixture(argv[1]);
    suite<0, 32>(h, f);
    suite<0, 64>(h, f);
    suite<0, 128>(h, f);
    suite<1, 64>(h, f);
    suite<1, 128>(h, f);
    suite<2, 64>(h, f);
    suite<2, 128>(h, f);
    suite<3, 64>(h, f);
    suite<3, 128>(h, f);
    suite<4, 64>(h, f);
    suite<4, 128>(h, f);
    suite<5, 64>(h, f);
    suite<5, 128>(h, f);
    suite<7, 64>(h, f);
    suite<7, 96>(h, f);
    suite<7, 128>(h, f);
    copy<64>(h);
    copy<128>(h);
    return h.passed ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
