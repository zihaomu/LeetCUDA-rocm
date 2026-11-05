// Independent C++ entry point: no Torch, Python, CuTe, CUDA, or NVIDIA runtime.
#include "interview/ffpa_attn.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
template <class T> struct Buffer {
  T *p = nullptr;
  size_t n;
  explicit Buffer(size_t size) : n(size) {
    check(hipMalloc(&p, std::max(size_t(1), n) * sizeof(T)));
  }
  ~Buffer() { (void)hipFree(p); }
  void upload(const std::vector<T> &v) {
    if (n)
      check(hipMemcpy(p, v.data(), n * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> download() {
    std::vector<T> v(n);
    if (n)
      check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};
int main(int argc, char **argv) {
  try {
    check(hipSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
    bool performance = argc > 2;
    std::mt19937 generator(20261106);
    std::uniform_real_distribution<float> random(-1, 1);
    hipStream_t stream;
    check(hipStreamCreate(&stream));
    bool all = true;
    for (int d : {64, 128, 192, 256, 320, 384, 448, 512, 1024}) {
      for (auto shape :
           std::vector<std::pair<int, int>>{{17, 79}, {128, 192}, {16, 0}}) {
        int nq = shape.first, nk = shape.second;
        std::vector<_Float16> q(nq * d), k(nk * d), v(nk * d);
        for (auto *x : {&q, &k, &v})
          for (auto &value : *x)
            value = (_Float16)random(generator);
        std::vector<double> expected(nq * d, 0), lse(nq, -INFINITY), scores(nk);
        for (int row = 0; row < nq; row++) {
          double maximum = -INFINITY;
          for (int key = 0; key < nk; key++) {
            double score = 0;
            for (int c = 0; c < d; c++)
              score += double(q[row * d + c]) * double(k[key * d + c]);
            scores[key] = score / std::sqrt(double(d));
            maximum = std::max(maximum, scores[key]);
          }
          double sum = 0;
          for (auto &score : scores) {
            score = std::exp(score - maximum);
            sum += score;
          }
          if (nk) {
            lse[row] = maximum + std::log(sum);
            for (int c = 0; c < d; c++)
              for (int key = 0; key < nk; key++)
                expected[row * d + c] +=
                    scores[key] / sum * double(v[key * d + c]);
          }
        }
        Buffer<_Float16> dq(q.size()), dk(k.size()), dv(v.size()),
            out(q.size() + 2);
        Buffer<float> dl(nq);
        dq.upload(q);
        dk.upload(k);
        dv.upload(v);
        for (int kind = 0; kind < 3; kind++)
          for (int sq = 1; sq <= 3; sq++)
            for (int sv = 1; sv <= 3; sv++) {
              std::vector<_Float16> poison(q.size() + 2, (_Float16)NAN);
              poison.front() = poison.back() = (_Float16)91;
              out.upload(poison);
              check(leetcuda_ffpa::forward(dq.p, dk.p, dv.p, out.p + 1, dl.p, 1,
                                           nq, nk, d, kind, sq, sv, stream));
              check(hipStreamSynchronize(stream));
              auto actual = out.download();
              auto logs = dl.download();
              bool passed = actual.front() == (_Float16)91 &&
                            actual.back() == (_Float16)91;
              double error = 0, le = 0;
              for (size_t i = 0; i < expected.size(); i++) {
                double delta = std::abs(double(actual[i + 1]) - expected[i]);
                error = std::max(error, delta);
                passed &= std::isfinite(double(actual[i + 1])) &&
                          delta <= .003 + .003 * std::abs(expected[i]);
              }
              for (int i = 0; i < nq; i++) {
                if (nk) {
                  double delta = std::abs(double(logs[i]) - lse[i]);
                  le = std::max(le, delta);
                  passed &= std::isfinite(logs[i]) &&
                            delta <= 2e-5 + 2e-5 * std::abs(lse[i]);
                } else
                  passed &= logs[i] == -INFINITY;
              }
              all &= passed;
              std::cout << "{\"kind\":\"correctness\",\"d\":" << d
                        << ",\"nq\":" << nq << ",\"nk\":" << nk
                        << ",\"variant\":" << kind << ",\"sq\":" << sq
                        << ",\"sv\":" << sv
                        << ",\"passed\":" << (passed ? "true" : "false")
                        << ",\"max_abs\":" << error << ",\"lse_error\":" << le
                        << "}\n";
              if (performance && nq == 128 && sq == 2 && sv == 2) {
                hipEvent_t start, stop;
                check(hipEventCreate(&start));
                check(hipEventCreate(&stop));
                for (int i = 0; i < 10; i++)
                  check(leetcuda_ffpa::forward(dq.p, dk.p, dv.p, out.p + 1,
                                               dl.p, 1, nq, nk, d, kind, sq, sv,
                                               stream));
                check(hipEventRecord(start, stream));
                for (int i = 0; i < 100; i++)
                  check(leetcuda_ffpa::forward(dq.p, dk.p, dv.p, out.p + 1,
                                               dl.p, 1, nq, nk, d, kind, sq, sv,
                                               stream));
                check(hipEventRecord(stop, stream));
                check(hipEventSynchronize(stop));
                float ms = 0;
                check(hipEventElapsedTime(&ms, start, stop));
                std::cout << "{\"kind\":\"performance\",\"d\":" << d
                          << ",\"nq\":" << nq << ",\"nk\":" << nk
                          << ",\"variant\":" << kind
                          << ",\"mean_ms\":" << ms / 100
                          << ",\"calls\":100,\"passed\":"
                          << (passed ? "true" : "false") << "}\n";
                check(hipEventDestroy(start));
                check(hipEventDestroy(stop));
              }
            }
      }
    }
    check(hipStreamDestroy(stream));
    return all ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
