#pragma once
// Shared host-only test machinery for identical HIP/CUDA standalone fixtures.
#if defined(LEETCUDA_ROCM)
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace leetcuda_bench {
inline void check(GPU(Error_t) error) {
  if (error != GPU(Success))
    throw std::runtime_error(GPU(GetErrorString)(error));
}
template <class T> struct Buffer {
  T *p = nullptr;
  size_t n;
  explicit Buffer(size_t count) : n(count) {
    check(GPU(Malloc)(&p, std::max(size_t(1), n) * sizeof(T)));
  }
  ~Buffer() { (void)GPU(Free)(p); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void upload(const std::vector<T> &x) {
    if (x.size() != n)
      throw std::runtime_error("upload size");
    if (n)
      check(GPU(Memcpy)(p, x.data(), n * sizeof(T), GPU(MemcpyHostToDevice)));
  }
  std::vector<T> download() {
    std::vector<T> x(n);
    if (n)
      check(GPU(Memcpy)(x.data(), p, n * sizeof(T), GPU(MemcpyDeviceToHost)));
    return x;
  }
};
inline std::vector<float> read_fixture(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("fixture open");
  auto bytes = file.tellg();
  if (bytes <= 0 || bytes % 4)
    throw std::runtime_error("fixture size");
  std::vector<float> data(size_t(bytes) / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(data.data()), bytes);
  if (!file)
    throw std::runtime_error("fixture read");
  return data;
}
inline std::vector<float> slice(const std::vector<float> &data, size_t offset,
                                size_t count) {
  if (offset + count > data.size())
    throw std::runtime_error("fixture too short");
  return {data.begin() + offset, data.begin() + offset + count};
}
struct Harness {
  GPU(Stream_t) stream {};
  std::ofstream output;
  size_t offset = 0;
  bool passed = true;
  bool performance;
  Harness(const std::string &path, bool timing)
      : output(path, std::ios::binary), performance(timing) {
    if (!output)
      throw std::runtime_error("output open");
    check(GPU(StreamCreate)(&stream));
  }
  ~Harness() { (void)GPU(StreamDestroy)(stream); }
  template <class T>
  void record(const std::string &id, const std::vector<T> &values,
              const std::vector<double> &expected, double atol, double rtol,
              bool canary = true, double ms = -1) {
    if (values.size() != expected.size())
      throw std::runtime_error("reference size");
    bool ok = canary;
    double error = 0;
    for (size_t i = 0; i < values.size(); i++) {
      double value = double(values[i]), delta = std::abs(value - expected[i]);
      ok &=
          std::isfinite(value) && delta <= atol + rtol * std::abs(expected[i]);
      error = std::max(error, delta);
      output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
    passed &= ok;
    std::cout << std::setprecision(12) << "{\"id\":\"" << id
              << "\",\"passed\":" << (ok ? "true" : "false")
              << ",\"count\":" << values.size() << ",\"offset\":" << offset
              << ",\"atol\":" << atol << ",\"rtol\":" << rtol
              << ",\"max_abs\":" << error
              << ",\"canary\":" << (canary ? "true" : "false");
    if (ms >= 0)
      std::cout << ",\"mean_ms\":" << ms;
    std::cout << "}\n";
    offset += values.size();
  }
  template <class F> double timed(F call) {
    if (!performance)
      return -1;
    for (int i = 0; i < 5; i++)
      call();
    GPU(Event_t) start, stop;
    check(GPU(EventCreate)(&start));
    check(GPU(EventCreate)(&stop));
    check(GPU(EventRecord)(start, stream));
    for (int i = 0; i < 100; i++)
      call();
    check(GPU(EventRecord)(stop, stream));
    check(GPU(EventSynchronize)(stop));
    float ms = 0;
    check(GPU(EventElapsedTime)(&ms, start, stop));
    check(GPU(EventDestroy)(start));
    check(GPU(EventDestroy)(stop));
    return ms / 100;
  }
  template <class F>
  void floats(const std::string &id, const std::vector<double> &expected,
              F call, double atol = 2e-5, double rtol = 2e-5) {
    Buffer<float> out(expected.size() + 8);
    std::vector<float> initial(out.n, std::numeric_limits<float>::quiet_NaN());
    for (int i = 0; i < 4; i++)
      initial[i] = initial[out.n - 1 - i] = 91;
    out.upload(initial);
    call(out.p + 4);
    check(GPU(GetLastError)());
    check(GPU(StreamSynchronize)(stream));
    auto actual = out.download();
    bool canary = true;
    for (int i = 0; i < 4; i++)
      canary &= actual[i] == 91 && actual[out.n - 1 - i] == 91;
    std::vector<float> values(actual.begin() + 4, actual.end() - 4);
    double ms = timed([&] { call(out.p + 4); });
    record(id, values, expected, atol, rtol, canary, ms);
  }
};
} // namespace leetcuda_bench
