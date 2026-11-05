// Appended to the production source by the test harness; never a public export.
template <int Rows, bool Transpose, bool Swizzle>
__global__ void layout_probe_kernel(const _Float16 *input, _Float16 *physical,
                                    _Float16 *logical) {
  constexpr int E = Rows * 16, Threads = 128;
  __shared__ __align__(16) _Float16 data[2 * E];
  U4 r[(E + Threads * 8 - 1) / (Threads * 8)];
  for (int slot = 0; slot < 2; ++slot) {
    issue<Rows, 16, 16, Threads>(input + slot * E, 0, 0, Rows, r);
    publish<Rows, 16, Threads, Transpose, Swizzle>(data + slot * E, r);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 2 * E; i += Threads) {
    physical[i] = data[i];
    int slot = i / E, x = i % E;
    constexpr int Cols = Transpose ? Rows : 16;
    logical[i] = data[slot * E + lds_index<Cols, Swizzle>(x / Cols, x % Cols)];
  }
}
std::vector<torch::Tensor> layout_probe(torch::Tensor input, bool transpose,
                                        bool swizzle) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.is_contiguous() && input.dim() == 3 &&
                  input.size(0) == 2 && input.size(2) == 16 &&
                  (input.size(1) == 64 || input.size(1) == 128),
              "probe expects contiguous [2,64|128,16] FP16");
  c10::cuda::CUDAGuard guard(input.device());
  auto physical = torch::empty_like(input), logical = torch::empty_like(input);
#define PROBE(R, T, S)                                                         \
  layout_probe_kernel<R, T, S>                                                 \
      <<<1, 128>>>(reinterpret_cast<const _Float16 *>(input.data_ptr()),       \
                   reinterpret_cast<_Float16 *>(physical.data_ptr()),          \
                   reinterpret_cast<_Float16 *>(logical.data_ptr()))
#define FLAGS(R)                                                               \
  if (transpose) {                                                             \
    if (swizzle) {                                                             \
      PROBE(R, true, true);                                                    \
    } else {                                                                   \
      PROBE(R, true, false);                                                   \
    }                                                                          \
  } else {                                                                     \
    if (swizzle) {                                                             \
      PROBE(R, false, true);                                                   \
    } else {                                                                   \
      PROBE(R, false, false);                                                  \
    }                                                                          \
  }
  if (input.size(1) == 64) {
    FLAGS(64);
  } else {
    FLAGS(128);
  }
#undef FLAGS
#undef PROBE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {physical, logical};
}
