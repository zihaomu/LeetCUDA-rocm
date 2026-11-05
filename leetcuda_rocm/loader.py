"""HIPify original sources in a project-local cache, without CUDA-only flags.

These are educational low-level kernels: their documented shape/alignment and
dtype contracts still apply. Use the smoke suite for the qualified workloads.
No torch implementation is substituted for a LeetCUDA kernel.
"""

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

from .environment import ROOT, cache_key, verify

KERNELS = {
    "elementwise": "elementwise/elementwise.cu",
    **{
        name: f"{name}/{name}.cu"
        for name in (
            "relu",
            "gelu",
            "sigmoid",
            "swish",
            "hardswish",
            "hardshrink",
            "elu",
            "softmax",
            "rope",
            "embedding",
            "histogram",
            "sgemv",
            "hgemv",
            "sgemm",
            "nms",
        )
    },
    "reduce": "reduce/block_all_reduce.cu",
    "dot_product": "dot-product/dot_product.cu",
    "layer_norm": "layer-norm/layer_norm.cu",
    "rms_norm": "rms-norm/rms_norm.cu",
    "mat_transpose": "mat-transpose/mat_transpose.cu",
    "merge_attn_states": "openai-triton/merge-attn-states/cuda_merge_attn_states.cu",
    "hgemm": "hgemm/pybind/hgemm.cc",
    "flash_attn": "flash-attn/pybind/flash_attn.cc",
    "ws_hgemm": "ws-hgemm/naive_ws_hgemm_sm8x.cu",
}

EXTRA_SOURCES = {
    "reduce": ("rocm/reduce_precision.cu",),
    "ws_hgemm": ("rocm/ws_hgemm.cu",),
    "hgemv": (
        "rocm/hgemv_matrix_precision.cu",
        "rocm/hgemv_vector_precision.cu",
    ),
    "sgemm": (
        "sgemm/sgemm_async.cu",
        "rocm/blas_backend.cu",
        "rocm/sgemm_tf32.cu",
    ),
    "mat_transpose": ("rocm/mat_transpose_layouts.cu",),
    "hgemm": (
        "rocm/blas_backend.cu",
        "rocm/hgemm_wmma.cu",
        "rocm/hgemm_mma.cu",
        "rocm/hgemm_async.cu",
        "rocm/hgemm_simt_precision.cu",
    ),
    "flash_attn": (
        "rocm/flash_attention_shared.cu",
        "rocm/flash_attention_tiled.cu",
        "rocm/flash_attention_low_shared.cu",
        "rocm/flash_attention_low_tiled.cu",
        "rocm/flash_attention_split_kv.cu",
    ),
}


def source_paths(name):
    return tuple(
        ROOT / "kernels" / path
        for path in (KERNELS[name], *EXTRA_SOURCES.get(name, ()))
    )


def _prepare(source):
    # HIP's sync intrinsics require a 64-bit mask even on wave32 targets.
    source = re.sub(r"\b0xffffffff\b", "0xffffffffULL", source)
    # PyTorch's HIPify table does not yet cover the CUDA FP8 header/symbols.
    source = source.replace("<cuda_fp8.h>", "<hip/hip_fp8.h>")
    for cuda, hip in {
        "__nv_fp8_storage_t": "__hip_fp8_storage_t",
        "__nv_cvt_fp8_to_halfraw": "__hip_cvt_fp8_to_halfraw",
        "__nv_cvt_float_to_fp8": "__hip_cvt_float_to_fp8",
        "__NV_E4M3": "__HIP_E4M3",
        "__NV_E5M2": "__HIP_E5M2",
        "__NV_SATFINITE": "__HIP_SATFINITE",
    }.items():
        source = source.replace(cuda, hip)
    # Existing examples use the default stream and some allocate on device 0.
    # Adapt those host bindings before PyTorch's official CUDA -> HIP pass.
    source = source.replace(
        "#include <torch/extension.h>",
        "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>",
    )

    def stream(match):
        args = match[1].split(",")
        if len(args) == 2:
            args.append(" 0")
        if len(args) != 3:
            raise RuntimeError(f"Unreviewed kernel launch: {match[0]}")
        return "<<<" + ",".join(args) + ", at::cuda::getCurrentCUDAStream()>>>"

    source = re.sub(r"<<<([^<>]+)>>>", stream, source)
    # Preserve the current device set by the caller, including logical GPU1+.
    source = source.replace(
        ".device(torch::kCUDA, 0)",
        ".device(torch::kCUDA, c10::cuda::current_device())",
    )
    return source


def load(name, *, verbose=False):
    """Build a low-level module on the current ROCm device (no CPU fallback).

    Raw bindings retain the original example contracts and do not implement
    autograd. Inputs must be contiguous, correctly aligned GPU tensors on the
    current device. See docs/rocm/README.md before calling them directly.
    """
    if name not in KERNELS:
        raise ValueError(
            f"Unsupported ROCm module {name!r}; choose from {', '.join(KERNELS)}"
        )
    identity = verify()
    sources = tuple(_prepare(path.read_text()) for path in source_paths(name))
    if name == "ws_hgemm":
        # Native producer/consumer implementation replaces CUDA pipeline/CuTe.
        # Keep the upstream path in source_paths for provenance and CUDA builds.
        binding = (
            "#include <torch/extension.h>\n"
            "void ws_hgemm_naive_cute(torch::Tensor,torch::Tensor,torch::Tensor);\n"
            'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("ws_hgemm_naive_cute",&ws_hgemm_naive_cute);}'
        )
        sources = (_prepare(binding), *sources[1:])
    if name == "hgemm":
        sources = tuple(
            "#define LEETCUDA_ROCM_BLAS_ONLY 1\n" + s for s in sources
        )
    return _load(
        name,
        sources,
        cache_key(identity),
        identity["gpu_architecture"],
        verbose,
    )


@lru_cache(maxsize=None)
def _load(name, sources, environment_key, architecture, verbose):
    import importlib.util
    import json
    import subprocess
    import sys
    import tempfile

    digest = hashlib.sha256("\0".join(sources).encode()).hexdigest()[:16]
    cache = ROOT / ".cache/rocm10" / environment_key / "torch"
    # Never reuse another project's compiler/JIT cache.
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
    os.environ["PYTORCH_ROCM_ARCH"] = architecture
    os.environ.setdefault("MAX_JOBS", "2")
    build = cache / f"{name}_{digest}"
    build.mkdir(parents=True, exist_ok=True)
    staged_sources = []
    for index, source in enumerate(sources):
        filename = f"{name}.cu" if len(sources) == 1 else f"{name}_{index}.cu"
        staged = build / filename
        if not staged.exists() or staged.read_text() != source:
            staged.write_text(source)
        staged_sources.append(str(staged))
    options = dict(
        name=f"leetcuda_rocm_{name}_{digest}",
        sources=staged_sources,
        build_directory=str(build),
        verbose=verbose,
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=["-lhipblas"] if name in ("sgemm", "hgemm") else [],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-DLEETCUDA_ROCM=1",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
        ],
    )

    # Pybind can reuse the first module of a given name, even when subsequently
    # imported from another file. Build in a separate process, then make the
    # immutable path the parent's first and only import of this native module.
    worker = ROOT / "leetcuda_rocm/build_extension.py"
    recipe = dict(
        build={k: v for k, v in options.items() if k != "verbose"},
        compiler_worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
    )
    recipe_hash = hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode()
    ).hexdigest()
    receipt_path = build / ("immutable-" + recipe_hash + ".json")
    if not receipt_path.exists():
        request = dict(
            build=options,
            frozen_root=str(cache / "frozen"),
            recipe_sha256=recipe_hash,
            receipt=str(receipt_path),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=build, delete=False
        ) as stream:
            json.dump(request, stream)
            request_path = Path(stream.name)
        try:
            subprocess.run(
                [sys.executable, str(worker), str(request_path)], check=True
            )
        finally:
            request_path.unlink(missing_ok=True)
    receipt = json.loads(receipt_path.read_text())
    frozen = Path(receipt["binary"])
    if (
        receipt["recipe_sha256"] != recipe_hash
        or hashlib.sha256(frozen.read_bytes()).hexdigest()
        != receipt["binary_sha256"]
    ):
        raise RuntimeError("Immutable native extension receipt/hash mismatch")
    spec = importlib.util.spec_from_file_location(
        receipt["module_name"], frozen
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
