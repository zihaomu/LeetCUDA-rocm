"""ROCm 10 entry point for the qualified subset of LeetCUDA kernels."""

__all__ = [
    "KERNELS",
    "load",
    "load_unbound",
    "gemm",
    "flash_attention",
    "attention",
    "ffpa_attention",
    "fp8_gemm",
    "fp8_gemm_prepacked",
    "quantize_fp8_weight",
    "FP8Weight",
    "fp4_gemm",
    "fp4_gemm_prepacked",
    "quantize_fp4_weight",
    "FP4Weight",
]


def __getattr__(name):
    if name in ("KERNELS", "load"):
        from . import loader

        return getattr(loader, name)
    if name in ("gemm", "flash_attention"):
        from . import matrix_attention

        return getattr(matrix_attention, name)
    if name == "load_unbound":
        from .unbound import load_unbound

        return load_unbound
    if name == "attention":
        from .attention_training import attention

        return attention
    if name == "ffpa_attention":
        from .ffpa import ffpa_attention

        return ffpa_attention
    if name in (
        "fp8_gemm",
        "fp8_gemm_prepacked",
        "quantize_fp8_weight",
        "FP8Weight",
    ):
        from . import fp8

        return getattr(fp8, name)
    if name in (
        "fp4_gemm",
        "fp4_gemm_prepacked",
        "quantize_fp4_weight",
        "FP4Weight",
    ):
        from . import fp4

        return getattr(fp4, name)
    raise AttributeError(name)
