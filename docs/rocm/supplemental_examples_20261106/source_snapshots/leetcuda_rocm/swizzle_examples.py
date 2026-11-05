"""Five standalone swizzle GEMM definitions with explicit FP32 accumulation."""
from functools import lru_cache
import torch
from .environment import ROOT,verify,cache_key
from .loader import _load

NAMES=('hgemm_mma_m16n8k16_naive_kernel','hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel','hgemm_mma_m16n8k16_naive_smem_swizzle_kernel','hgemm_mma_m16n8k16_mma2x4_warp4x4_smem_swizzle_kernel','mma_simple_swizzle_kernel')
def source_paths():return (ROOT/'kernels/rocm/swizzle_examples_hip.cuh',ROOT/'leetcuda_rocm/native/swizzle_examples_bindings.cpp')
@lru_cache(maxsize=None)
def native(device=0):
    with torch.cuda.device(device):
        identity=verify();source='\n'.join(p.read_text().replace('#pragma once','') for p in source_paths());return _load('swizzle_examples',(source,),cache_key(identity),identity['gpu_architecture'],False)
