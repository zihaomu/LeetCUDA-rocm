"""HIP implementations of the three CuTeDSL examples, preserving their schedules.

This is a backend replacement, not a ROCm port of NVIDIA's CuTeDSL compiler.
The naive example preserves its FP16-rounded FP32 output. MMA atom and pipeline
are SIMT FP32, with respectively one padded LDS tile or three unpadded slots.
"""
from functools import lru_cache
import torch
from .environment import ROOT,verify,cache_key
from .loader import _load

VARIANTS={'naive':0,'mma_atom':1,'pipelining':2}

def source_paths():
    return (ROOT/'kernels/rocm/cute_dsl_gemm_hip.cuh',ROOT/'leetcuda_rocm/native/cute_dsl_bindings.cpp')

@lru_cache(maxsize=None)
def _native(device):
    with torch.cuda.device(device):
        identity=verify();source='\n'.join(p.read_text().replace('#pragma once','') for p in source_paths());return _load('cute_dsl',(source,),cache_key(identity),identity['gpu_architecture'],False)

def matmul(a,b,*,variant='naive',out=None):
    if variant not in VARIANTS:raise ValueError('unknown DSL variant')
    if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or not a.is_cuda or a.ndim!=2 or b.ndim!=2:raise ValueError('GPU matrices required')
    if out is None:out=torch.empty((a.shape[0],b.shape[1]),device=a.device,dtype=a.dtype)
    bt=b if variant=='naive' else b.T.contiguous()
    _native(a.device.index).forward(a,bt,out,VARIANTS[variant],None)
    return out

def run_example(variant,M=2048,K=2048,N=2048,verify=True,warmup_iterations=10,iterations=100):
    a=torch.randn(M,K,device='cuda');b=torch.randn(K,N,device='cuda');out=matmul(a,b,variant=variant)
    if verify:
        ref=a.double()@b.double();ref=ref.half().float() if variant=='naive' else ref.float()
        torch.testing.assert_close(out,ref,atol=.01 if variant=='naive' else .001,rtol=.01 if variant=='naive' else .001)
    from .benchmark import graph_pair
    bt=b if variant=='naive' else b.T.contiguous();lib=_native(a.device.index)
    if type(iterations) is not int or iterations<=0 or type(warmup_iterations) is not int or warmup_iterations<0:raise ValueError('invalid benchmark iteration count')
    for _ in range(warmup_iterations):lib.forward(a,bt,out,VARIANTS[variant],None)
    result=graph_pair(lambda:lib.forward(a,bt,out,VARIANTS[variant],None),lambda:torch.mm(a,b,out=out),calls=iterations)
    print(dict(backend='HIP',variant=variant,**result));return result
