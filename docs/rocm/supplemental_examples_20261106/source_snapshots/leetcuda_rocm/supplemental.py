"""Explicit supplemental examples; their inventory is separate from 229 bindings."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import re
import torch
from .environment import ROOT, verify, cache_key
from .loader import _load
from .cross_validate import CUTLASS, UPSTREAM, sha

SOURCES = {
    'profile': ('kernels/nvidia-nsight/elementwise.cu','kernels/nvidia-nsight/relu.cu'),
    'vector': ('kernels/cutlass/cute/vector_add.cu',),
    'transpose': ('kernels/swizzle/mat_trans_swizzle.cu',),
}
NAMES = {
    'profile': tuple(prefix+suffix+'_kernel' for prefix in ('elementwise_add_','relu_') for suffix in ('f32','f32x4','f16','f16x2','f16x8','f16x8_pack')),
    'vector': ('vector_add_local_tile_multi_elem_per_thread_half',),
    'transpose': ('mat_trans_smem_naive_kernel','mat_trans_smem_padding_kernel','mat_trans_smem_swizzle_kernel'),
}


def binding(group, hip):
    common = '''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <numeric>
using torch::Tensor;
void guard_tensor(Tensor t,Tensor ref){
 TORCH_CHECK(t.is_cuda() && t.device()==ref.device() && t.is_contiguous() && !t.requires_grad() && t.numel()<=INT_MAX,"supplemental tensor device/layout/gradient/range");
}
bool alias(Tensor a,Tensor b){return a.storage().unsafeGetStorageImpl()==b.storage().unsafeGetStorageImpl();}
'''
    if group=='profile':
        code='''
void forward(Tensor a,Tensor b,Tensor o,int kind){
 TORCH_CHECK(a.is_cuda(),"GPU required");c10::cuda::CUDAGuard guard(a.device());
 for(auto t:{a,b,o})guard_tensor(t,a);
 TORCH_CHECK(kind>=0 && kind<12 && a.sizes()==o.sizes() && (kind>=6 || a.sizes()==b.sizes()),"profile kind/shape");
 int v=kind%6;bool f32=v<2;auto dtype=f32?at::kFloat:at::kHalf;
 TORCH_CHECK(a.scalar_type()==dtype && o.scalar_type()==dtype && (kind>=6 || b.scalar_type()==dtype),"profile dtype");
 TORCH_CHECK(!alias(o,a) && (kind>=6 || !alias(o,b)),"profile output alias");
 int width=v==0||v==2?1:v==1?4:v==3?2:8;
 int n=a.numel();TORCH_CHECK(n%width==0,"original profile vector length must be aligned");
 int bytes=width*(f32?4:2);TORCH_CHECK(uintptr_t(a.data_ptr())%bytes==0 && uintptr_t(o.data_ptr())%bytes==0 && (kind>=6||uintptr_t(b.data_ptr())%bytes==0),"profile pointer alignment");
 if(n==0)return;int threads=std::gcd(n/width,256),blocks=n/width/threads;auto stream=at::cuda::getCurrentCUDAStream();
 switch(kind){
'''
        for i,name in enumerate(NAMES[group]):
            typ='float' if i%6<2 else 'half'
            inputs=f'reinterpret_cast<{typ}*>(a.data_ptr()),'+(f'reinterpret_cast<{typ}*>(b.data_ptr()),' if i<6 else '')+f'reinterpret_cast<{typ}*>(o.data_ptr()),n'
            code+=f'case {i}:{{\n  {name}<<<blocks,threads,0,stream>>>({inputs});break;}}\n'
        code+='}C10_CUDA_KERNEL_LAUNCH_CHECK();}\n'
        exports='m.def("forward",&forward);'
    elif group=='vector':
        code='''
void vector(Tensor x,Tensor y,Tensor o,float a,float b,float c,int elements){
 TORCH_CHECK(x.is_cuda(),"GPU required");c10::cuda::CUDAGuard guard(x.device());for(auto t:{x,y,o}){guard_tensor(t,x);TORCH_CHECK(t.scalar_type()==at::kHalf && t.sizes()==x.sizes(),"vector half shape");}
 TORCH_CHECK(!alias(o,x)&&!alias(o,y),"vector alias");TORCH_CHECK(elements==8||elements==16,"vector elements 8/16");int n=x.numel();if(!n)return;
'''
        if not hip:
            code+='TORCH_CHECK(n%elements==0 && uintptr_t(x.data_ptr())%16==0 && uintptr_t(y.data_ptr())%16==0 && uintptr_t(o.data_ptr())%16==0,"original CuTe alignment");\n'
        code+='''auto stream=at::cuda::getCurrentCUDAStream();int blocks=(n+256*elements-1)/(256*elements);
#define CALL(E) vector_add_local_tile_multi_elem_per_thread_half<E><<<blocks,256,0,stream>>>(reinterpret_cast<half*>(o.data_ptr()),n,reinterpret_cast<const half*>(x.data_ptr()),reinterpret_cast<const half*>(y.data_ptr()),__float2half(a),__float2half(b),__float2half(c))
if(elements==8){CALL(8);}else{CALL(16);}
#undef CALL
C10_CUDA_KERNEL_LAUNCH_CHECK();}
'''
        exports='m.def("vector",&vector);'
    else:
        code='''
void transpose(Tensor a,Tensor o,int mode){
 TORCH_CHECK(a.is_cuda(),"GPU required");c10::cuda::CUDAGuard guard(a.device());guard_tensor(a,a);guard_tensor(o,a);
 TORCH_CHECK(a.dim()==2 && o.dim()==2 && a.size(0)==o.size(1) && a.size(1)==o.size(0) && a.scalar_type()==at::kInt && o.scalar_type()==at::kInt && mode>=0 && mode<3,"transpose shape/type/mode");TORCH_CHECK(!alias(a,o),"transpose alias");int m=a.size(0),n=a.size(1);if(!m||!n)return;auto stream=at::cuda::getCurrentCUDAStream();
'''
        if hip:
            code+='dim3 block(32,8),grid((n+31)/32,(m+31)/32);\nswitch(mode){\n'
            for i in range(3):code+=f'case {i}:{{\n  leetcuda_transpose_examples::transpose<{i}><<<grid,block,0,stream>>>(a.data_ptr<int>(),o.data_ptr<int>(),m,n);break;}}\n'
        else:
            code+='TORCH_CHECK(m%32==0 && n%32==0,"original transpose requires full tiles");dim3 block(32,32),grid(n/32,m/32);switch(mode){\n'
            for i,name in enumerate(NAMES[group]):code+=f'case {i}:{{\n  {name}<<<grid,block,0,stream>>>(a.data_ptr<int>(),m,n,o.data_ptr<int>());break;}}\n'
        code+='}C10_CUDA_KERNEL_LAUNCH_CHECK();}\n';exports='m.def("transpose",&transpose);'
    return common+code+'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){'+exports+'}\n'


@lru_cache(maxsize=None)
def build(group,backend='hip',device=0):
    if group not in SOURCES:raise ValueError(group)
    root=ROOT if backend=='hip' else ROOT/'original'
    paths=[root/p for p in SOURCES[group]];texts=[p.read_text().split('int main(',1)[0] for p in paths]
    extra=[]
    if backend=='hip' and group in ('vector','transpose'):
        p=ROOT/'kernels/rocm'/('cute_vector_hip.cuh' if group=='vector' else 'transpose_examples_hip.cuh');extra.append(p);body=p.read_text().replace('#pragma once','')
    else:
        if group=='vector':body='\n'.join(texts)
        else:body='#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n'+re.sub(r'^#include[^\n]*\n','', '\n'.join(texts),flags=re.M)
    if backend=='hip' and group=='profile':
        body=body.replace('__hmax2(', 'leetcuda_hmax2(')
        body=body.replace('#include <cuda_fp16.h>', '#include <cuda_fp16.h>\n__device__ __forceinline__ half2 leetcuda_hmax2(half2 a,half2 b){half2 r;r.x=__hmax(a.x,b.x);r.y=__hmax(a.y,b.y);return r;}')
    source=body+'\n'+binding(group,backend=='hip')
    meta=dict(source_sha256={str(p):sha(p) for p in paths+extra},adapter_sha256=sha(__file__),original_kernel_names=NAMES[group],source_kind='native_layout' if backend=='hip' and group!='profile' else 'original_compute_body',upstream_commit=UPSTREAM)
    if backend=='hip':
        with torch.cuda.device(device):
            identity=verify();lib=_load('supplemental_'+group,(source,),cache_key(identity),identity['gpu_architecture'],False)
    else:
        from torch.utils.cpp_extension import load
        digest=hashlib.sha256(source.encode()).hexdigest()[:16];folder=ROOT/'.cache/cuda_comparison'/('supplemental_'+group+'_'+digest);folder.mkdir(parents=True,exist_ok=True);p=folder/'source.cu'
        if not p.exists() or p.read_text()!=source:p.write_text(source)
        os.environ['TORCH_CUDA_ARCH_LIST']='12.0a';os.environ.setdefault('MAX_JOBS','2')
        lib=load(name='leetcuda_supplemental_'+digest,sources=[str(p)],build_directory=str(folder),extra_cflags=['-O3','-std=c++20'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__'],extra_include_paths=[str(ROOT/'.cache/cutlass/include')])
        meta['cutlass_commit']=CUTLASS
    meta.update(binary=lib.__file__,binary_sha256=sha(lib.__file__));return lib,meta
