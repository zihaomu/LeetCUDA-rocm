"""Derive precision-upgraded kernels, preserving upstream per-variant dataflow."""
from pathlib import Path
import re,json,hashlib
root=Path.cwd();dst=root/'artifacts/rocm/continuation_all/u3'
names=(
'hgemm_naive_f16','hgemm_sliced_k_f16','hgemm_t_8x8_sliced_k_f16x4','hgemm_t_8x8_sliced_k_f16x4_pack','hgemm_t_8x8_sliced_k_f16x4_bcf','hgemm_t_8x8_sliced_k_f16x4_pack_bcf','hgemm_t_8x8_sliced_k_f16x8_pack_bcf','hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf','hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf','hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf','hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf')
header='''// Explicit FP32-state upgrades of eleven original synchronous SIMT variants.
// Original tile geometry, vector widths, BCF layouts and LDS slot dataflow are
// retained on aligned full tiles. Bounded native scalar tails extend the API.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
namespace {
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
__device__ __forceinline__ float accumulate(half a, half b, float c) {
 return fmaf(__half2float(a), __half2float(b), c);
}
__device__ __forceinline__ half2 pack2(const float *x) {
 return __floats2half2_rn(x[0], x[1]);
}
__device__ __forceinline__ float2 pack4(const float *x) {
 float2 out;
 #pragma unroll
 for(int j=0;j<4;j++) reinterpret_cast<half*>(&out)[j]=__float2half_rn(x[j]);
 return out;
}
__device__ __forceinline__ float4 pack8(const float *x) {
 float4 out;
 #pragma unroll
 for(int j=0;j<8;j++) reinterpret_cast<half*>(&out)[j]=__float2half_rn(x[j]);
 return out;
}
'''
parts=[header];manifest={}
for i,name in enumerate(names):
 src=root/'kernels/hgemm/naive'/('hgemm.cu' if i<8 else 'hgemm_async.cu')
 raw=src.read_text();s=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/','',raw)
 start=s.index('__global__ void '+name+'_kernel')
 if i: start=s.rfind('template <',0,start)
 opening=s.index('{',start);end=opening+1;depth=1
 while depth: depth+=(s[end]=='{')-(s[end]=='}');end+=1
 body=s[start:end].replace(name+'_kernel',name+'_fp32_kernel')
 body=re.sub(r'half (psum|sum|r_c)\b',r'float \1',body).replace('{__float2half(0.0f)}','{0.f}').replace('float sum = __float2half(0.f)','float sum = 0.f')
 body=body.replace('__hfma(', 'accumulate(')
 body=body.replace('psum += a[m * K + k] * b[k * N + n];', 'psum = accumulate(a[m * K + k], b[k * N + n], psum);')
 body,n=re.subn(r'(r_c\[[^;\n]+?\]|psum|sum)\s*\+=\s*(.+?)\s+\*\s+(.+?);',r'\1 = accumulate(\2, \3, \1);',body)
 for macro,fn in [('HALF2','pack2'),('LDST64BITS','pack4'),('LDST128BITS','pack8')]:
  body=re.sub(macro+r'\((r_c\[[^;\n]+?)\)',fn+r'(&\1)',body)
 assert '__hfma' not in body and not re.search(r'half r_c\b', body)
 parts.append(body)
 manifest[name]=dict(source=str(src.relative_to(root)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),BM=32 if i<2 else 128,BN=32 if i<2 else 128,BK=1 if i==0 else 32 if i==1 or i>=9 else 16 if i==8 else 8,TM=1 if i<2 else 16 if i==10 else 8,TN=1 if i<2 else 8,alignment=2 if i<2 else 4 if i in (2,4) else 8 if i in (3,5) else 16,double_buffer=i>=7)
parts.append('''
void validate(torch::Tensor a,torch::Tensor b,torch::Tensor c) {
 for(const auto &t:{a,b,c}) {
 TORCH_CHECK(t.is_cuda() && t.scalar_type()==at::kHalf && t.dim()==2 && t.is_contiguous(), "expected contiguous rank-2 FP16 GPU tensors");
 TORCH_CHECK(!t.requires_grad(),"forward only");
 TORCH_CHECK(t.numel()<INT32_MAX-16384 && t.size(0)<INT32_MAX-256 && t.size(1)<INT32_MAX-256,"index limit");
 }
 TORCH_CHECK(a.device()==b.device() && a.device()==c.device(),"same device required");
 TORCH_CHECK(a.size(1)==b.size(0) && c.size(0)==a.size(0) && c.size(1)==b.size(1),"shape mismatch");
 TORCH_CHECK(!c.is_alias_of(a) && !c.is_alias_of(b),"output alias unsupported");
 TORCH_CHECK((a.size(0)+31)/32<=65535,"grid y limit");
}
} // namespace
''')
for i,(name,g) in enumerate(manifest.items()):
 bm,bn,bk,tm,tn,align=[g[x] for x in ['BM','BN','BK','TM','TN','alignment']]
 args='' if i==0 else f'<{bm},{bn},{bk}'+('>' if i==1 else f',{tm},{tn}>')
 parts.append(f'''void {name}(torch::Tensor a,torch::Tensor b,torch::Tensor c) {{
 validate(a,b,c); c10::cuda::CUDAGuard guard(a.device());
 int M=a.size(0),N=b.size(1),K=a.size(1); if(!M||!N)return;
 auto A=reinterpret_cast<half*>(a.data_ptr());auto B=reinterpret_cast<half*>(b.data_ptr());auto C=reinterpret_cast<half*>(c.data_ptr());
 bool full = K>0 && M%{bm}==0 && N%{bn}==0 && K%{bk}==0 && (uintptr_t(A)|uintptr_t(B)|uintptr_t(C))%{align}==0;
 if ({'true' if i==0 else '!full'}) {{
 dim3 grid((N+31)/32,(M+31)/32),block(32,32);
 hgemm_naive_f16_fp32_kernel<<<grid,block>>>(A,B,C,M,N,K);
 }} else {{
 dim3 grid(N/{bn},M/{bm}),block({bn//tn},{bm//tm});
 {name}_fp32_kernel{args}<<<grid,block>>>(A,B,C,M,N,K);
 }}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
''')
(dst/'hgemm_simt_precision.cu').write_text('\n'.join(parts))
(dst/'derivation.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Derived',len(names),'distinct kernels')
