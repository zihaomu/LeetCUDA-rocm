"""Generate an explicit CUDA FP32-state comparator; never overwrite original/ported."""
from pathlib import Path
import json,re,hashlib
root=Path.cwd();out=root/'artifacts/rocm/continuation_all/u3/cuda_precision';out.mkdir(exist_ok=True)
header='''
// Explicit precision-reference variant. FP16 I/O, FP32 FMA/state.
// Original tile/copy/BCF/cp.async mechanics remain in the source below.
namespace {
__device__ __forceinline__ float simt_accumulate_fp32(half a,half b,float c) {return fmaf(__half2float(a),__half2float(b),c);}
__device__ __forceinline__ half2 simt_pack2(const float *x) {return __floats2half2_rn(x[0],x[1]);}
__device__ __forceinline__ float2 simt_pack4(const float *x) {
 float2 out;
 #pragma unroll
 for(int j=0;j<4;j++)reinterpret_cast<half*>(&out)[j]=__float2half_rn(x[j]);
 return out;
}
__device__ __forceinline__ float4 simt_pack8(const float *x) {
 float4 out;
 #pragma unroll
 for(int j=0;j<8;j++)reinterpret_cast<half*>(&out)[j]=__float2half_rn(x[j]);
 return out;
}
}
'''
manifest=dict(purpose='Explicit matched FP32-state CUDA comparator, separate from original FP16 evidence',original_accumulator='FP16',reference_accumulator='FP32',changed_interfaces=14,mechanisms='Keep original tile geometries, vector loads/stores, BCF layouts and actual CUDA cp.async; retain previous odd-slot correction',files={})
for name in ('hgemm.cu','hgemm_async.cu'):
 src=root/'kernels/hgemm/naive'/name;s=src.read_text();s=s.replace('#include <cuda_fp16.h>','#include <cuda_fp16.h>\n'+header)
 s=re.sub(r'half (psum|sum|r_c)\b',r'float \1',s).replace('{__float2half(0.0f)}','{0.f}').replace('float sum = __float2half(0.f)','float sum = 0.f')
 s=s.replace('__hfma(', 'simt_accumulate_fp32(')
 s=s.replace('psum += a[m * K + k] * b[k * N + n];','psum = simt_accumulate_fp32(a[m * K + k], b[k * N + n], psum);')
 s=re.sub(r'(r_c\[[^;\n]+?\]|sum)\s*\+=\s*(.+?)\s+\*\s+(.+?);',r'\1 = simt_accumulate_fp32(\2, \3, \1);',s)
 for macro,fn in [('HALF2','simt_pack2'),('LDST64BITS','simt_pack4'),('LDST128BITS','simt_pack8')]:s=re.sub(macro+r'\((r_c\[[^;\n]+?)\)',fn+r'(&\1)',s)
 p=out/name;p.write_text(s);manifest['files'][str(src.relative_to(root))]=dict(slot_corrected_fp16_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),fp32_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(out/'precision_reference.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
