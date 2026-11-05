#!/usr/bin/env python3
"""Build a standalone HIP executable with no PyTorch link dependency."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from leetcuda_rocm.cross_validate import sha
from leetcuda_rocm.environment import verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/rocm/continuation_all/supplemental/standalone');args=p.parse_args();args.output=args.output.resolve();folder=args.output.parent;folder.mkdir(parents=True,exist_ok=True)
 paths=[ROOT/'kernels/nvidia-nsight'/n for n in ('elementwise.cu','relu.cu')];body='\n'.join(p.read_text().split('int main(',1)[0] for p in paths)
 import re
 body=re.sub(r'^#include[^\n]*\n','',body,flags=re.M).replace('__hmax2(','leetcuda_hmax2(')
 body='#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n__device__ __forceinline__ half2 leetcuda_hmax2(half2 a,half2 b){half2 r;r.x=__hmax(a.x,b.x);r.y=__hmax(a.y,b.y);return r;}\n'+body
 staged=folder/'profile_compute.hpp';staged.write_text(body);source=ROOT/'kernels/rocm/supplemental_bench.cpp';files=[*paths,source,Path(__file__),*[ROOT/'kernels/rocm'/p for p in ('bench_runtime.hpp','cute_vector_hip.cuh','transpose_examples_hip.cuh','cute_dsl_gemm_hip.cuh','swizzle_examples_hip.cuh')]]
 command=['/opt/rocm/bin/hipcc','-O3','-std=c++20','--offload-arch=gfx1201','-DLEETCUDA_ROCM=1','-I'+str(folder),'-I'+str(ROOT/'kernels/rocm'),str(source),'-o',str(args.output)]
 identity=verify();subprocess.run(command,check=True);report=dict(environment=identity,sources={str(p.relative_to(ROOT)):sha(p) for p in files},generated_profile_sha256=sha(staged),command=command,binary_sha256=sha(args.output));args.output.with_suffix('.build.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output)
if __name__=='__main__':main()
