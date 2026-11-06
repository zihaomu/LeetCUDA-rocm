import json,sys
from pathlib import Path
import torch
from leetcuda_rocm import fp8
from leetcuda_rocm.environment import ROOT,verify,cache_key
from leetcuda_rocm.loader import _load,_prepare
from leetcuda_rocm.cross_validate import sha,tensor_sha
from leetcuda_rocm.benchmark import graph_pair
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import warmup
torch.set_num_threads(4);torch.set_grad_enabled(False);env=verify()
folder=ROOT/'artifacts/rocm/performance_phase2/fp8/tiled_quantization'
header=ROOT/'artifacts/rocm/performance_phase2/baseline/kernels/rocm/fp8_gemm_hip.cuh';binding=ROOT/'leetcuda_rocm/native/fp8_bindings.cpp'
report=dict(environment=env,harness_sha256=sha(__file__),source_sha256=sha(ROOT/'kernels/rocm/fp8_gemm_hip.cuh'),baseline_sha256=sha(header),performance=[],passed=False,method='Exploratory dispatch sweep: three alternating 20ms samples, 200ms paired warmup, not a substitute for the three-round acceptance')
for fmt in fp8.FORMATS:
 source=header.read_text().removeprefix('#pragma once\n')+'\n'+_prepare(binding.read_text())
 if fmt=='e5m2':source='#define LEETCUDA_FP8_E5M2 1\n'+source
 old=_load('fp8_pipeline_'+fmt,(source,),cache_key(env),env['gpu_architecture'],False);new=fp8._native(0,fmt)
 for k in (256,512,1024,4096):
  for n in (64,136,512,2048):
   g=torch.Generator().manual_seed(97+k+n);cpu=torch.randn(k,n,generator=g).bfloat16();b=cpu.cuda()
   for col in (False,True):
    funcs=[];outputs=[]
    for lib in (new,old):
     bt=torch.empty(n,k,device='cuda',dtype=torch.uint8);s=torch.empty(n if col else (n+127)//128,device='cuda')
     fn=lambda lib=lib,bt=bt,s=s,b=b,col=col:lib.quantize_b(b,bt,s,col);fn();funcs.append(fn);outputs.append((bt,s))
    assert all(torch.equal(x,y) for x,y in zip(*outputs))
    warmup(*funcs);r=dict(format=fmt,k=k,n=n,per_col=col,input_sha256=tensor_sha(cpu),bitwise=True,**graph_pair(*funcs,calls=16,sample_ms=20,trials=3));report['performance'].append(r)
   print(fmt,k,n,'done',flush=True)
report['passed']=all(r['bitwise'] for r in report['performance']);report['regressions_gt_10_percent']=[{k:r[k] for k in ('format','k','n','per_col','baseline_over_ours')} for r in report['performance'] if r['ours_ms']/r['baseline_ms']>1.1]
(folder/'dispatch_sweep.json').write_text(json.dumps(report,indent=2)+'\n');print('regressions',report['regressions_gt_10_percent'],flush=True)
