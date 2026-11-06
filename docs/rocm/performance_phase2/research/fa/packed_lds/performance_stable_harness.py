import argparse, json, math, statistics, re, subprocess
from pathlib import Path
from types import SimpleNamespace
import torch, yaml
from leetcuda_rocm import interview_attention as api
from leetcuda_rocm import interview_attention_check as checks
from leetcuda_rocm.environment import ROOT, verify, cache_key
from leetcuda_rocm.loader import _load
from leetcuda_rocm.cross_validate import sha, tensor_sha
from leetcuda_rocm.benchmark import graph_pair
from leetcuda_rocm.isa_objects import hip_isa_objects

p=argparse.ArgumentParser();p.add_argument('phase',choices=['correct','performance','isa']);p.add_argument('--header',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=1);p.add_argument('--warmup-ms',type=float,default=0);a=p.parse_args()
torch.set_num_threads(4);torch.set_grad_enabled(False);env=verify();a.output.parent.mkdir(parents=True,exist_ok=True)
common=ROOT/'kernels/rocm/interview_common_hip.cuh';binding=ROOT/'leetcuda_rocm/native/interview_attention_bindings.cpp'
source=common.read_text().replace('#pragma once','')+'\n'+a.header.read_text().replace('#pragma once','').replace('#include "interview_common_hip.cuh"','')+'\n'+binding.read_text()
candidate=_load('interview_attention_opt',(source,),cache_key(env),env['gpu_architecture'],False)
meta=dict(binary=candidate.__file__,binary_sha256=sha(candidate.__file__),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (common,a.header.resolve(),binding)})
if a.phase=='correct':
 api._native=lambda device:candidate
 api.source_paths=lambda:(common,a.header.resolve(),binding)
 args=SimpleNamespace(device=0,backend='hip',fixtures=ROOT/'artifacts/rocm/continuation_all/interview_fa/fixtures.pt',output=a.output,kind=None,d=None,performance=False,layout=ROOT/'artifacts/rocm/continuation_all/interview_fa/cute_layout_reference.jsonl')
 checks.run(args)
elif a.phase=='isa':
 isa=hip_isa_objects(candidate.__file__,(a.output.parent/(a.output.stem+'_isa')).resolve());ks=[]
 for obj in isa['objects']:
  notes=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readobj','--notes',str(ROOT/obj['hsaco'])],text=True)
  for e in yaml.safe_load(notes.split('AMDGPU Metadata: ',1)[1].split('...',1)[0])['amdhsa.kernels']:
   m=re.search(r'kernelILi(\d+)ELi(\d+)ELb([01])EE',e['.name'])
   if not m:continue
   k,d,audit=map(int,m.groups());ks.append(dict(kind=k,d=d,audit=bool(audit),vgpr=e['.vgpr_count'],spill=e['.vgpr_spill_count'],scratch=e['.private_segment_fixed_size'],lds=e['.group_segment_fixed_size']))
 result=dict(environment=env,build=meta,isa=isa,kernels=ks);a.output.write_text(json.dumps(result,indent=2)+'\n');print([(x['kind'],x['d'],x['spill'],x['scratch']) for x in ks if not x['audit']],flush=True)
else:
 baseline=api._native(0);cases=[];correctness=[]
 for name,kind in api.VARIANTS.items():
  for d in checks.dimensions(kind):
   for n in (512,2048):
    gen=torch.Generator().manual_seed(71);cpu=[torch.randn(1,2,n,d,generator=gen).mul_(.5).half() for _ in range(3)]
    ref,lref=checks.reference(*cpu,persist=kind==7)
    q,k,v=[x.cuda() for x in cpu];oo,ob=[torch.empty_like(q) for _ in range(2)];lo,lb=[torch.empty(q.shape[:-1],device='cuda') for _ in range(2)]
    sk=1 if kind in (2,5) and d==128 else 2;sv=2 if kind==7 else 1
    extra=(kind,sk,sv,7 if kind==0 else 0,False,d**-.5,0,None,0)
    ours=lambda q=q,k=k,v=v,o=oo,l=lo,e=extra:candidate.forward(q,k,v,o,l,*e)
    old=lambda q=q,k=k,v=v,o=ob,l=lb,e=extra:baseline.forward(q,k,v,o,l,*e)
    for label,fn,o,l in [('candidate',ours,oo,lo),('baseline',old,ob,lb)]:
     fn();accuracy=checks.accurate(o.cpu(),ref);lse=checks.accurate(l.cpu(),lref,lse=True)
     stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):fn()
     with torch.cuda.stream(stream):o.fill_(float('nan'));l.fill_(float('nan'));g.replay()
     stream.synchronize();ok=checks.accurate(o.cpu(),ref)['passed'] and checks.accurate(l.cpu(),lref,lse=True)['passed']
     correctness.append(dict(variant=name,d=d,n=n,backend=label,output=accuracy,lse=lse,graph=ok,passed=accuracy['passed'] and lse['passed'] and ok))
    assert all(x['passed'] for x in correctness)
    cases.append((dict(variant=name,kind=kind,d=d,n=n,sk=sk,sv=sv,inputs_sha256=[tensor_sha(x) for x in cpu]),ours,old))
 report=dict(environment=env,build=meta,baseline_binary=baseline.__file__,baseline_binary_sha256=sha(baseline.__file__),harness_sha256=sha(__file__),checks=correctness,performance=[],passed=False)
 def warmup(ours,old):
  if a.warmup_ms<=0:return
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):
   for _ in range(8):ours();old()
  with torch.cuda.stream(stream):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
   while True:
    for _ in range(16):g.replay()
    end.record();end.synchronize()
    if start.elapsed_time(end)>=a.warmup_ms:break
 report['warmup_ms_per_configuration']=a.warmup_ms
 for rid in range(a.rounds):
  ordered=cases[rid:]+cases[:rid]
  for case,ours,old in ordered:
   warmup(ours,old)
   report['performance'].append(dict(**case,round=rid,**graph_pair(ours,old,calls=16)))
  a.output.write_text(json.dumps(report,indent=2)+'\n');print('round',rid+1,'gain',math.exp(statistics.mean(math.log(x['baseline_over_ours']) for x in report['performance'] if x['round']==rid)),flush=True)
 report['passed']=all(x['passed'] for x in correctness);a.output.write_text(json.dumps(report,indent=2)+'\n')
