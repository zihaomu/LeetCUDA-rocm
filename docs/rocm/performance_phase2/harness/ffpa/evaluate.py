import argparse,json,sys,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from leetcuda_rocm import ffpa,ffpa_check as ref
from leetcuda_rocm.loader import _load,_prepare
from leetcuda_rocm.environment import ROOT,verify,cache_key
from leetcuda_rocm.cross_validate import sha
from leetcuda_rocm.benchmark import graph_pair
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import warmup,gates
p=argparse.ArgumentParser();p.add_argument('phase',choices=['correct','performance']);p.add_argument('--header',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=1);p.add_argument('--variant');a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(__file__,a.output.with_name(a.output.stem+'_harness.py'));torch.set_num_threads(4);torch.set_grad_enabled(False);env=verify()
fixture=ROOT/'artifacts/rocm/continuation_all/ffpa/fixtures.pt'
binding=ROOT/'leetcuda_rocm/native/ffpa_bindings.cpp'
base_header=ROOT/'artifacts/rocm/performance_phase2/baseline/kernels/rocm/ffpa_attention_hip.cuh'
def build(header,old=False):
 if header.resolve()==ROOT/'kernels/rocm/ffpa_attention_hip.cuh':return ffpa._native(0)
 source=header.read_text().removeprefix('#pragma once\n')+'\n'+_prepare(binding.read_text())
 return _load('ffpa_split_d' if old else 'ffpa_opt',(source,),cache_key(env),env['gpu_architecture'],False)
candidate=build(a.header)
if a.phase=='correct':
 ffpa._native=lambda device:candidate
 ref.run(SimpleNamespace(backend='hip',device=0,fixtures=fixture,output=a.output,dims=ffpa.HEAD_DIMS,extras=True))
 report=json.loads(a.output.read_text());report['candidate_header_sha256']=sha(a.header);report['candidate_header']=str(a.header);report['candidate_binary_sha256']=sha(candidate.__file__);report['candidate_binary']=candidate.__file__;report['optimization_harness_sha256']=sha(__file__);a.output.write_text(json.dumps(report,indent=2)+'\n')
else:
 baseline=build(base_header,True);data=torch.load(fixture,weights_only=True);cases=[]
 report=dict(environment=env,fixture_sha256=sha(fixture),source_sha256={str(p):sha(p) for p in (a.header,base_header,binding)},binaries=[dict(path=l.__file__,sha256=sha(l.__file__)) for l in (candidate,baseline)],harness_sha256=sha(__file__),common_sha256=sha(Path(__file__).resolve().parents[1]/'common.py'),checks=[],performance=[],passed=False)
 for d in ffpa.HEAD_DIMS:
  c=next(c for c in data['cases'] if c['id']==f'random/{d}/17/384');xs=[c[k].cuda() for k in ('q','k','v')]
  for name,kind in ffpa.VARIANTS.items():
   if a.variant and name!=a.variant:continue
   for sq,sv in ref.STAGES:
    funcs=[];meta=dict(d=d,variant=name,sq=sq,sv=sv)
    for lib in (candidate,baseline):
     o=torch.empty_like(xs[0]);l=torch.empty(xs[0].shape[:-1],device='cuda')
     fn=lambda lib=lib,xs=xs,o=o,l=l,kind=kind,sq=sq,sv=sv:lib.forward(*xs,o,l,None,kind,sq,sv,0)
     fn();assert ref.accurate(o,c['expected'])['passed'];assert ref.accurate(l,c['lse'],True)['passed']
     stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g,stream=stream):fn()
     with torch.cuda.stream(stream):o.fill_(float('nan'));l.fill_(float('nan'));g.replay()
     stream.synchronize();assert ref.accurate(o,c['expected'])['passed'];assert ref.accurate(l,c['lse'],True)['passed'];funcs.append(fn)
    cases.append((meta,*funcs));report['checks'].append(dict(**meta,passed=True,graph=True))
 for rid in range(a.rounds):
  for meta,ours,old in cases[rid:]+cases[:rid]:
   warmup(ours,old)
   report['performance'].append(dict(**meta,round=rid,**graph_pair(ours,old,calls=8)))
  a.output.write_text(json.dumps(report,indent=2)+'\n');print('round',rid+1,'complete',flush=True)
 report['gates']={v:gates([r for r in report['performance'] if r['variant']==v],lambda r:(r['d'],r['sq'],r['sv'])) for v in ffpa.VARIANTS if any(r['variant']==v for r in report['performance'])}
 report['passed']=all(c['passed'] for c in report['checks']);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['gates']),flush=True)
