import argparse,importlib.util,json,sys,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from leetcuda_rocm import attention_training as api,attention_training_check as ref
from leetcuda_rocm.environment import ROOT,verify
from leetcuda_rocm.cross_validate import sha,tensor_sha
from leetcuda_rocm.benchmark import graph_pair
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import warmup,gates
p=argparse.ArgumentParser();p.add_argument('phase',choices=['correct','performance']);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=1);p.add_argument('--d256-only',action='store_true');a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(__file__,a.output.with_name(a.output.stem+'_harness.py'));torch.set_num_threads(4);env=verify()
def load(path,name):
 if path.resolve()==ROOT/'leetcuda_rocm/attention_training.py':return api
 name='leetcuda_rocm.'+name;spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
candidate=load(a.source,'phase2_candidate');basepath=ROOT/'artifacts/rocm/performance_phase2/baseline/leetcuda_rocm/attention_training.py'
if a.phase=='correct':
 ref.implementation=candidate
 result=ref.run(SimpleNamespace(backend='hip',device=0,guard_device=1,fixtures=ROOT/'artifacts/rocm/continuation_all/attention_training/fixtures.pt',output=a.output))
 raise SystemExit(result)
baseline=load(basepath,'phase2_baseline');cases=[]
report=dict(environment=env,source_sha256={str(p):sha(p) for p in (a.source,basepath)},harness_sha256=sha(__file__),common_sha256=sha(Path(__file__).resolve().parents[1]/'common.py'),checks=[],performance=[],isa={'candidate':{},'baseline':{}},memory=[],passed=False)
for label,mod in [('candidate',candidate),('baseline',baseline)]:
 for name in ('_forward','_delta','_backward_q','_backward_kv'):
  setattr(mod,name,ref.Capture(getattr(mod,name),name,report['isa'][label],a.output.parent/(a.output.stem+'_isa')/label,'hip'))
for dtype in (torch.float16,torch.bfloat16):
 for n,d,hk,causal in ((256,64,1,False),(512,128,2,True),(1024,256,4,True)):
  if a.d256_only and d!=256:continue
  gen=torch.Generator().manual_seed(20261106+n+d);cpu=[(torch.randn(shape,generator=gen)*.5).to(dtype) for shape in ((1,4,n,d),(1,hk,n,d),(1,hk,n,d))];dy=(torch.randn(cpu[0].shape,generator=gen)*.25).to(dtype)
  leaves=[x.double().requires_grad_() for x in cpu];eo=ref.reference(*leaves,None,causal,0,0,0);expected=(eo,*torch.autograd.grad(eo,leaves,dy.double()));gpu=[x.cuda() for x in cpu];dout=dy.cuda()
  def operation(mod,backward,gpu=gpu,dout=dout,causal=causal):
   def call():
    leaves=tuple(x.detach().requires_grad_() for x in gpu);o=mod.attention(*leaves,causal=causal)
    return (o,*torch.autograd.grad(o,leaves,dout)) if backward else o
   return call
  for label,mod in [('candidate',candidate),('baseline',baseline)]:
   values=operation(mod,True)()
   for name,value,exp in zip(('out','dq','dk','dv'),values,expected):ref.check(value.cpu(),exp,dtype,f'{label}/{dtype}/{n}/{d}/{name}',report['checks'])
  assert all(c['passed'] for c in report['checks'])
  for backward in (False,True):
   fns=[operation(mod,backward) for mod in (candidate,baseline)]
   meta=dict(dtype=str(dtype),n=n,d=d,hk=hk,causal=causal,path='forward_backward' if backward else 'forward',inputs_sha256=[tensor_sha(x) for x in [*cpu,dy]])
   for label,fn in zip(('candidate','baseline'),fns):
    prior=fn();prior=(prior,) if not backward else prior;prior=tuple(x.detach().clone() for x in prior)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):got=fn()
    got=(got,) if not backward else got
    with torch.cuda.stream(stream):
     for x in got:x.detach().fill_(float('nan'))
     g.replay()
    stream.synchronize();assert all(torch.equal(x,y) for x,y in zip(prior,got))
    report['checks'].append(dict(id=f'{label}/{dtype}/{d}/{backward}/graph',passed=True))
   cases.append((meta,*fns))
  fn=operation(candidate,True);torch.cuda.synchronize();start=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();result=fn();torch.cuda.synchronize();extra=torch.cuda.max_memory_allocated()-start
  linear=(2*gpu[0].numel()+gpu[1].numel()+gpu[2].numel())*gpu[0].element_size()+2*gpu[0].shape[0]*gpu[0].shape[1]*n*4
  assert extra<=2*linear+65536;report['memory'].append(dict(dtype=str(dtype),d=d,n=n,peak_bytes=extra,limit=2*linear+65536,passed=True));del result
for rid in range(a.rounds):
 for meta,ours,old in cases[rid:]+cases[:rid]:
  warmup(ours,old)
  report['performance'].append(dict(**meta,round=rid,**graph_pair(ours,old,calls=8)))
 a.output.write_text(json.dumps(report,indent=2)+'\n');print('round',rid+1,'complete',flush=True)
key=lambda r:(r['dtype'],r['n'],r['d'],r['path'])
report['gates']={path:gates([r for r in report['performance'] if r['path']==path and r['d']==256],key) for path in ('forward','forward_backward')};report['passed']=all(c['passed'] for c in report['checks']);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['gates']),flush=True)
