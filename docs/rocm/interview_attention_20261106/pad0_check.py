import argparse,json,torch
from pathlib import Path
from leetcuda_rocm.cross_validate import sha
from leetcuda_rocm.interview_attention_check import accurate
p=argparse.ArgumentParser();p.add_argument('--backend',choices=('hip','cuda'),required=True);a=p.parse_args();torch.set_grad_enabled(False)
root=Path('artifacts/rocm/continuation_all/interview_fa');data=torch.load(root/'fixtures.pt',weights_only=True);rows=[];unavailable=[];outputs={};builds=[]
for d in (32,64,128):
 if a.backend=='cuda':
  from leetcuda_rocm.interview_attention_cuda import build
  lib,meta=build(0,d,slotfix=True);builds.append(meta)
 else:
  from leetcuda_rocm.interview_attention import _native
  lib=_native(0)
 for n in (128,512):
  case=next(c for c in data['cases'] if c['id']==f'random/{d}/17/{n}');q,k,v=[case[x].cuda() for x in ('q','k','v')];o=torch.empty_like(q);lse=torch.empty(q.shape[:-1],device='cuda');ref=case['expected']['standard']['o'];maps=lib.prepare(q,k,v) if a.backend=='cuda' else None
  for sk in (1,2,3,4):
   key=f'pad0/d{d}/n{n}/s{sk}'
   fn=(lambda:lib.forward(q,k,v,o,maps,sk,1,0,False,d**-.5)) if a.backend=='cuda' else (lambda:lib.forward(q,k,v,o,lse,0,sk,1,0,False,d**-.5,0,None,0))
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    try:fn();stream.synchronize()
    except RuntimeError as e:
     if a.backend=='cuda' and 'shared-memory requirement' in str(e):unavailable.append(dict(id=key,reason=str(e)));continue
     raise
    got=o.cpu();rows.append(dict(id=key,**accurate(got,ref)));outputs[key]=got
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):fn()
    o.fill_(float('nan'));g.replay();stream.synchronize();rows.append(dict(id=key+'/graph',passed=torch.equal(got,o.cpu())));outputs[key+'/graph']=o.cpu()
 torch.cuda.synchronize()
report=dict(backend=a.backend,builds=builds,fixture_sha256=sha(root/'fixtures.pt'),harness_sha256=sha(__file__),checks=rows,unavailable=unavailable,passed=all(r['passed'] for r in rows));out=root/(a.backend+'_pad0.json');torch.save(outputs,out.with_suffix('.pt'));report['outputs_sha256']=sha(out.with_suffix('.pt'));out.write_text(json.dumps(report,indent=2)+'\n');print('checks',len(rows),'unavailable',len(unavailable),'passed',report['passed']);assert report['passed']
