import torch,json,sys
from pathlib import Path
from leetcuda_rocm.loader import _load,_prepare,load
from leetcuda_rocm.environment import cache_key,verify
from leetcuda_rocm.tf32_check import STATIC,DYNAMIC,configs,numerical,isa
from leetcuda_rocm.benchmark import graph_pair
root=Path('artifacts/rocm/continuation_all/b6/tuning');kind=sys.argv[1];identity=verify();names=[STATIC,DYNAMIC]
b='#include <torch/extension.h>\n'+''.join('void '+n+'(torch::Tensor,torch::Tensor,torch::Tensor,int,bool,int);' for n in names)+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){'+''.join('m.def("'+n+'",&'+n+');' for n in names)+'}\n'
lib=_load('tf32_tune_'+kind,tuple(_prepare(x) for x in (b,(root/(kind+'.cu')).read_text())),cache_key(identity),identity['gpu_architecture'],False)
base=load('sgemm');r=dict(checks=[],comparisons=[],performance=[]);torch.set_num_threads(4)
for n in names:
 for cfg in configs(n):
  for case in torch.load(root.parent/'fixtures.pt',weights_only=True):
   a,b=case['a'].cuda(),case['b'].cuda();out=numerical(getattr(lib,n),a,b,cfg,r['checks'],n+str(cfg)+case['id'],case['reference']);old=torch.empty_like(out,device='cuda');getattr(base,n)(a,b,old,*cfg);r['comparisons'].append(dict(id=n+str(cfg)+case['id'],passed=torch.equal(out,old.cpu())))
print('checks',len(r['checks']),sum(not c['passed'] for c in r['checks']),flush=True)
r['isa']=isa(lib,root/(kind+'_isa'));print(r['isa']['kernels'],flush=True)
for size in (256,1024,2048):
 a,b=[torch.randn(size,size,device='cuda')*.25 for _ in range(2)];o=torch.empty_like(a);old=torch.empty_like(a)
 for n in names:
  x=graph_pair(lambda:getattr(lib,n)(a,b,o,2,False,256),lambda:getattr(base,n)(a,b,old,2,False,256),calls=8,sample_ms=20,trials=5)
  r['performance'].append(dict(function=n,size=size,**x));print(n,size,x['baseline_over_ours'],flush=True)
r['passed']=all(x['passed'] for k in ('checks','comparisons') for x in r[k]) and r['isa']['passed'];(root/(kind+'.json')).write_text(json.dumps(r,indent=2)+'\n')
