import json
from pathlib import Path
import torch
from leetcuda_rocm import attention_training as api, attention_training_check as ref
from leetcuda_rocm.environment import verify
from leetcuda_rocm.cross_validate import sha
torch.set_num_threads(4)
folder=Path('artifacts/rocm/performance_phase2/training/delta8')
report=dict(environment=verify(),source_sha256=sha(api.__file__),harness_sha256=sha(__file__),checks=[],passed=False)
cases=[]
for dtype in (torch.float16,torch.bfloat16):
 for nq,nk,kind,causal,p,pattern in (
  (65,1,'none',True,0.,'normal'),(65,1,'none',False,.25,'normal'),
  (65,129,'none',False,0.,'fully_masked'),(65,129,'add',True,.25,'normal'),
  (0,17,'none',False,0.,'normal'),(17,0,'none',False,0.,'normal'),
  (65,129,'bool',True,.25,'large_logits'),(65,129,'none',False,0.,'zero_qk')):
  name=f'{dtype}/{nq},{nk}/{kind}/{causal}/{p}/{pattern}'
  c=ref.fixture_case(name,dtype,256,(1,4,1,nq,nk),kind,causal,p,97,pattern)
  cases.append(c);ref.evaluate(c,torch.device('cuda',0),report['checks'])
torch.save(dict(cases=cases),folder/'d256_edges_fixtures.pt')
report['fixture_sha256']=sha(folder/'d256_edges_fixtures.pt');report['passed']=all(c['passed'] for c in report['checks']);(folder/'d256_edges.json').write_text(json.dumps(report,indent=2)+'\n');print(len(report['checks']),report['passed']);assert report['passed']
