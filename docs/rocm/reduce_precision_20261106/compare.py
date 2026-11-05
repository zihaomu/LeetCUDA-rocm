import json
from pathlib import Path
import torch
from leetcuda_rocm.cross_validate import sha
root=Path('artifacts/rocm/continuation_all/u2')
h=json.loads((root/'hip.json').read_text())
hv=torch.load(root/'hip.pt',weights_only=True)
result={'hip_report_sha256':sha(root/'hip.json'),'sources':{}}
for kind in ('cuda',):
 p=root/(kind+'.json'); c=json.loads(p.read_text())
 cv=torch.load(p.with_suffix('.pt'),weights_only=True)
 assert c['fixture_sha256']==h['fixture_sha256']
 rows=[]
 for x in c['checks']:
  a,b=cv[x['id']],hv[x['id']]
  rows.append(dict(id=x['id'],cuda_reference_passed=x['passed'],hip_reference_passed=next(z['passed'] for z in h['checks'] if z['id']==x['id']),parity_passed=torch.allclose(a,b,atol=.15,rtol=.01),bitwise_equal=torch.equal(a,b),max_difference=float((a.float()-b.float()).abs().max())))
 # API precision changed deliberately; never count an upstream failure as cross-platform success.
 result['sources'][kind]=dict(report_sha256=sha(p),checks=rows,both_reference_passed=sum(x['cuda_reference_passed'] and x['hip_reference_passed'] for x in rows),parity_when_both_reference_passed=all(x['parity_passed'] for x in rows if x['cuda_reference_passed'] and x['hip_reference_passed']))
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:{f:v for f,v in r.items() if f!='checks'} for k,r in result['sources'].items()})
