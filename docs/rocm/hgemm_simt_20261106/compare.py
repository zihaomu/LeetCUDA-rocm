import json
from pathlib import Path
import torch
from leetcuda_rocm.cross_validate import sha
root=Path('artifacts/rocm/continuation_all/u3');h=json.loads((root/'hip.json').read_text());hv=torch.load(root/'hip.pt',weights_only=True)
r=dict(hip_report_sha256=sha(root/'hip.json'),sources={})
for kind in ('cuda_original','cuda_fixed'):
 p=root/(kind+'.json');c=json.loads(p.read_text());cv=torch.load(p.with_suffix('.pt'),weights_only=True);assert c['fixture_sha256']==h['fixture_sha256'];rows=[]
 for x in c['checks']:
  a,b=cv[x['id']],hv[x['id']];both=x['passed'] and next(z['passed'] for z in h['checks'] if z['id']==x['id'])
  rows.append(dict(id=x['id'],both_reference_passed=both,cuda_reference_passed=x['passed'],parity_passed=torch.allclose(a,b,atol=.01,rtol=.01),bitwise_equal=torch.equal(a,b),max_difference=float((a.float()-b.float()).abs().max()) if torch.isfinite(a).all() else None))
 r['sources'][kind]=dict(report_sha256=sha(p),checks=rows,both_reference_passed=sum(x['both_reference_passed'] for x in rows),parity_when_both_reference_passed=all(x['parity_passed'] for x in rows if x['both_reference_passed']))
(root/'comparison.json').write_text(json.dumps(r,indent=2)+'\n');print({k:{f:v for f,v in s.items() if f!='checks'} for k,s in r['sources'].items()})
