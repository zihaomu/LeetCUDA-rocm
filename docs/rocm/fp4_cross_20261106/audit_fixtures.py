from pathlib import Path
import json,torch
from leetcuda_rocm.cross_validate import sha,tensor_sha
root=Path.cwd();a=root/'artifacts/rocm/continuation_all/fp4_cross'
x,y=(torch.load(a/n,map_location='cpu',weights_only=True) for n in ('fixtures.pt','regenerated.pt'))
checks=[]
assert len(x['cases'])==len(y['cases'])==13
for c,d in zip(x['cases'],y['cases']):
 assert c['id']==d['id']
 for key in ('a','b'):checks.append(dict(id=c['id']+'/'+key,passed=tensor_sha(c[key])==tensor_sha(d[key])==c['input_sha256'][key]))
 for mode in c['modes']:
  for key,tensor in c['modes'][mode].items():checks.append(dict(id=c['id']+'/'+mode+'/'+key,passed=tensor_sha(tensor)==tensor_sha(d['modes'][mode][key])))
checks.append(dict(id='generator_sha256',passed=x['generator_sha256']==y['generator_sha256']==sha(root/'leetcuda_rocm/fp4_cross_check.py')))
checks.append(dict(id='reference_sha256',passed=x['reference_sha256']==y['reference_sha256']==sha(root/'leetcuda_rocm/fp4_check.py')))
report=dict(passed=all(c['passed'] for c in checks),checks=checks,files={n:sha(a/n) for n in ('fixtures.pt','regenerated.pt')})
(a/'fixture_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('FIXTURE AUDIT',report['passed'],len(checks));assert report['passed']
