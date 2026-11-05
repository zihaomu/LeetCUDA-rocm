from pathlib import Path
import json,torch
from leetcuda_rocm import ffpa_attention
from leetcuda_rocm.ffpa import ffpa_attention as implementation,VARIANTS
from leetcuda_rocm.ffpa_check import accurate
from leetcuda_rocm.cross_validate import sha,tensor_sha
from leetcuda_rocm.environment import ROOT,verify
root=ROOT/'artifacts/rocm/continuation_all/ffpa';torch.set_grad_enabled(False);torch.set_num_threads(8);env=verify()
a=torch.load(root/'fixtures.pt',weights_only=True);b=torch.load(root/'fixtures_regenerated.pt',weights_only=True);checks=[]
assert len(a['cases'])==len(b['cases'])==96
for c,d in zip(a['cases'],b['cases']):
 assert c['id']==d['id']
 for key in ('q','k','v','expected','lse'):
  checks.append(dict(id='regenerate/'+c['id']+'/'+key,passed=torch.equal(c[key],d[key]),tensor_sha256=tensor_sha(c[key])))
checks.append(dict(id='public_export',passed=ffpa_attention is implementation))
c=a['cases'][0];xs=[c[key].cuda() for key in ('q','k','v')]
for variant in VARIANTS:
 o=torch.empty_like(xs[0]);l=torch.empty(xs[0].shape[:-1],device='cuda')
 returned=ffpa_attention(*xs,variant=variant,out=o,lse=l,return_lse=True)
 checks.append(dict(id='output_identity/'+variant,passed=returned[0] is o and returned[1] is l and accurate(o,c['expected'])['passed'] and accurate(l,c['lse'],True)['passed']))
for d in (64,192,512,1024):
 q=torch.zeros((1,1,16,d),device='cuda',dtype=torch.float16);k=torch.zeros((1,1,2048,d),device='cuda',dtype=torch.float16);v=torch.full_like(k,2**-11);v[:,:,:64]=1
 for variant in VARIANTS:
  out=ffpa_attention(q,k,v,variant=variant)
  checks.append(dict(id=f'fp32_output_state/{d}/{variant}',passed=torch.equal(out,torch.full_like(out,.03173828125)),expected=.03173828125,fp16_state_control=.03125))
for width in (32,64):
 for rows in (64,80):
  addresses=[r*width+(c^((r&3)*8)) for r in range(rows) for c in range(width)]
  for stages in (1,2,3):
   combined=[s*rows*width+a for s in range(stages) for a in addresses]
   checks.append(dict(id=f'layout/{rows}/{width}/{stages}',passed=len(set(combined))==rows*width*stages and min(combined)==0 and max(combined)==rows*width*stages-1))
r=dict(passed=all(x['passed'] for x in checks),checks=checks,environment=env,fixture_sha256=sha(root/'fixtures.pt'),regenerated_sha256=sha(root/'fixtures_regenerated.pt'),generator_metadata_note='Container and generator source metadata differ; all 480 tensor payloads regenerated bitwise identically from current checked-in harness.',sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'leetcuda_rocm/ffpa_check.py',ROOT/'leetcuda_rocm/ffpa.py',ROOT/'leetcuda_rocm/__init__.py')})
(root/'api_regeneration.json').write_text(json.dumps(r,indent=2)+'\n');print('DONE',r['passed'],len(checks));assert r['passed']
