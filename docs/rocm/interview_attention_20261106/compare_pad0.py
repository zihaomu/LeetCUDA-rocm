import torch,json
from pathlib import Path
from leetcuda_rocm.interview_attention_check import accurate
from leetcuda_rocm.cross_validate import sha
r=Path('artifacts/rocm/continuation_all/interview_fa');a=torch.load(r/'hip_pad0.pt',weights_only=True);b=torch.load(r/'cuda_pad0.pt',weights_only=True)
rows=[dict(id=k,**accurate(a[k],b[k]),exact=torch.equal(a[k],b[k])) for k in sorted(a.keys()&b.keys())];out=dict(checks=rows,common=len(rows),passed=bool(rows) and all(x['passed'] for x in rows),hip_sha256=sha(r/'hip_pad0.pt'),cuda_sha256=sha(r/'cuda_pad0.pt'))
(r/'cross_pad0.json').write_text(json.dumps(out,indent=2)+'\n');print(len(rows),out['passed']);assert out['passed']
