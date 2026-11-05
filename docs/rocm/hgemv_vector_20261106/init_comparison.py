import torch,json
from pathlib import Path
from leetcuda_rocm.cross_validate import sha
r=Path('artifacts/rocm/continuation_all/u1');a=torch.load(r/'cuda_original.pt',weights_only=True);b=torch.load(r/'cuda_fixed.pt',weights_only=True)
checks=[dict(id=k,bitwise_equal=torch.equal(v,b[k])) for k,v in a.items()]
(r/'initialization_comparison.json').write_text(json.dumps(dict(checks=checks,bitwise_count=sum(c['bitwise_equal'] for c in checks),report_sha256={k:sha(r/(k+'.json')) for k in ('cuda_original','cuda_fixed')}),indent=2)+'\n');print('unchanged',sum(c['bitwise_equal'] for c in checks),len(checks))
