import json,torch
from pathlib import Path
import leetcuda_rocm
from leetcuda_rocm import interview_flash_attention
from leetcuda_rocm.cross_validate import sha
from leetcuda_rocm.interview_attention import VARIANTS
rows=[]
for name in VARIANTS:
 q=torch.zeros(1,2,65,64,device='cuda',dtype=torch.float16)
 k=torch.zeros(1,1 if name=='persist_d' else 2,137 if name=='persist_d' else 65,64,device='cuda',dtype=torch.float16);v=torch.full_like(k,.125)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  o,l=interview_flash_attention(q,k,v,variant=name,return_lse=True)
  stream.synchronize();ok=bool((o==.125).all()) and bool(torch.allclose(l,torch.full_like(l,torch.log(torch.tensor(k.shape[-2],dtype=torch.float32)).item()),atol=2e-5,rtol=2e-5))
  rows.append(dict(id=name,passed=ok))
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):leetcuda_rocm.interview_flash_attention(q,k,v,variant=name,out=o,lse=l)
  o.fill_(float('nan'));l.fill_(float('nan'));g.replay();stream.synchronize();rows.append(dict(id=name+'/graph',passed=bool((o==.125).all()) and bool(torch.isfinite(l).all())))
report=dict(checks=rows,passed=all(x['passed'] for x in rows),sources={p:sha(Path(p)) for p in ['leetcuda_rocm/__init__.py','leetcuda_rocm/interview_attention.py']},harness_sha256=sha(__file__))
p=Path('artifacts/rocm/continuation_all/interview_fa/public_api.json');p.write_text(json.dumps(report,indent=2)+'\n');print(len(rows),report['passed']);assert report['passed']
