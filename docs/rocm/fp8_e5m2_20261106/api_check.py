from pathlib import Path
import importlib.util,sys,json,torch,numpy as np
from leetcuda_rocm.environment import verify
from leetcuda_rocm.cross_validate import sha
root=Path('artifacts/rocm/continuation_all/e5m2')
from leetcuda_rocm import fp8 as api, fp8_check as ref
torch.cuda.set_device(1);torch.set_grad_enabled(False);torch.set_num_threads(4);env=verify();checks=[]
torch.manual_seed(93);a=(torch.randn(129,80,device='cuda')*.25).bfloat16();b=(torch.randn(80,136,device='cuda')*.25).bfloat16()
for fmt in ('e4m3','e5m2'):
 ref.configure(fmt)
 for mode in api.MODES:
  row,col=mode.startswith('row'),mode.endswith('col');aq,sa=ref.quantize_reference(a,row);bq,sb=ref.quantize_reference(b.T,col);expected=ref.oracle(aq,bq,sa,sb,row,col,a.device);w=api.quantize_fp8_weight(b,per_col=col,fp8_format=fmt)
  assert w.fp8_format==fmt and np.array_equal(w.data.cpu().numpy(),bq)
  for ws in (False,True):
   out=api.fp8_gemm(a,b,mode=mode,use_ws=ws,fp8_format=fmt);offline=api.fp8_gemm_prepacked(a,w,per_row=row,use_ws=ws)
   checks.append(dict(format=fmt,mode=mode,use_ws=ws,passed=torch.allclose(out,expected,atol=ref.ATOL,rtol=ref.RTOL) and torch.equal(out,offline)))
  if fmt=='e4m3':checks.append(dict(format=fmt,mode=mode,kind='default_compatibility',passed=torch.equal(api.fp8_gemm(a,b,mode=mode),api.fp8_gemm(a,b,mode=mode,fp8_format=fmt))))
 for value in ('fnuz',None,0):
  try:api.fp8_gemm(a,b,fp8_format=value)
  except ValueError:checks.append(dict(format=fmt,kind='invalid_format',value=value,passed=True))
  else:checks.append(dict(format=fmt,kind='invalid_format',value=value,passed=False))
 print(fmt,sum(not c['passed'] for c in checks),flush=True)
r=dict(environment=env,passed=all(c['passed'] for c in checks),private_candidate=False,checks=checks,source_sha256={str(f):sha(f) for f in (Path(__file__),Path('leetcuda_rocm/fp8.py'),Path('kernels/rocm/fp8_gemm_hip.cuh'),Path('leetcuda_rocm/fp8_check.py'))});(root/'api.json').write_text(json.dumps(r,indent=2)+'\n');print('DONE',r['passed'],len(checks),flush=True)
