from pathlib import Path
import json,torch
from leetcuda_rocm import attention
from leetcuda_rocm.attention_training import attention as native
from leetcuda_rocm.attention_training_check import reference,check
from leetcuda_rocm.cross_validate import sha
from leetcuda_rocm.environment import verify
r=dict(environment=verify(),checks=[],passed=False,source_sha256={p:sha(p) for p in ['leetcuda_rocm/__init__.py','leetcuda_rocm/attention_training.py','leetcuda_rocm/attention_training_check.py',__file__]})
assert attention is native
torch.manual_seed(93)
for dtype in (torch.float16,torch.bfloat16):
 cpu=[torch.randn(*shape).to(dtype) for shape in [(1,4,17,32),(1,2,31,32),(1,2,31,32)]];dy=torch.randn_like(cpu[0])*.25
 for scale in (0.,-.25,.375):
  for bits in (1,2,4,7):
   tensors=[x.cuda().requires_grad_(bool(bits&(1<<i))) for i,x in enumerate(cpu)]
   refs=[x.double().requires_grad_(bool(bits&(1<<i))) for i,x in enumerate(cpu)]
   y=attention(*tensors,scale=scale);yr=reference(*refs,None,False,0,0,0,scale=scale)
   inputs=[x for x in tensors if x.requires_grad];rin=[x for x in refs if x.requires_grad]
   grads=torch.autograd.grad(y,inputs,dy.cuda());gr=torch.autograd.grad(yr,rin,dy.double())
   for i,(out,ref) in enumerate(zip([y,*grads],[yr,*gr])):check(out,ref,dtype,f'{dtype}/{scale}/{bits}/{i}',r['checks'])
 q,k,v=[x.cuda().requires_grad_() for x in cpu]
 y=attention(q,k,v);g=torch.autograd.grad(y,q,dy.cuda(),create_graph=True)[0]
 try:torch.autograd.grad(g.sum(),q)
 except RuntimeError:r['checks'].append(dict(id=str(dtype)+'/double_backward_rejected',passed=True))
 else:r['checks'].append(dict(id=str(dtype)+'/double_backward_rejected',passed=False))
r['passed']=all(c['passed'] for c in r['checks']);Path('artifacts/rocm/continuation_all/attention_training/api.json').write_text(json.dumps(r,indent=2)+'\n');print('passed',r['passed'],'checks',len(r['checks']));assert r['passed']
