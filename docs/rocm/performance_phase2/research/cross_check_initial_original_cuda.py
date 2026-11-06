"""Compare optimized AMD outputs with the previously frozen 5090 outputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from leetcuda_rocm import interview_attention_check as fa,fp8_cross_check as fp8,ffpa_check as ffpa,attention_training_check as training
from leetcuda_rocm.environment import ROOT
from leetcuda_rocm.cross_validate import sha
torch.set_num_threads(4)
root=ROOT/'artifacts/rocm/performance_phase2';old=ROOT/'artifacts/rocm/continuation_all'
manifest=json.loads((old/'interview_fa/cross_matched.json').read_text())
assert sha(old/'interview_fa/cuda_matched.pt')==manifest['cuda_sha256']
fa.compare(SimpleNamespace(hip=root/'fa/packed_lds/canonical_correctness.pt',cuda=old/'interview_fa/cuda_matched.pt',output=root/'fa/packed_lds/cross.json'))
assert json.loads((root/'fa/packed_lds/cross.json').read_text())['common']==1280
fp8.compare(SimpleNamespace(hip=root/'fp8/tiled_quantization/shared_hip.json',cuda=old/'e5m2/shared_cuda.json',output=root/'fp8/tiled_quantization/cross.json'))
prior=json.loads((old/'ffpa/cuda.json').read_text())
assert sha(old/'ffpa/cuda.pt')==prior['output_sha256']
ffpa.compare(SimpleNamespace(hip=root/'ffpa/packed_lds/canonical_correctness.pt',cuda=old/'ffpa/cuda.pt',output=root/'ffpa/packed_lds/cross.json'))
paths=[root/'training/delta8/canonical_correctness.json',old/'attention_training/cuda_recomputed.json']
reports=[json.loads(p.read_text()) for p in paths];baseline=json.loads((root/'baseline/manifest.json').read_text())
assert reports[1]['implementation_sha256']==baseline['files']['leetcuda_rocm/attention_training.py']
assert reports[0]['fixture_sha256']==reports[1]['fixture_sha256'];assert all(r['passed'] for r in reports)
for p,r in zip(paths,reports):assert sha(p.with_suffix('.pt'))==r['outputs_sha256']
outputs=[torch.load(p.with_suffix('.pt'),weights_only=True) for p in paths];assert outputs[0].keys()==outputs[1].keys();checks=[]
for key in outputs[0]:
 a,b=[o[key] for o in outputs];training.check(a,b,a.dtype,key,checks);checks[-1]['exact']=torch.equal(a,b)
result=dict(passed=all(c['passed'] for c in checks),checks=checks,fixture_sha256=reports[0]['fixture_sha256'],implementation_sha256=[r['implementation_sha256'] for r in reports],report_sha256={str(p):sha(p) for p in paths},harness_sha256=sha(__file__),scope='Optimized AMD delta launch compared with frozen prior Triton CUDA algorithm; different source hashes explicitly retained. No new NVIDIA execution in this phase.')
(root/'training/delta8/cross.json').write_text(json.dumps(result,indent=2)+'\n');print('training',len(checks),result['passed']);assert result['passed']
