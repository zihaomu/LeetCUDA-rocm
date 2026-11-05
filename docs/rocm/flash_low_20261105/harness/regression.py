import torch,json,importlib.util
from pathlib import Path
from leetcuda_rocm.loader import load,source_paths
from leetcuda_rocm.environment import ROOT,verify
from leetcuda_rocm.cross_validate import sha
from leetcuda_rocm.benchmark import graph_pair
from leetcuda_rocm.flash_swizzle_check import telemetry
p=ROOT/'artifacts/rocm/continuation_all/b4';identity=verify();torch.set_num_threads(4)
baseline=json.loads((ROOT/'docs/rocm/flash_rr_20261105/hip.json').read_text());binary=ROOT/baseline['isa']['binary'];assert sha(binary)==baseline['isa']['binary_sha256']
spec=importlib.util.spec_from_file_location(Path(baseline['isa']['build_cache_binary']).stem,binary);frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
current=load('flash_attn');fixtures=torch.load(p/'fixtures_final.pt',weights_only=True);checks=[];cases=[]
for name in sorted(n for n in dir(frozen) if n.startswith('flash_attn_')):
 for c in fixtures:
  d=c['inputs'][0].shape[-1]
  if d==16 or ('_shared_' in name and d>256):continue
  for stage in (1,2):
   if '_shared_' in name and d==256 and stage==2 and not name.endswith('_shared_qkv_acc_f32_rr'):continue
   inputs=[t.cuda() for t in c['inputs']];old=torch.empty_like(inputs[0]);new=torch.empty_like(old)
   getattr(frozen,name)(*inputs,old,stage);getattr(current,name)(*inputs,new,stage)
   checks.append(dict(id=f"{name}/{c['id']}/s{stage}",function=name,passed=torch.equal(old,new),bitwise_equal=torch.equal(old,new)))
   if c['performance']:cases.append((name,c,stage))
 print(name,'regression',flush=True)
r=dict(environment=identity,harness_sha256=sha(__file__),source_sha256={str(s.relative_to(ROOT)):sha(s) for s in source_paths('flash_attn')},baseline_binary_sha256=sha(binary),current_binary_sha256=sha(current.__file__),checks=checks,performance=[],telemetry=[])
for round_id in range(3):
 before=telemetry('hip')
 for name,c,stage in cases[round_id:]+cases[:round_id]:
  inputs=[t.cuda() for t in c['inputs']];old=torch.empty_like(inputs[0]);new=torch.empty_like(old)
  pair=graph_pair(lambda:getattr(current,name)(*inputs,new,stage),lambda:getattr(frozen,name)(*inputs,old,stage),calls=8,sample_ms=20,trials=3)
  r['performance'].append(dict(id=f"{name}/{c['id']}/s{stage}/r{round_id}",function=name,round=round_id,within_ten_percent=pair['ours_ms']<=pair['baseline_ms']*1.1,**pair))
 r['telemetry'].append(dict(before=before,after=telemetry('hip')));print('round',round_id,flush=True)
r['passed']=all(c['passed'] for c in checks) and all(c['within_ten_percent'] for c in r['performance']);(p/'existing_regression.json').write_text(json.dumps(r,indent=2)+'\n');print(r['passed'],len(checks),len(r['performance']),sum(not c['within_ten_percent'] for c in r['performance']),flush=True)
