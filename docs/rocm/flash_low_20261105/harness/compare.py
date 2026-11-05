import torch,json
from pathlib import Path
from leetcuda_rocm.cross_validate import sha
p=Path('artifacts/rocm/continuation_all/b4')
hip=json.loads((p/'hip.json').read_text());actual=torch.load(p/'hip.pt',weights_only=True);by_id={c['id']:c for c in hip['checks']}
cuda=dict(checks=[],performance=[],not_executed=[],builds={},reports={})
comparisons=[]
for path in sorted((p/'cuda_evidence/cuda_final').glob('*/cuda_audit.json')):
 r=json.loads(path.read_text());assert r['fixture_sha256']==hip['fixture_sha256']
 values=torch.load(path.with_suffix('.pt'),weights_only=True)
 assert sha(path.with_suffix('.pt'))==r['outputs_sha256']
 cuda['reports'][path.parent.name]=dict(path=str(path),sha256=sha(path),environment=r['environment'],harness_sha256=r['harness_sha256'])
 cuda['builds'].update(r['builds'])
 for key in ('checks','performance','not_executed'):cuda[key]+=r[key]
 for c in r['checks']:
  h=by_id[c['id']];assert c['input_sha256']==h['input_sha256'];assert c['packed_input_sha256']==h['packed_input_sha256']
  x=actual[c['id']];y=values[c['id']]
  parity=torch.allclose(x,y,atol=.003,rtol=.003)
  comparisons.append(dict(id=c['id'],function=c['function'],hip_accuracy_passed=h['passed'],cuda_accuracy_passed=c['passed'],raw_parity_passed=parity,bitwise_equal=torch.equal(x,y),max_abs_difference=(x.float()-y.float()).abs().max().item(),passed=h['passed'] and (parity if c['passed'] else True),original_precision_failure=not c['passed']))
report=dict(hip_sha256=sha(p/'hip.json'),harness_sha256=sha(__file__),checks=comparisons,not_executed=cuda['not_executed'],summary=dict(planned=len(hip['checks']),hip_passed=sum(c['passed'] for c in hip['checks']),cuda_executed=len(comparisons),cuda_accuracy_passed=sum(c['cuda_accuracy_passed'] for c in comparisons),cuda_numerical_failures=sum(not c['cuda_accuracy_passed'] for c in comparisons),qualified_pair_count=sum(c['cuda_accuracy_passed'] for c in comparisons),qualified_parity_passed=sum(c['cuda_accuracy_passed'] and c['raw_parity_passed'] for c in comparisons),raw_parity_passed=sum(c['raw_parity_passed'] for c in comparisons),bitwise_equal=sum(c['bitwise_equal'] for c in comparisons),not_executed=len(cuda['not_executed'])),passed=hip['passed'] and all(c['passed'] for c in comparisons))
(p/'cuda_original.json').write_text(json.dumps(cuda,indent=2)+'\n');(p/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['summary'],report['passed'])
for c in comparisons:
 if not c['passed']:print(c)
