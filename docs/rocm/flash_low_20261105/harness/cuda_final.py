from pathlib import Path
from types import SimpleNamespace
import torch,json,os
from leetcuda_rocm.cuda_pending_check import build
from leetcuda_rocm.cross_validate import ROOT,sha,tensor_sha
from leetcuda_rocm.benchmark import graph_pair

torch.set_num_threads(4);torch.set_grad_enabled(False)
p=ROOT/'artifacts/rocm/continuation_all/b4'
fixtures=torch.load(p/'fixtures_final.pt',weights_only=True)
contracts=json.loads((ROOT/'leetcuda_rocm/flash_low_contracts.json').read_text())
args=SimpleNamespace(source_root=ROOT/'original',cutlass=ROOT/'.cache/cutlass',cuda_current_stream=True)
report=dict(environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),image=os.environ['LEETCUDA_CUDA_IMAGE']),fixture_sha256=sha(p/'fixtures_final.pt'),harness_sha256=sha(__file__),contracts=contracts,builds={},checks=[],performance=[],not_executed=[])
selected=os.environ.get('B4_FUNCTION')
if selected:
 contracts=[c for c in contracts if c['function']==selected]
 p=p/'cuda_final'/selected;p.mkdir(parents=True,exist_ok=True)
values={}
for spec in contracts:
 name=spec['function'];lib,metadata=build(spec['group'],args);fn=getattr(lib,name);report['builds'][name]=metadata
 for c in fixtures:
  d=c['inputs'][0].shape[-1]
  for stage,ds in spec['dimensions'].items():
   if d not in ds:continue
   stage=int(stage);conf=() if spec['argc']==4 else (stage,)
   inputs=[t.cuda() for t in c['inputs']]
   logical_inputs=inputs.copy()
   if name.endswith('_swizzle_qkv') and 'tiling_qkv_' not in name:inputs[2]=inputs[2].transpose(-1,-2).contiguous()
   out=torch.empty_like(inputs[0]);identifier=f"{name}/s{stage}/{c['id']}"
   reason=None
   if name.endswith('_swizzle_qkv') and '_shared_' in name and d==96:
    reason='original_indexing_error: 256//96=2, V row=tid//2 reaches127 while D=96; illegal access reproduced with correct transposed V'
   elif name.endswith('tiling_qk_swizzle_qkv') and inputs[0].shape[2]%d:
    reason='outside_original_shape_contract: original BR=BC=D requires N%D==0'
   if reason:
    report['not_executed'].append(dict(id=identifier,function=name,stage=stage,shape=list(out.shape),executed=False,reason=reason));continue
   report['current_case']=identifier;(p/'cuda_current_case.json').write_text(json.dumps(dict(id=identifier))+'\n')
   fn(*inputs,out,*conf);actual=out.cpu();reference=c['reference'].half()
   ok=torch.allclose(actual,reference,atol=.003,rtol=.003)
   row=dict(id=identifier,function=name,stage=stage,shape=list(out.shape),passed=ok,atol=.003,rtol=.003,max_abs_error=(actual.double()-c['reference']).abs().max().item(),input_sha256=c['input_sha256'],packed_input_sha256=[tensor_sha(t.cpu()) for t in inputs])
   report['checks'].append(row);values[identifier]=actual
   (p/'cuda_audit_progress.json').write_text(json.dumps(report,indent=2)+'\n')
   if not ok:print('FAIL',identifier,row['max_abs_error'],flush=True)
   if c['performance']:
    import torch.nn.functional as F
    old=out.clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):fn(*inputs,out,*conf)
    with torch.cuda.stream(stream):out.fill_(float('nan'));graph.replay()
    stream.synchronize();gp=torch.equal(out,old)
    perf=graph_pair(lambda:fn(*inputs,out,*conf),lambda:F.scaled_dot_product_attention(*logical_inputs),calls=8,sample_ms=10,trials=3)
    report['performance'].append(dict(id=identifier,function=name,stage=stage,shape=list(out.shape),accuracy_passed=ok,graph_replay_verified=gp,**perf))
 print(name,'complete',flush=True)
 report['passed']=all(c['passed'] for c in report['checks']);(p/'cuda_audit_progress.json').write_text(json.dumps(report,indent=2)+'\n')
torch.save(values,p/'cuda_audit.pt');report['outputs_sha256']=sha(p/'cuda_audit.pt');(p/'cuda_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('DONE',len(report['checks']),sum(c['passed'] for c in report['checks']),flush=True)
