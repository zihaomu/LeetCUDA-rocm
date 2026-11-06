import argparse, json, sys, shutil
from pathlib import Path
import torch
from leetcuda_rocm import fp8, fp8_check as ref
from leetcuda_rocm.loader import _load, _prepare
from leetcuda_rocm.environment import ROOT, verify, cache_key
from leetcuda_rocm.cross_validate import sha, tensor_sha
from leetcuda_rocm.benchmark import graph_pair
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import warmup, gates

p=argparse.ArgumentParser();p.add_argument('--header',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=3);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,a.output.with_name(a.output.stem+'_harness.py'))
torch.set_num_threads(4);torch.set_grad_enabled(False);env=verify()
baseline_header=ROOT/'artifacts/rocm/performance_phase2/baseline/kernels/rocm/fp8_gemm_hip.cuh'
binding=ROOT/'leetcuda_rocm/native/fp8_bindings.cpp'
def build(header,fmt,old=False):
    if header.resolve()==ROOT/'kernels/rocm/fp8_gemm_hip.cuh':return fp8._native(0,fmt)
    source=header.read_text().removeprefix('#pragma once\n')+'\n'+_prepare(binding.read_text())
    if fmt=='e5m2':source='#define LEETCUDA_FP8_E5M2 1\n'+source
    return _load(('fp8_pipeline_'+fmt) if old else ('fp8_opt_'+fmt),(source,),cache_key(env),env['gpu_architecture'],False)
report=dict(environment=env,source_sha256={str(x):sha(x) for x in (a.header,baseline_header,binding)},harness_sha256=sha(__file__),common_sha256=sha(Path(__file__).resolve().parents[1]/'common.py'),builds={},checks=[],performance=[],passed=False)
cases=[]
for fmt in ('e4m3','e5m2'):
    ref.configure(fmt);libs=[build(a.header,fmt),build(baseline_header,fmt,True)]
    report['builds'][fmt]=[dict(binary=l.__file__,sha256=sha(l.__file__)) for l in libs]
    for m,n,k in ((256,256,256),(256,256,1024),(65,136,80)):
        gen=torch.Generator().manual_seed(73);cpu=[(torch.randn(s,generator=gen)*.25).bfloat16() for s in ((m,k),(k,n))];x,y=[v.cuda() for v in cpu]
        prefix=dict(format=fmt,shape=[m,n,k],input_sha256=[tensor_sha(v) for v in cpu])
        for col in (False,True):
            bt,sb=ref.quantize_reference(cpu[1].T,col);pairs=[(torch.empty((n,k),dtype=torch.uint8,device='cuda'),torch.empty(len(sb),device='cuda')) for _ in libs]
            funcs=[lambda l=l,t=t,s=s,c=col,y=y:l.quantize_b(y,t,s,c) for l,(t,s) in zip(libs,pairs)]
            for fn,(t,s) in zip(funcs,pairs):
                fn();assert torch.equal(t.cpu(),torch.from_numpy(bt));assert torch.equal(s.cpu(),torch.from_numpy(sb))
            cases.append((dict(**prefix,path='quantize_b',mode='col' if col else 'block',ws=False),*funcs))
            report['checks'].append(dict(**cases[-1][0],bitwise_codes_and_scale=True,passed=True))
        # Small unchanged dispatch is covered above; full GEMM timings focus on
        # the two fixed production workloads, all modes and both schedules.
        if k<256:continue
        for mode in ref.MODES:
            row,col=mode.startswith('row'),mode.endswith('col')
            ab,sa=ref.quantize_reference(cpu[0],row);bt,sb=ref.quantize_reference(cpu[1].T,col)
            expected=ref.oracle(ab,bt,sa,sb,row,col,'cpu')
            a8,b8,sa8,sb8=[torch.from_numpy(z).cuda() for z in (ab,bt,sa,sb)]
            for ws in (False,True):
                outputs=[torch.empty((m,n),device='cuda',dtype=torch.bfloat16) for _ in libs]
                work=[torch.empty(l.workspace_size(m,n,k),device='cuda',dtype=torch.uint8) for l in libs]
                act=[torch.empty(l.activation_size(m,k),device='cuda',dtype=torch.uint8) for l in libs]
                for path in ('quantized','full','offline_b'):
                    funcs=[]
                    for l,o,w,ac in zip(libs,outputs,work,act):
                        if path=='quantized':fn=lambda l=l,o=o,a8=a8,b8=b8,sa8=sa8,sb8=sb8,row=row,col=col,ws=ws:l.quantized(a8,b8,sa8,sb8,o,row,col,ws)
                        elif path=='full':fn=lambda l=l,o=o,w=w,x=x,y=y,mode=mode,ws=ws:l.forward(x,y,o,w,ref.MODES[mode],ws)
                        else:fn=lambda l=l,o=o,ac=ac,x=x,b8=b8,sb8=sb8,row=row,col=col,ws=ws:l.offline(x,b8,sb8,o,ac,row,col,ws)
                        fn();torch.testing.assert_close(o.cpu(),expected,atol=ref.ATOL,rtol=ref.RTOL)
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):fn()
                        with torch.cuda.stream(stream):o.fill_(float('nan'));graph.replay()
                        stream.synchronize();torch.testing.assert_close(o.cpu(),expected,atol=ref.ATOL,rtol=ref.RTOL);funcs.append(fn)
                    meta=dict(**prefix,path=path,mode=mode,ws=ws)
                    cases.append((meta,*funcs));report['checks'].append(dict(**meta,graph=True,passed=True))
for rid in range(a.rounds):
    for meta,ours,old in cases[rid:]+cases[:rid]:
        warmup(ours,old)
        report['performance'].append(dict(**meta,round=rid,**graph_pair(ours,old,calls=16)))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print('round',rid+1,'completed',len(report['performance']),flush=True)
key=lambda r:(r['format'],*r['shape'],r['path'],r['mode'],r['ws'])
report['gates']={path:gates([r for r in report['performance'] if r['path']==path],key) for path in ('quantize_b','quantized','full','offline_b')}
report['passed']=all(r['passed'] for r in report['checks'])
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['gates']),flush=True)
