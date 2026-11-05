"""Shared CPU fixtures, exact schedule audit, runtime and original CuTeDSL checks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import torch
from .environment import ROOT
from .cross_validate import sha,tensor_sha,UPSTREAM
from .benchmark import graph_pair
from .cute_dsl import VARIANTS


def fixtures(args):
    torch.set_num_threads(4);rows=[]
    for seed in (17,93):
        g=torch.Generator().manual_seed(seed)
        for m,n,k in ((128,128,16),(128,256,64),(256,128,192),(128,128,4096),(17,33,1),(65,137,17),(129,65,65)):
            a=torch.randn(m,k,generator=g).mul_(.5);b=torch.randn(k,n,generator=g).mul_(.5);ref=a.double()@b.double();rows.append(dict(id=f'random/{seed}/{m}/{n}/{k}',a=a,b=b,reference=ref,hashes=[tensor_sha(a),tensor_sha(b)],cuda=m%128==0 and n%128==0 and k%8==0 and k>=16))
    for label in ('rounding','cancellation','zero'):
        a=torch.ones(128,128);b=torch.full((128,128),.0078125)
        if label=='rounding':b[0]=.00825
        if label=='cancellation':b[1::2]=-b[::2]
        if label=='zero':a.zero_()
        rows.append(dict(id=label,a=a,b=b,reference=a.double()@b.double(),hashes=[tensor_sha(a),tensor_sha(b)],cuda=True))
    args.fixtures.parent.mkdir(parents=True,exist_ok=True);torch.save(rows,args.fixtures);print(len(rows),sha(args.fixtures))


def cuda_module(name,args):
    from cuda.bindings import driver as cuda
    path=args.original/'kernels/cutlass/cute_dsl/sgemm'/('sgemm.py' if name=='naive' else name+'.py');source=path.read_text();original_sha=sha(path)
    # Only adapt launch stream and disable print diagnostics, preserving every
    # computation/layout. Freeze generated source separately from original.
    source='from cuda.bindings import driver as cuda\n'+source.replace('VERBOSE = True','VERBOSE = False')
    if name=='naive':source=source.replace('def host(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):','def host(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, stream: cuda.CUstream):')
    else:source=source.replace('copy_bits: cutlass.Constexpr = 128):','copy_bits: cutlass.Constexpr = 128, stream: cuda.CUstream = cuda.CUstream(0)):')
    assert source.count('.launch(')==1;source=source.replace('.launch(','.launch(stream=stream,')
    folder=args.output.parent/'cuda_sources';folder.mkdir(parents=True,exist_ok=True);staged=folder/(name+'.py');staged.write_text(source)
    spec=importlib.util.spec_from_file_location('original_cute_dsl_'+name,staged);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,dict(original=str(path),original_sha256=original_sha,staged=str(staged),staged_sha256=sha(staged),upstream_commit=UPSTREAM,adaptation='current-stream argument and diagnostics only')


def run(args):
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    rows=torch.load(args.fixtures,weights_only=True);report=dict(backend=args.backend,fixture_sha256=sha(args.fixtures),harness_sha256=sha(__file__),builds=[],checks=[],boundaries=[],pipeline=[],performance=[]);outputs={}
    if args.backend=='hip':
        from .cute_dsl import _native,source_paths
        from .environment import verify
        lib=_native(args.device);report['environment']=verify();report['builds'].append(dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths()},binary_sha256=sha(lib.__file__),binary=lib.__file__))
    else:
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from cuda.bindings import driver as cuda
        import importlib.metadata
        report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),dsl=importlib.metadata.version('nvidia-cutlass-dsl'),image=os.environ.get('LEETCUDA_CUDA_IMAGE'))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for name,kind in VARIANTS.items():
        if args.variant and name!=args.variant:continue
        if args.backend=='cuda':module,meta=cuda_module(name,args);report['builds'].append(meta)
        for case in rows:
            if args.backend=='cuda' and not case['cuda']:continue
            assert case['hashes']==[tensor_sha(case[x]) for x in ('a','b')]
            with torch.cuda.stream(stream):
                a,b=[case[x].cuda() for x in ('a','b')];bt=b if kind==0 else b.T.contiguous();m,k=a.shape;n=b.shape[1];back=torch.full((m*n+8,),-91.,device='cuda');o=back[4:-4].view(m,n)
                ref=case['reference'].half().float() if kind==0 else case['reference'];tol=.01 if kind==0 else .0002
                if args.backend=='hip':fn=lambda:lib.forward(a,bt,o,kind,None)
                else:
                    ta,tb,tc=[from_dlpack(t,assumed_align=16) for t in (a,bt,o)];cs=cuda.CUstream(stream.cuda_stream);host=module.host if kind==0 else module.sgemm_host
                    compiled=cute.compile(host,ta,tb,tc,stream=cs);fn=lambda:compiled(ta,tb,tc,cuda.CUstream(torch.cuda.current_stream().cuda_stream))
                key=f'{name}/{case["id"]}';o.fill_(float('nan'));fn();stream.synchronize();got=o.cpu()
                passed=bool(torch.allclose(got.double(),ref.double(),atol=tol,rtol=tol) and (back[:4]==-91).all() and (back[-4:]==-91).all());report['checks'].append(dict(id=key,passed=passed,max_abs=float((got.double()-ref.double()).abs().max()),atol=tol,rtol=tol));outputs[key]=got
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):fn()
                o.fill_(float('nan'));graph.replay();stream.synchronize();report['checks'].append(dict(id=key+'/graph',passed=passed and torch.equal(got,o.cpu()) and bool((back[:4]==-91).all() and (back[-4:]==-91).all())));outputs[key+'/graph']=o.cpu()
                if case['id']=='rounding':report['checks'].append(dict(id=key+'/precision_contract',passed=bool((o==(1.0 if kind==0 else case['reference'][0,0].float())).all())))
                if args.backend=='cuda' and case['id']=='random/17/128/128/16':
                    alternate=torch.cuda.Stream();alternate.wait_stream(stream)
                    with torch.cuda.stream(alternate):
                        fn();alternate.synchronize();other_graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(other_graph,stream=alternate):fn()
                        o.fill_(float('nan'));other_graph.replay();alternate.synchronize()
                        report['checks'].append(dict(id=key+'/different_capture_stream',passed=torch.equal(got,o.cpu())))
                    stream.wait_stream(alternate)
                if args.performance and case['id'].startswith('random/17') and case['cuda']:report['performance'].append(dict(variant=name,m=m,n=n,k=k,baseline='torch_mm_tf32_disabled',**graph_pair(fn,lambda:torch.mm(a,b,out=o),calls=16)))
            print(key,passed,flush=True)
    if args.backend=='hip':
        from .cute_dsl import matmul
        for kind in (1,2):
            for k in (1,8,16,24,65,4096):
                a=torch.ones(137,k,device='cuda');b=torch.ones(65,k,device='cuda');o=torch.empty(137,65,device='cuda');metrics=torch.zeros(2,8,device='cuda',dtype=torch.int32)
                lib.forward(a,b,o,kind,metrics);torch.cuda.synchronize();s=metrics.cpu();tiles=(k+7)//8
                ok=bool((s[:,0]==tiles).all() and (s[:,1]==tiles).all() and (s[:,2]==((1<<min(3,tiles))-1 if kind==2 else 1)).all() and (s[:,5]==tiles).all() and (o==k).all())
                if kind==2:ok &= bool((s[:,4]==max(0,tiles-2)).all())
                report['pipeline'].append(dict(kind=kind,k=k,metrics=s.tolist(),passed=ok))
        a=torch.ones(16,16,device='cuda');b=a.clone();o=a.clone()
        def reject(label,fn):
            try:fn();ok=False
            except (ValueError,RuntimeError,TypeError):ok=True
            report['boundaries'].append(dict(id=label,passed=ok))
        for kind in range(3):
            for i in range(3):
                for label,x in (('cpu',a.cpu()),('half',a.half()),('transpose',a.T),('grad',a.clone().requires_grad_())):
                    inputs=[a,b,o];inputs[i]=x;reject(f'{kind}/{i}/{label}',lambda inputs=inputs,kind=kind:lib.forward(*inputs,kind,None))
            reject(f'{kind}/alias',lambda kind=kind:lib.forward(a,b,a,kind,None))
        reject('invalid_variant',lambda:matmul(a,b,variant='none'))
        if torch.cuda.device_count()>1:
            previous=torch.cuda.current_device();target=1 if previous==0 else 0;aa,bb,oo=[x.to(target) for x in (a,b,o)];lib.forward(aa,bb,oo,2,None);torch.cuda.synchronize(target);report['boundaries'].append(dict(id='device_guard',passed=torch.cuda.current_device()==previous and bool((oo==16).all())))
    report['counts']={k:len(report[k]) for k in ('checks','boundaries','pipeline','performance')};report['passed']=all(x['passed'] for f in ('checks','boundaries','pipeline') for x in report[f]);args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(outputs,args.output.with_suffix('.pt'));report['outputs_sha256']=sha(args.output.with_suffix('.pt'));args.output.write_text(json.dumps(report,indent=2)+'\n');print(report['counts'],report['passed']);assert report['passed']


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('fixtures','run'));p.add_argument('--backend',default='hip',choices=('hip','cuda'));p.add_argument('--variant',choices=tuple(VARIANTS));p.add_argument('--device',type=int,default=0);p.add_argument('--fixtures',type=Path,default=Path('artifacts/rocm/continuation_all/cute_dsl/fixtures.pt'));p.add_argument('--output',type=Path,default=Path('artifacts/rocm/continuation_all/cute_dsl/hip_primary.json'));p.add_argument('--original',type=Path,default=ROOT/'original');p.add_argument('--performance',action='store_true');args=p.parse_args();globals()[args.command](args)

if __name__=='__main__':main()
