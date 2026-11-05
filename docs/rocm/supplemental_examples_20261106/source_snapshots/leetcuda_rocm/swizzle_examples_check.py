"""Five original swizzle definitions, explicit precision and layout qualification."""
import argparse,itertools,json,os
from pathlib import Path
import torch
from .environment import ROOT
from .cross_validate import sha,tensor_sha
from .swizzle_examples import NAMES
from .benchmark import graph_pair


def fixtures(args):
    torch.set_num_threads(4);rows=[]
    for seed in (17,93):
        g=torch.Generator().manual_seed(seed)
        for m,n,k in ((128,128,16),(128,256,64),(256,128,192),(128,128,4096),(17,33,1),(65,137,17),(129,65,65)):
            a=torch.randn(m,k,generator=g).mul_(.25).half();b=torch.randn(k,n,generator=g).mul_(.25).half();rows.append(dict(id=f'random/{seed}/{m}/{n}/{k}',a=a,b=b,reference=a.double()@b.double(),hashes=[tensor_sha(a),tensor_sha(b)],cuda=m%128==0 and n%128==0 and k%16==0))
    a=torch.ones(128,1040,dtype=torch.float16);b=torch.full((1040,128),2**-15,dtype=torch.float16);b[:16]=.0625
    rows.append(dict(id='precision_state',a=a,b=b,reference=a.double()@b.double(),hashes=[tensor_sha(a),tensor_sha(b)],cuda=True))
    args.fixtures.parent.mkdir(parents=True,exist_ok=True);torch.save(rows,args.fixtures);print(len(rows),sha(args.fixtures))


def run(args):
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_device(args.device)
    if args.backend=='hip':
        from .swizzle_examples import native,source_paths
        from .environment import verify
        lib=native(args.device);identity=verify();meta=dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths()},binary=lib.__file__,binary_sha256=sha(lib.__file__),precision='explicit_fp32')
    else:
        from .swizzle_examples_cuda import build
        lib,meta=build(args.precision);identity=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),image=os.environ.get('LEETCUDA_CUDA_IMAGE'))
    report=dict(backend=args.backend,environment=identity,build=meta,fixture_sha256=sha(args.fixtures),harness_sha256=sha(__file__),checks=[],precision_controls=[],boundaries=[],layout=[],performance=[]);outputs={};rows=torch.load(args.fixtures,weights_only=True)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for kind,name in enumerate(NAMES):
        for case in rows:
            if args.backend=='cuda' and not case['cuda']:continue
            assert case['hashes']==[tensor_sha(case[x]) for x in ('a','b')]
            with torch.cuda.stream(stream):
                a,b=[case[x].cuda() for x in ('a','b')];m,k=a.shape;n=b.shape[1];storage=torch.full((m*n+16,),-91,device='cuda',dtype=torch.float16);o=storage[8:-8].view(m,n);ref=case['reference'].half()
                for ap,bp in (itertools.product((0,8),repeat=2) if kind in (1,3) else ((0,0),)):
                    key=f'{kind}/{case["id"]}/a{ap}b{bp}';fn=lambda:lib.forward(a,b,o,kind,ap,bp);o.fill_(float('nan'));fn();stream.synchronize();got=o.cpu();ok=bool(torch.allclose(got,ref,atol=.01,rtol=0) and (storage[:8]==-91).all() and (storage[-8:]==-91).all());report['checks'].append(dict(id=key,passed=ok,max_abs_to_rounded_fp64=float((got-ref).abs().max()),max_abs_to_fp64=float((got.double()-case['reference']).abs().max())));outputs[key]=got
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):fn()
                    o.fill_(float('nan'));graph.replay();stream.synchronize();report['checks'].append(dict(id=key+'/graph',passed=ok and torch.equal(got,o.cpu()) and bool((storage[:8]==-91).all() and (storage[-8:]==-91).all())));outputs[key+'/graph']=o.cpu()
                    if case['id']=='precision_state':
                        value=1. if args.backend=='cuda' and args.precision=='original' else 1.03125;report['precision_controls'].append(dict(id=key,expected=value,passed=bool((got==value).all())))
                    if args.performance and case['id'].startswith('random/17') and case['cuda'] and ap==0 and bp==0:report['performance'].append(dict(kind=kind,m=m,n=n,k=k,precision=meta.get('precision'),**graph_pair(fn,lambda:torch.mm(a,b,out=o),calls=16)))
            print(kind,case['id'],'done',flush=True)
    if args.backend=='hip':
        for nr in (16,128):
            want=[r*16+(((c//8)^(r//4))%2)*8+c%8 for r in range(nr) for c in range(16)];actual=lib.layout(nr);report['layout'].append(dict(rows=nr,passed=actual==want and sorted(actual)==list(range(nr*16))))
        a=torch.ones(16,16,device='cuda',dtype=torch.float16);b=a.clone();o=a.clone()
        def reject(label,fn):
            try:fn();ok=False
            except (RuntimeError,ValueError,TypeError):ok=True
            report['boundaries'].append(dict(id=label,passed=ok))
        for kind in range(5):
            for i in range(3):
                for label,x in (('cpu',a.cpu()),('float',a.float()),('strided',a.T),('grad',a.clone().requires_grad_())):
                    inputs=[a,b,o];inputs[i]=x;reject(f'{kind}/{i}/{label}',lambda inputs=inputs,kind=kind:lib.forward(*inputs,kind,0,0))
            reject(f'{kind}/alias',lambda kind=kind:lib.forward(a,b,a,kind,0,0));reject(f'{kind}/pad',lambda kind=kind:lib.forward(a,b,o,kind,4,0))
        if torch.cuda.device_count()>1:
            previous=torch.cuda.current_device();target=1 if previous==0 else 0;aa,bb,oo=[x.to(target) for x in (a,b,o)];lib.forward(aa,bb,oo,3,0,8);torch.cuda.synchronize(target);report['boundaries'].append(dict(id='device_guard',passed=torch.cuda.current_device()==previous and bool((oo==16).all())))
    report['counts']={f:len(report[f]) for f in ('checks','precision_controls','boundaries','layout','performance')};report['passed']=all(x['passed'] for f in ('checks','precision_controls','boundaries','layout') for x in report[f]);args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(outputs,args.output.with_suffix('.pt'));report['outputs_sha256']=sha(args.output.with_suffix('.pt'));args.output.write_text(json.dumps(report,indent=2)+'\n');print(report['counts'],report['passed']);
    if args.backend=='hip' or args.precision=='fp32':assert report['passed']


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('fixtures','run'));p.add_argument('--backend',default='hip',choices=('hip','cuda'));p.add_argument('--precision',default='original',choices=('original','fp32'));p.add_argument('--device',type=int,default=0);p.add_argument('--fixtures',type=Path,default=Path('artifacts/rocm/continuation_all/swizzle/fixtures.pt'));p.add_argument('--output',type=Path,default=Path('artifacts/rocm/continuation_all/swizzle/hip_primary.json'));p.add_argument('--performance',action='store_true');args=p.parse_args();globals()[args.command](args)
if __name__=='__main__':main()
