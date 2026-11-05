"""Shared-input validation for CuTe vector, profiling kernels and LDS layouts."""
import argparse
import json
import math
import os
from pathlib import Path
import torch
from .environment import ROOT
from .cross_validate import sha,tensor_sha
from .supplemental import build,NAMES
from .benchmark import graph_pair


def fixtures(args):
    g=torch.Generator().manual_seed(20261106)
    data={name:torch.randn(131072,generator=g).mul_(.75) for name in ('x','y')}
    data['x'][:8]=torch.tensor([0.,-1.,1.,-1e-5,1e-5,-100.,100.,0.])
    data['hashes']={name:tensor_sha(data[name]) for name in ('x','y')}
    args.fixtures.parent.mkdir(parents=True,exist_ok=True);torch.save(data,args.fixtures);print(sha(args.fixtures))


def run(args):
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_device(args.device)
    data=torch.load(args.fixtures,weights_only=True)
    assert data['hashes']=={name:tensor_sha(data[name]) for name in ('x','y')}
    r=dict(backend=args.backend,fixture_sha256=sha(args.fixtures),harness_sha256=sha(__file__),builds=[],checks=[],boundaries=[],performance=[]);outputs={}
    if args.backend=='hip':
        from .environment import verify
        r['environment']=verify()
    else:r['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),image=os.environ.get('LEETCUDA_CUDA_IMAGE'))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    def check(key,fn,shape,dtype,ref):
        with torch.cuda.stream(stream):
            buf=torch.full((math.prod(shape)+16,),-91,device='cuda',dtype=dtype);o=buf[8:-8].view(shape)
            if dtype.is_floating_point:o.fill_(float('nan'))
            fn(o);stream.synchronize();got=o.cpu();exact=torch.equal(got,ref)
            ok=exact and bool((buf[:8]==-91).all() and (buf[-8:]==-91).all())
            r['checks'].append(dict(id=key,passed=ok,max_abs=float((got.double()-ref.double()).abs().max()) if got.numel() else 0.));outputs[key]=got
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):fn(o)
            o.fill_(float('nan') if dtype.is_floating_point else -171);graph.replay();stream.synchronize();got=o.cpu()
            r['checks'].append(dict(id=key+'/graph',passed=torch.equal(got,ref) and bool((buf[:8]==-91).all() and (buf[-8:]==-91).all())));outputs[key+'/graph']=got
    def reject(key,fn):
        try:fn();ok=False
        except (RuntimeError,ValueError,TypeError):ok=True
        r['boundaries'].append(dict(id=key,passed=ok))
    for group in NAMES:
        if args.group and group!=args.group:continue
        lib,meta=build(group,args.backend,args.device);r['builds'].append(meta)
        if group=='profile':
            for kind,name in enumerate(NAMES[group]):
                mode=kind%6;dtype=torch.float32 if mode<2 else torch.float16;width=(1,4,1,2,8,8)[mode]
                for n in (width,width*17,width*256,width*1009):
                    a,b=[data[x][:n].to(dtype) for x in ('x','y')];ref=(a+b) if kind<6 else a.clamp(min=0)
                    ga,gb=a.cuda(),b.cuda();check(f'{name}/{n}',lambda o:lib.forward(ga,gb,o,kind),(n,),dtype,ref)
                out=torch.empty_like(ga);reject(name+'/alias',lambda:lib.forward(ga,gb,ga,kind));reject(name+'/dtype',lambda:lib.forward(ga,gb,out.double(),kind))
                if width>1:reject(name+'/tail',lambda:lib.forward(ga[:-1],gb[:-1],out[:-1],kind))
                reject(name+'/cpu',lambda:lib.forward(a,b,torch.empty_like(a),kind))
                if args.performance:
                    ga,gb=[data[x].repeat(8).to(dtype).cuda() for x in ('x','y')];o=torch.empty_like(ga)
                    baseline=(lambda:torch.add(ga,gb,out=o)) if kind<6 else (lambda:torch.clamp(ga,min=0,out=o))
                    r['performance'].append(dict(group=group,name=name,n=ga.numel(),**graph_pair(lambda:lib.forward(ga,gb,o,kind),baseline)))
        elif group=='vector':
            for elements in (8,16):
                for n in (16,272,8192,16384):
                    a,b=[data[x][:n].half() for x in ('x','y')];ga,gb=a.cuda(),b.cuda()
                    for constants in ((1.,1.,0.),(.3,-.7,.13),(1.001,1.0001,-.0003)):
                        aa,bb,cc=[torch.tensor(x).half().double() for x in constants]
                        ref=(a.double()*aa+(b.double()*bb+cc).half().double()).half()
                        check(f'vector/{elements}/{n}/{constants}',lambda o:lib.vector(ga,gb,o,*constants,elements),(n,),torch.float16,ref)
                if args.backend=='hip':
                    for n in (1,17,137):
                        a,b=[data[x][:n].half() for x in ('x','y')];xs=[]
                        for x in (a,b):
                            storage=torch.empty(n+1,device='cuda',dtype=torch.float16);z=storage[1:];z.copy_(x);xs.append(z)
                        check(f'vector/tail/{elements}/{n}',lambda o:lib.vector(*xs,o,1.,1.,0.,elements),(n,),torch.float16,a+b)
            ga,gb=[data[x][:256].half().cuda() for x in ('x','y')];o=torch.empty_like(ga)
            for e in (0,1,4,32):reject(f'vector/e{e}',lambda e=e:lib.vector(ga,gb,o,1,1,0,e))
            reject('vector/alias',lambda:lib.vector(ga,gb,ga,1,1,0,8));reject('vector/dtype',lambda:lib.vector(ga.float(),gb.float(),o.float(),1,1,0,8))
            if args.performance:
                ga,gb=[data[x].repeat(8).half().cuda() for x in ('x','y')];o=torch.empty_like(ga)
                for e in (8,16):r['performance'].append(dict(group=group,elements=e,n=ga.numel(),baseline='torch_add_a1b1c0',**graph_pair(lambda:lib.vector(ga,gb,o,1,1,0,e),lambda:torch.add(ga,gb,out=o))))
        else:
            for mode,name in enumerate(NAMES[group]):
                shapes=[(32,32),(64,96),(256,512),(1024,1024)]+([(1,1),(17,33),(65,137)] if args.backend=='hip' else [])
                for m,n in shapes:
                    a=torch.arange(m*n,dtype=torch.int32).reshape(m,n).mul_(13).sub_(719);ga=a.cuda();check(f'{name}/{m}/{n}',lambda o:lib.transpose(ga,o,mode),(n,m),torch.int32,a.T.contiguous())
                ga=torch.arange(64*64,dtype=torch.int32,device='cuda').view(64,64);reject(name+'/alias',lambda:lib.transpose(ga,ga,mode));reject(name+'/dtype',lambda:lib.transpose(ga.float(),ga.float(),mode))
                if args.performance:
                    ga=torch.arange(2048**2,dtype=torch.int32,device='cuda').view(2048,2048);o=torch.empty_like(ga)
                    r['performance'].append(dict(group=group,name=name,m=2048,n=2048,**graph_pair(lambda:lib.transpose(ga,o,mode),lambda:o.copy_(ga.T))))
        print(group,'done',flush=True)
    r['passed']=all(x['passed'] for field in ('checks','boundaries') for x in r[field]);r['counts']={k:len(r[k]) for k in ('checks','boundaries','performance')}
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(outputs,args.output.with_suffix('.pt'));r['outputs_sha256']=sha(args.output.with_suffix('.pt'));args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=r['passed'],counts=r['counts'])));assert r['passed']


def compare(args):
    a=torch.load(args.hip,weights_only=True);b=torch.load(args.cuda,weights_only=True);keys=sorted(a.keys()&b.keys());rows=[dict(id=k,passed=torch.equal(a[k],b[k])) for k in keys];r=dict(checks=rows,common=len(rows),passed=bool(rows) and all(x['passed'] for x in rows),hip_only=len(a.keys()-b.keys()),cuda_only=len(b.keys()-a.keys()),hip_sha256=sha(args.hip),cuda_sha256=sha(args.cuda));args.output.write_text(json.dumps(r,indent=2)+'\n');print(r['common'],r['passed']);assert r['passed']


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('fixtures','run','compare'));p.add_argument('--backend',default='hip',choices=('hip','cuda'));p.add_argument('--device',type=int,default=0);p.add_argument('--group',choices=tuple(NAMES));p.add_argument('--fixtures',type=Path,default=Path('artifacts/rocm/continuation_all/supplemental/fixtures.pt'));p.add_argument('--output',type=Path,default=Path('artifacts/rocm/continuation_all/supplemental/hip.json'));p.add_argument('--performance',action='store_true');p.add_argument('--hip',type=Path);p.add_argument('--cuda',type=Path);args=p.parse_args();globals()[args.command](args)

if __name__=='__main__':main()
