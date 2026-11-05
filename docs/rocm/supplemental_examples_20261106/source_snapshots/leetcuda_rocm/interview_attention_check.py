"""Independent interview FA qualification with frozen CPU inputs and CUDA oracle."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from .cross_validate import sha, tensor_sha
from .environment import ROOT
from .interview_attention import VARIANTS
from .benchmark import graph_pair

ATOL = RTOL = .003
LSE_ATOL = LSE_RTOL = 2e-5


def reference(q, k, v, persist=False, causal=False, scale=None):
    scale = scale if scale is not None else q.shape[-1] ** -.5
    qq = q.double()
    factor = scale
    if persist:
        qs = torch.tensor(scale, dtype=torch.float32) * torch.tensor(math.log2(math.e), dtype=torch.float32)
        qq = (q.float() * qs).half().double()
        factor = math.log(2)
    ratio = q.shape[1] // k.shape[1]
    kk, vv = [x.repeat_interleave(ratio, 1).double() for x in (k, v)]
    # Chunk queries to keep the independent CPU oracle's memory linear in NQ.
    outs, lses = [], []
    for start in range(0, q.shape[-2], 128):
        scores = (qq[..., start:start+128, :] @ kk.transpose(-1, -2)) * factor
        if causal:
            mask = torch.arange(k.shape[-2])[None, :] > torch.arange(start, min(start+128, q.shape[-2]))[:, None] + k.shape[-2] - q.shape[-2]
            scores.masked_fill_(mask, -torch.inf)
        outs.append(torch.nan_to_num(scores.softmax(-1)) @ vv)
        lses.append(scores.logsumexp(-1))
    return torch.cat(outs, -2), torch.cat(lses, -1)


def dimensions(kind):
    return (32, 64, 128) if kind == 0 else (64, 96, 128) if kind == 7 else (64, 128)


def stages(kind, backend):
    if kind == 7 and backend == 'cuda':
        return [(2, 2)]
    return list(itertools.product(range(1, 3 if kind == 7 else 5), range(1, 3 if kind in (1, 4, 7) else 2)))


def fixtures(args):
    torch.set_num_threads(8)
    rows = []
    def add(label, d, nq, nk, seed, pattern=None, hq=2, hk=2, causal=False, scale=None, native=False):
        gen = torch.Generator().manual_seed(seed)
        q,k,v = [torch.randn(shape, generator=gen).mul(.5).half() for shape in ((1,hq,nq,d),(1,hk,nk,d),(1,hk,nk,d))]
        if pattern == 'uniform': q.zero_(); k.zero_()
        if pattern == 'zero_v': v.zero_()
        if pattern == 'large': q.mul_(8); k.mul_(8)
        if pattern == 'cancel': v[...,1::2,:] = -v[...,:2*(nk//2):2,:]
        if pattern in ('rescale','split_max'):
            q.zero_(); q[...,0] = 1; k.zero_()
            for t in range((nk+31)//32):
                logit = (0,4,16,18,32,33)[t%6] if pattern == 'rescale' else (12 if t%2 else -12)
                k[...,t*32:(t+1)*32,0] = logit * d**.5 * math.log(2)
        eligible = [name for name, kind in VARIANTS.items() if d in dimensions(kind) and (kind == 7 or (hq==hk and nq==nk and not causal and scale is None))]
        expected = {}
        for mode in ('standard','prescaled'):
            if (mode == 'standard' and any(VARIANTS[x]!=7 for x in eligible)) or (mode == 'prescaled' and 'persist_d' in eligible):
                o,l = reference(q,k,v,mode=='prescaled',causal,scale)
                expected[mode] = {'o':o,'lse':l}
        rows.append(dict(id=label,q=q,k=k,v=v,input_sha256=[tensor_sha(x) for x in (q,k,v)],expected=expected,variants=eligible,causal=causal,scale=scale,native_only=native))
    for d in (32,64,96,128):
        for seed in (17,93):
            for n in (128,512): add(f'random/{d}/{seed}/{n}',d,n,n,seed)
        for pattern in ('uniform','zero_v','large','cancel','rescale','split_max'):
            add(f'{pattern}/{d}',d,384,384,29,pattern)
        for n in (17,65,137): add(f'tail/{d}/{n}',d,n,n,71,native=True)
        add(f'long/{d}',d,2048,2048,97,hq=1,hk=1)
    for d in (64,96,128):
        for nq,nk in ((65,137),(137,65),(128,4096)):
            for causal in (False,True): add(f'gqa/{d}/{nq}/{nk}/{causal}',d,nq,nk,93,hq=4,hk=2,causal=causal)
        add(f'custom_scale/{d}',d,128,256,17,hq=4,hk=2,scale=.125)
    args.fixtures.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(cases=rows,generator_sha256=sha(__file__)),args.fixtures)
    print(json.dumps(dict(cases=len(rows),sha256=sha(args.fixtures))),flush=True)


def accurate(got, ref, lse=False):
    got = got.cpu().double(); ref = ref.double(); finite=torch.isfinite(ref)
    return dict(passed=bool(torch.allclose(got,ref,atol=LSE_ATOL if lse else ATOL,rtol=LSE_RTOL if lse else RTOL)),max_abs=(float((got[finite]-ref[finite]).abs().max()) if torch.isfinite(got[finite]).all() else None) if finite.any() else 0.)


def save(path, report):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def pipeline(lib):
    records=[]
    for name,kind in VARIANTS.items():
        d=64; persist=kind==7; nq=65 if persist else 768; nk=768
        q=torch.zeros(1,1,nq,d,device='cuda',dtype=torch.float16)
        k=torch.zeros(1,1,nk,d,device='cuda',dtype=torch.float16);v=torch.full_like(k,.125)
        out=torch.empty_like(q);lse=torch.empty(q.shape[:-1],device='cuda')
        br=64 if kind in (0,3) else 32;bc=32 if kind in (2,5) else 64
        grid=1 if persist else (nq+br-1)//br
        for sk,sv in stages(kind,'hip'):
            for delay in ((0,) if kind in (0,3) else (1,2)):
                audit=torch.zeros(grid,16,device='cuda',dtype=torch.int32)
                lib.forward(q,k,v,out,lse,kind,sk,sv,7 if kind==0 else 0,False,d**-.5,1 if persist else 0,audit,delay)
                torch.cuda.synchronize();a=audit.cpu();queries=(nq+br-1)//br;tiles=(nk+bc-1)//bc;groups=2 if kind in (2,5) else 1
                totals=a.sum(0).tolist();ok=totals[0]==queries and totals[1]==tiles*queries and totals[2]==tiles*queries and totals[7]==0 and totals[12]==queries and totals[13]==queries
                for c in range(groups):
                    count=(tiles+groups-1-c)//groups*queries
                    ok &= totals[3+2*c]==count and totals[4+2*c]==count
                    if kind not in (0,3):
                        ok &= int(a[:,8+2*c].max())<=sk and int(a[:,9+2*c].max())<=sv
                        ok &= int(a[:,(8 if delay==1 else 9)+2*c].max())==(sk if delay==1 else sv)
                ok &= bool((out==.125).all())
                records.append(dict(variant=name,sk=sk,sv=sv,delay=delay,totals=totals,max_lead=a[:,8:12].max(0).values.tolist(),passed=bool(ok)))
    # Persistent causal tiles must be pruned per query iteration, including empty rows.
    for nq,nk in ((137,65),(65,137)):
        q=torch.zeros(1,1,nq,64,device='cuda',dtype=torch.float16);k=torch.zeros(1,1,nk,64,device='cuda',dtype=torch.float16);v=torch.full_like(k,.125)
        out=torch.empty_like(q);lse=torch.empty(q.shape[:-1],device='cuda');a=torch.zeros(1,16,device='cuda',dtype=torch.int32)
        lib.forward(q,k,v,out,lse,7,2,2,0,True,.125,1,a,1);torch.cuda.synchronize()
        expected_tiles=sum(max(0,min((nk+63)//64,(base+31+nk-nq)//64+1)) for base in range(0,nq,32));totals=a.cpu()[0].tolist()
        ref,rl=reference(q.cpu(),k.cpu(),v.cpu(),True,True)
        records.append(dict(variant='persist_d',causal=True,nq=nq,nk=nk,totals=totals,passed=totals[1]==expected_tiles and totals[2]==expected_tiles and totals[7]==0 and accurate(out,ref)['passed'] and accurate(lse,rl,True)['passed']))
    return records


def boundaries(lib):
    from .interview_attention import interview_attention
    rows=[]
    q=torch.randn(1,2,64,64,device='cuda',dtype=torch.float16);k=q.clone();v=q.clone();o=torch.empty_like(q);l=torch.empty(q.shape[:-1],device='cuda')
    base=[q,k,v,o,l,0,2,1,7,False,.125,0,None,0]
    def reject(label,fn):
        try: fn();ok=False
        except (ValueError,RuntimeError,TypeError):ok=True
        rows.append(dict(id=label,passed=ok))
    for i in range(5):
        for what,x in [('cpu',base[i].cpu()),('dtype',base[i].double()),('grad',base[i].clone().requires_grad_()),('strided',base[i].transpose(-1,-2))]:
            args=base.copy();args[i]=x;reject(f'tensor/{i}/{what}',lambda args=args:lib.forward(*args))
    for i,values in {5:[-1,6,8],6:[0,5],7:[0,2],8:[-1,8],9:[True],10:[0,-1,float('inf'),float('nan')],11:[-1,1],13:[-1,3]}.items():
        for value in values:
            args=base.copy();args[i]=value;reject(f'argument/{i}/{value}',lambda args=args:lib.forward(*args))
    for i in (0,1,2):
        args=base.copy();args[3]=base[i];reject(f'alias/output/{i}',lambda args=args:lib.forward(*args))
    args=base.copy();args[12]=torch.zeros(3,dtype=torch.int32,device='cuda');reject('metrics/size',lambda:lib.forward(*args))
    for kw in ({'variant':'missing'},{'stages_k':True},{'stages_v':1.5},{'pad_mask':False},{'causal':1},{'persistent_ctas':1.5}):
        reject('public/'+str(kw),lambda kw=kw:interview_attention(q,k,v,**kw))
    reject('public/empty_D',lambda:interview_attention(q[...,:0],k[...,:0],v[...,:0]))
    for args in ((q[0,0],q[0,0]),(q[0,0].float(),o[0,0].float()),(q[0,0,:,:32],o[0,0,:,:32])):
        reject('copy/reject/'+str(len(rows)),lambda args=args:lib.copy(*args))
    if torch.cuda.device_count()>1:
        previous=torch.cuda.current_device();target=1 if previous==0 else 0
        other=[x.to(target) for x in (q,k,v,o,l)]
        lib.forward(*other,*base[5:]);torch.cuda.synchronize(target)
        rows.append(dict(id='device_guard/restore',passed=torch.cuda.current_device()==previous and bool(torch.isfinite(other[3]).all())))
        args=base.copy();args[1]=other[1];reject('device_guard/mismatch',lambda:lib.forward(*args))
    return rows


def run(args):
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_device(args.device)
    data=torch.load(args.fixtures,weights_only=True);outputs={}
    report=dict(backend=args.backend,fixture_sha256=sha(args.fixtures),harness_sha256=sha(__file__),checks=[],unavailable=[],builds=[],pipeline=[],boundaries=[],layout=[],performance=[],memory=[])
    if args.backend=='hip':
        from .interview_attention import _native,source_paths
        from .environment import verify
        lib=_native(args.device);report['environment']=verify();report['builds'].append(dict(binary=lib.__file__,binary_sha256=sha(lib.__file__),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths()}))
    else:
        report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),image=os.environ.get('LEETCUDA_CUDA_IMAGE'))
        from .interview_attention_cuda import build
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    variants=list(VARIANTS.items())+ [('copy',6)]
    for name,kind in variants:
        if args.kind is not None and kind!=args.kind:continue
        for d in dimensions(kind):
            if args.d is not None and d!=args.d:continue
            if args.backend=='cuda':
                lib,meta=build(kind,d,slotfix=args.cuda_fix);report['builds'].append(meta)
            if kind==6:
                for n in ((64,128,512,17,137) if args.backend=='hip' else (64,128,512)):
                    q=torch.arange(n*d,dtype=torch.float32).remainder(977).half().view(n,d).cuda();o=torch.empty_like(q)
                    maps=lib.prepare(q,q,q) if args.backend=='cuda' else None
                    fn=(lambda:lib.copy(q,o)) if args.backend=='hip' else (lambda:lib.forward(q,q,q,o,maps,1,1,0,False,1.))
                    with torch.cuda.stream(stream):
                        fn();stream.synchronize();g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g,stream=stream):fn()
                        o.fill_(float('nan'));g.replay();stream.synchronize()
                        key=f'copy/{d}/{n}';report['checks'].append(dict(id=key,passed=torch.equal(q,o)));outputs[key]=o.cpu()
                continue
            for case in data['cases']:
                if name not in case['variants'] or case['q'].shape[-1]!=d or (args.backend=='cuda' and case['native_only']):continue
                assert case['input_sha256']==[tensor_sha(case[x]) for x in ('q','k','v')]
                with torch.cuda.stream(stream):
                    inputs=[case[x].cuda() for x in ('q','k','v')];q,k,v=inputs
                    backing=torch.full((q.numel()+16,),-123,device='cuda',dtype=torch.float16);out=backing[8:-8].view(q.shape)
                    lback=torch.full((q.numel()//d+16,),-123,device='cuda',dtype=torch.float32);lse=lback[8:-8].view(q.shape[:-1])
                    maps=lib.prepare(*inputs) if args.backend=='cuda' else None
                    scale=case['scale'] or d**-.5;expected=case['expected']['prescaled' if kind==7 else 'standard']
                    for sk,sv in stages(kind,args.backend):
                        key=f'{name}/{case["id"]}/s{sk}v{sv}'
                        if args.backend=='cuda' and kind==0 and not args.cuda_fix and sk-1>k.shape[-2]//64:
                            report['unavailable'].append(dict(id=key,reason='original warmup reads past final K tile; tested by explicit slot/warmup fix'));continue
                        out.fill_(float('nan'));lse.fill_(float('nan'))
                        fn=(lambda:lib.forward(*inputs,out,lse,kind,sk,sv,7 if kind==0 else 0,case['causal'],scale,0,None,0)) if args.backend=='hip' else (lambda:lib.forward(*inputs,out,maps,sk,sv,7 if kind==0 else 0,case['causal'],scale))
                        try:fn();stream.synchronize()
                        except RuntimeError as e:
                            if args.backend=='cuda' and 'shared-memory requirement' in str(e):
                                report['unavailable'].append(dict(id=key,reason=str(e)));continue
                            raise
                        got=out.cpu();got_lse=lse.cpu() if args.backend=='hip' else None
                        check=accurate(got,expected['o']);check.update(id=key,canary=bool((backing[:8]==-123).all() and (backing[-8:]==-123).all()))
                        if got_lse is not None:check['lse']=accurate(got_lse,expected['lse'],True);check['passed'] &= check['lse']['passed'] and bool((lback[:8]==-123).all() and (lback[-8:]==-123).all())
                        check['passed'] &= check['canary'];report['checks'].append(check);outputs[key]=got
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):fn()
                        out.fill_(float('nan'));lse.fill_(float('nan'));graph.replay();stream.synchronize()
                        ok=accurate(out,expected['o'])['passed'] and torch.equal(got,out.cpu()) and bool((backing[:8]==-123).all() and (backing[-8:]==-123).all())
                        if got_lse is not None:ok &= torch.equal(got_lse,lse.cpu()) and bool((lback[:8]==-123).all() and (lback[-8:]==-123).all())
                        report['checks'].append(dict(id=key+'/graph',passed=bool(ok)));outputs[key+'/graph']=out.cpu()
                print(key,'done',flush=True)
            if args.performance:
                for n in (512,2048):
                    torch.manual_seed(71);q,k,v=[torch.randn(1,2,n,d,device='cuda',dtype=torch.float16).mul_(.5) for _ in range(3)];out=torch.empty_like(q);lse=torch.empty(q.shape[:-1],device='cuda');sk=1 if kind in (2,5) and d==128 else 2;sv=2 if kind==7 else 1
                    maps=lib.prepare(q,k,v) if args.backend=='cuda' else None
                    fn=(lambda:lib.forward(q,k,v,out,lse,kind,sk,sv,7 if kind==0 else 0,False,d**-.5,0,None,0)) if args.backend=='hip' else (lambda:lib.forward(q,k,v,out,maps,sk,sv,7 if kind==0 else 0,False,d**-.5))
                    report['performance'].append(dict(variant=name,d=d,n=n,sk=sk,sv=sv,baseline='torch_sdpa_fp16',**graph_pair(fn,lambda:F.scaled_dot_product_attention(q,k,v),calls=16)))
            save(args.output,report)
    if args.backend=='hip' and args.kind is None:
        report['pipeline']=pipeline(lib);report['boundaries']=boundaries(lib)
        for line in args.layout.read_text().splitlines():
            row=json.loads(line);actual=lib.layout(7 if row['d']==96 else 3,row['d'],row['rows'],False)
            report['layout'].append(dict(d=row['d'],rows=row['rows'],passed=actual==row['offsets']))
        # All eight Q/K/V padding combinations and deliberately misaligned inputs.
        for d in (32,64,128):
            q,k,v=[torch.randn(1,1,137,d).half() for _ in range(3)];ref,rl=reference(q,k,v)
            xs=[]
            for x in (q,k,v):
                b=torch.empty(x.numel()+1,device='cuda',dtype=torch.float16);z=b[1:].view(x.shape);z.copy_(x);xs.append(z)
            out=torch.empty_like(xs[0]);lse=torch.empty(q.shape[:-1],device='cuda')
            for pad in range(8):
                lib.forward(*xs,out,lse,0,4,1,pad,False,d**-.5,0,None,0);torch.cuda.synchronize()
                report['checks'].append(dict(id=f'misaligned_pad/{d}/{pad}',passed=accurate(out,ref)['passed'] and accurate(lse,rl,True)['passed']))
        for name,kind in VARIANTS.items():
            q=torch.randn(1,1,4096,64,device='cuda',dtype=torch.float16);k=q.clone();v=q.clone();o=torch.empty_like(q);lse=torch.empty(q.shape[:-1],device='cuda');torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            lib.forward(q,k,v,o,lse,kind,2,1,7 if kind==0 else 0,False,.125,0,None,0);torch.cuda.synchronize();extra=torch.cuda.max_memory_allocated()-before
            report['memory'].append(dict(variant=name,n=4096,extra_allocated=extra,passed=extra==0))
    report['passed']=all(x['passed'] for field in ('checks','pipeline','boundaries','layout','memory') for x in report[field])
    report['counts']={key:len(report[key]) for key in ('checks','unavailable','pipeline','boundaries','layout','performance','memory')}
    torch.save(outputs,args.output.with_suffix('.pt'));report['outputs_sha256']=sha(args.output.with_suffix('.pt'));save(args.output,report)
    print(json.dumps(dict(passed=report['passed'],counts=report['counts'])),flush=True)
    if args.backend=='hip':assert report['passed']


def compare(args):
    hip=torch.load(args.hip,weights_only=True);cuda=torch.load(args.cuda,weights_only=True);keys=sorted(set(hip)&set(cuda));rows=[]
    for key in keys:rows.append(dict(id=key,**accurate(hip[key],cuda[key]),exact=torch.equal(hip[key],cuda[key])))
    result=dict(hip_sha256=sha(args.hip),cuda_sha256=sha(args.cuda),checks=rows,common=len(keys),hip_only=len(hip.keys()-cuda.keys()),cuda_only=len(cuda.keys()-hip.keys()),passed=bool(keys) and all(x['passed'] for x in rows));save(args.output,result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}));assert result['passed']


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('fixtures','run','compare'));p.add_argument('--backend',choices=('hip','cuda'),default='hip');p.add_argument('--fixtures',type=Path,default=Path('artifacts/rocm/continuation_all/interview_fa/fixtures.pt'));p.add_argument('--output',type=Path,default=Path('artifacts/rocm/continuation_all/interview_fa/hip.json'));p.add_argument('--layout',type=Path,default=Path('artifacts/rocm/continuation_all/interview_fa/cute_layout_reference.jsonl'));p.add_argument('--device',type=int,default=0);p.add_argument('--kind',type=int);p.add_argument('--d',type=int);p.add_argument('--performance',action='store_true');p.add_argument('--cuda-fix',action='store_true');p.add_argument('--hip',type=Path);p.add_argument('--cuda',type=Path);args=p.parse_args();globals()[args.command](args)

if __name__=='__main__':main()
