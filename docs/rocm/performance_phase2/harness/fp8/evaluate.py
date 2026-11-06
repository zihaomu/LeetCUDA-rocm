import argparse,json,sys,shutil,hashlib
from pathlib import Path
from functools import lru_cache
import torch
from leetcuda_rocm import fp8,fp8_check
from leetcuda_rocm.loader import _load,_prepare
from leetcuda_rocm.environment import ROOT,verify,cache_key
from leetcuda_rocm.cross_validate import sha
p=argparse.ArgumentParser();p.add_argument('--header',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--format',default='e4m3');p.add_argument('--guard-device',type=int,default=1);args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(__file__,args.output.parent/(args.output.stem+'_harness.py'))
@lru_cache(None)
def native(device,fp8_format='e4m3'):
 with torch.cuda.device(device):
  identity=verify();header=args.header.read_text().removeprefix('#pragma once\n');binding=(ROOT/'leetcuda_rocm/native/fp8_bindings.cpp').read_text()
  if fp8_format=='e5m2':header='#define LEETCUDA_FP8_E5M2 1\n'+header
  return _load('fp8_opt_'+fp8_format,(header+'\n'+_prepare(binding),),cache_key(identity),identity['gpu_architecture'],False)
if args.header.resolve()!=ROOT/'kernels/rocm/fp8_gemm_hip.cuh':
 fp8._native=native;fp8_check._native=native
# Exhaust all positive finite BF16 amax values admitted by the original normal
# FP32 scale/inverse contract, for both formats.
values=torch.arange(0x7f80,dtype=torch.int16).view(torch.bfloat16).float();rows=[]
for fmt,mx in [('e4m3',448.),('e5m2',57344.)]:
 scales=values/mx;inv=mx/values;valid=(scales>=torch.finfo(torch.float32).tiny)&(inv>=torch.finfo(torch.float32).tiny)&torch.isfinite(inv)
 recovered=(scales[valid]*mx).bfloat16().float();ok=torch.equal(recovered,values[valid]);assert ok
 rows.append(dict(format=fmt,values=int(valid.sum()),passed=ok))
(args.output.parent/'scale_reconstruction.json').write_text(json.dumps(dict(checks=rows,passed=True),indent=2)+'\n')
sys.argv=[sys.argv[0],'--fp8-format',args.format,'--guard-device',str(args.guard_device),'--output',str(args.output)]
result=fp8_check.main()
r=json.loads(args.output.read_text());r['candidate_header']=str(args.header);r['candidate_header_sha256']=sha(args.header);r['optimization_harness_sha256']=sha(__file__);args.output.write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(result)
