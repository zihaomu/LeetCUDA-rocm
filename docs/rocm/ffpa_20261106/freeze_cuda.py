from pathlib import Path
import json,hashlib,shutil
root=Path.cwd();a=root/'artifacts/rocm/continuation_all/ffpa';out=a/'binaries';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for name in ('cuda_original_final.json','cuda_tail_fix.json','cuda_performance.json'):
 r=json.loads((a/name).read_text())
 for d,build in r['builds'].items():
  matches=[p for p in (root/'.cache/cuda_comparison').glob(f'ffpa_{d}_*/leetcuda_ffpa_cuda_{d}_*.so') if sha(p)==build['binary_sha256']]
  assert matches,(name,d)
  p=matches[0];dest=out/(build['binary_sha256']+'.so');shutil.copy2(p,dest)
  checks.append(dict(report=name,d=d,passed=sha(dest)==build['binary_sha256'],binary=str(dest.relative_to(root)),binary_sha256=sha(dest)))
(a/'cuda_binary_audit.json').write_text(json.dumps(dict(passed=all(c['passed'] for c in checks),checks=checks),indent=2)+'\n')
print('FROZEN',len(checks))
