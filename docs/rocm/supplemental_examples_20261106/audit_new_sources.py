from pathlib import Path
import hashlib,json,subprocess,shutil
root=Path.cwd();base=root/'artifacts/rocm/continuation_all';index={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=list((root/'leetcuda_rocm').rglob('*.py'))+list((root/'leetcuda_rocm/native').glob('*'))+list((root/'kernels/rocm').glob('*'))+list(base.rglob('*.py'))+list(base.rglob('*.cuh'))+list(base.rglob('*.cpp'))+list(base.rglob('*.cu'))
for p in paths:
 if p.is_file():index.setdefault(sha(p),str(p.relative_to(root)))
checks=[]
for group in ('interview_fa','supplemental','cute_dsl','swizzle'):
 for name in ('hip_primary','hip_secondary','hip_formatted'):
  p=base/group/(name+'.json');d=json.loads(p.read_text());todo=[('harness',d['harness_sha256'],None)]
  builds=d.get('builds',[d['build']] if 'build' in d else [])
  for b in builds:
   todo += [(path,value,path) for path,value in b['source_sha256'].items()]
   if 'adapter_sha256' in b:todo += [('adapter',b['adapter_sha256'],None)]
   if 'binary' in b:todo += [('binary',b['binary_sha256'],b['binary'])]
  for key,value,source in todo:
   q=Path(source) if source else None
   if q and q.is_file() and sha(q)==value:
    actual=str(q);passed=True
   else:actual=index.get(value);passed=actual is not None
   checks.append(dict(report=str(p.relative_to(root)),key=key,sha256=value,located=actual,passed=passed))
  for key,file in (('fixture_sha256',base/group/'fixtures.pt'),('outputs_sha256',p.with_suffix('.pt'))):
   checks.append(dict(report=str(p.relative_to(root)),key=key,sha256=d[key],located=str(file.relative_to(root)),passed=sha(file)==d[key]))
out=dict(checks=checks,passed=all(x['passed'] for x in checks));(base/'source_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(len(checks),'checks',out['passed']);print(json.dumps([x for x in checks if not x['passed']],indent=2))
