from pathlib import Path
import json,hashlib,shutil
root=Path('/home/zihaomu/zihaomu_workspace/test_lap/leetcuda_rocm10')
base=root/'artifacts/rocm/continuation_all';out=base/'cuda_frozen';out.mkdir(exist_ok=True)
rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(s):return root/s.removeprefix('/work/') if s.startswith('/work/') else (Path(s) if Path(s).is_absolute() else root/s)
def archive(name,value,origin):
 p=resolve(name)
 if not p.is_file():rows.append(dict(path=name,report=origin,passed=False,reason='missing'));return
 actual=sha(p);dst=out/actual/p.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(p,dst)
 rows.append(dict(path=name,report=origin,expected=value,actual=actual,frozen=str(dst.relative_to(base)),passed=actual==value))
for group,reports in {'interview_fa':['cuda_original','cuda_slotfix','cuda_emptyfix'],'supplemental':['cuda_original'],'cute_dsl':['cuda_original'],'swizzle':['cuda_original','cuda_fp32']}.items():
 for report in reports:
  path=base/group/(report+'.json')
  if not path.exists():continue
  d=json.loads(path.read_text());builds=d.get('builds',[d['build']] if 'build' in d else [])
  if isinstance(builds,dict):builds=list(builds.values())
  for b in builds:
   if 'binary' in b:archive(b['binary'],b['binary_sha256'],str(path.relative_to(base)))
   for key in ('original','staged'):
    if key in b and key+'_sha256' in b:archive(b[key],b[key+'_sha256'],str(path.relative_to(base)))
   for name,value in b.get('source_sha256',{}).items():
    if name.startswith('/'):archive(name,value,str(path.relative_to(base)))
   if 'binary' in b:
    for p in resolve(b['binary']).parent.glob('*'):
     if p.suffix in ('.cu','.cpp','.cuh','.ninja'):archive(str(p),sha(p),str(path.relative_to(base)))
result=dict(checks=rows,passed=bool(rows) and all(r['passed'] for r in rows));(base/'cuda_archive_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('checks',len(rows),'passed',result['passed'])
