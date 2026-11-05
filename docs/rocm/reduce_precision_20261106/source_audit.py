import json,hashlib
from pathlib import Path
root=Path.cwd(); folder=root/'docs/rocm/reduce_precision_20261106'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
q=json.loads((folder/'qualification218.json').read_text());h=json.loads((folder/'hip.json').read_text());h1=json.loads((folder/'hip_device1.json').read_text());checks=[]
for k in ['source_files_sha256','adapter_sha256']:
 for name,value in q[k].items():checks.append(dict(path=name,recorded=value,current=sha(root/name),passed=value==sha(root/name)))
for report,label in [(h,'primary'),(h1,'secondary')]:
 checks.append(dict(path=label+'/harness',passed=report['harness_sha256']==sha(root/'leetcuda_rocm/reduce_precision_check.py')))

 for path,value in report['builds']['hip']['source_sha256'].items():checks.append(dict(path=label+'/'+path,passed=value==sha(root/path)))
assert all(x['passed'] for x in checks)
r=dict(passed=True,checks=checks,report_sha256={p.name:sha(p) for p in sorted(folder.glob('*.json')) if p.name!='source_audit.json'})
(folder/'source_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(len(checks),'source checks passed')
