import json,subprocess,os
from pathlib import Path
p=Path('artifacts/rocm/continuation_all/b4')
results=[]
for c in json.loads((p/'contracts.json').read_text()):
 name=c['function'];folder=p/'cuda_final'/name;folder.mkdir(parents=True,exist_ok=True)
 with (folder/'run.log').open('w') as f:
  r=subprocess.run(['python',str(p/'cuda_final.py')],env={**os.environ,'B4_FUNCTION':name},stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(function=name,exit_code=r.returncode));(p/'cuda_final_status.json').write_text(json.dumps(results,indent=2)+'\n');print(name,r.returncode,flush=True)
