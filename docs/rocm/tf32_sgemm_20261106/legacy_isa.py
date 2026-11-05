import json,hashlib,subprocess
from pathlib import Path
from leetcuda_rocm.isa_objects import hip_isa_objects
from leetcuda_rocm.loader import load,source_paths
root=Path.cwd();folder=root/'artifacts/rocm/continuation_all/b6/legacy_isa';folder.mkdir(parents=True,exist_ok=True)
old_s=json.loads((root/'docs/rocm/async_sgemm_20261103.json').read_text())
b5=json.loads((root/'artifacts/rocm/continuation_all/b5/qualification209.json').read_text())
for path in ('kernels/sgemm/sgemm.cu','kernels/sgemm/sgemm_async.cu'):assert old_s['source_sha256'][path]==b5['source_files_sha256'][path]
old_h=json.loads((root/'artifacts/rocm/continuation_all/b5/hip.json').read_text())
new_s=json.loads((root/'artifacts/rocm/continuation_all/b6/hip.json').read_text())
new_h=hip_isa_objects(load('hgemm').__file__,folder/'hgemm/new')
results={}
for module,old,new in [('sgemm',old_s['isa'],new_s['isa']),('hgemm',old_h['isa'],new_h)]:
 records={}
 for version,report in [('old',old),('new',new)]:
  hashes=[]
  for i,obj in enumerate(report['objects']):
   path=root/obj['hsaco'];out=folder/f'{module}_{version}_{i}.text';discard=folder/'discard'
   subprocess.run(['/opt/rocm/llvm/bin/llvm-objcopy','--dump-section','.text='+str(out),str(path),str(discard)],check=True);discard.unlink();hashes.append(dict(file=str(path.relative_to(root)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size))
  records[version]=dict(binary_sha256=report['binary_sha256'],objects=hashes)
 records['old_instruction_objects_preserved']={x['sha256'] for x in records['old']['objects']}<={x['sha256'] for x in records['new']['objects']};results[module]=records
result=dict(passed=all(x['old_instruction_objects_preserved'] for x in results.values()),scope='SGEMM frozen async qualification has identical two GPU source hashes to B5; HGEMM frozen B5 binary compared to current source rebuild. Whole qualification cache paths are mutable, so only persisted ISA snapshots are used.',modules=results)
(root/'artifacts/rocm/continuation_all/b6/legacy_instruction_bytes.json').write_text(json.dumps(result,indent=2)+'\n');print(result['passed'])
