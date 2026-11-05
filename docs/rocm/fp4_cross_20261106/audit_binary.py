from pathlib import Path
import json,hashlib,re,subprocess,shutil,sys
root=Path.cwd();a=root/'artifacts/rocm/continuation_all/fp4_cross'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sys.argv[1]=='cuda':
 checks=[]
 for name in ('cuda_original.json','cuda_performance.json'):
  r=json.loads((a/name).read_text());b=r['build'];p=Path(b['binary']);assert sha(p)==b['binary_sha256']
  dest=a/'binaries'/(b['binary_sha256']+'.so');dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
  asm=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(dest)],text=True)
  (a/'cuda.sass').write_text(asm)
  lines=[line.strip() for line in asm.splitlines() if 'MMA' in line and not 'Function' in line]
  checks.append(dict(report=name,passed=sha(dest)==b['binary_sha256'] and bool(lines),binary=str(dest.relative_to(root)),binary_sha256=sha(dest),sass_sha256=sha(a/'cuda.sass'),matrix_instruction_examples=list(dict.fromkeys(re.sub(r'^/\*.*?\*/\s*','',line) for line in lines))[:8]))
 (a/'cuda_binary_audit.json').write_text(json.dumps(dict(passed=all(c['passed'] for c in checks),checks=checks),indent=2)+'\n');print(json.dumps(checks));assert all(c['passed'] for c in checks)
else:
 import yaml
 from leetcuda_rocm.isa_objects import hip_isa_objects
 r=json.loads((a/'hip.json').read_text());report=hip_isa_objects(r['build']['binary'],a/'isa')
 assert report['binary_sha256']==r['build']['binary_sha256']
 kernels=[]
 for obj in report['objects']:
  notes=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readobj','--notes',str(root/obj['hsaco'])],text=True)
  for e in yaml.safe_load(notes.split('AMDGPU Metadata: ',1)[1].split('...',1)[0])['amdhsa.kernels']:
   kernels.append(dict(symbol=e['.name'],vgpr=e['.vgpr_count'],spill=e['.vgpr_spill_count'],scratch=e['.private_segment_fixed_size'],lds=e['.group_segment_fixed_size']))
 report['kernels']=kernels;report['passed']=bool(kernels) and all(k['scratch']==0 and k['spill']==0 for k in kernels) and any('f16' in i for i in report['instructions'])
 (a/'hip_binary_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('ISA',report['passed'],len(kernels));assert report['passed']
