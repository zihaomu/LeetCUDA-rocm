from pathlib import Path
import json,re,subprocess,yaml
from leetcuda_rocm.interview_attention import _native,source_paths
from leetcuda_rocm.isa_objects import hip_isa_objects
from leetcuda_rocm.cross_validate import sha
root=Path.cwd();a=root/'artifacts/rocm/continuation_all/interview_fa';lib=_native(0);isa=hip_isa_objects(lib.__file__,(a/'formatted_isa').resolve());metadata={}
for obj in isa['objects']:
 notes=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readobj','--notes',str(root/obj['hsaco'])],text=True)
 for e in yaml.safe_load(notes.split('AMDGPU Metadata: ',1)[1].split('...',1)[0])['amdhsa.kernels']:metadata[e['.name']]=e
parts=re.split(r'^([0-9a-f]+) <([^>]+)>:\n',(root/isa['assembly']).read_text(),flags=re.M);ks=[]
for i in range(1,len(parts),3):
 symbol,body=parts[i+1:i+3];match=re.search(r'kernelILi(\d+)ELi(\d+)ELb([01])EE',symbol)
 if not match:continue
 kind,d,audit=map(int,match.groups());e=metadata[symbol]
 ks.append(dict(kind=kind,d=d,audit=bool(audit),symbol=symbol,wmma=body.count('v_wmma_f32_16x16x16_f16'),load128=body.count('global_load_b128'),vgpr=e['.vgpr_count'],spill=e['.vgpr_spill_count'],scratch=e['.private_segment_fixed_size'],lds=e['.group_segment_fixed_size']))
out=dict(**isa,kernels=ks,source_sha256={str(p.relative_to(root)):sha(p) for p in source_paths()},functional_count_passed=len(ks)==32 and all(k['wmma'] and k['load128'] and k['lds']<=65536 for k in ks));(a/'formatted_resources.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(ks));print(out['functional_count_passed']);assert out['functional_count_passed']
