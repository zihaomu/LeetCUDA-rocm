from pathlib import Path
import ctypes,hashlib,json
from leetcuda_rocm.loader import load
lib=load('relu');path=Path(lib.__file__);h=hashlib.sha256(path.read_bytes()).hexdigest()
fn=lib.relu_f32
get=ctypes.pythonapi.PyCFunction_GetFunction;get.argtypes=[ctypes.py_object];get.restype=ctypes.c_void_p
address=get(fn);mapping=[]
for line in Path('/proc/self/maps').read_text().splitlines():
 lo,hi=[int(x,16) for x in line.split()[0].split('-')]
 if lo<=address<hi:mapping.append(line.split()[-1])
r=dict(binary=str(path),binary_sha256=h,path_is_content_addressed=path.parent.name==h and path.parent.parent.name=='frozen',dispatcher_mapped_files=mapping,dispatcher_in_frozen_binary=str(path) in mapping)
r['passed']=r['path_is_content_addressed'] and r['dispatcher_in_frozen_binary']
r['source_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('leetcuda_rocm/loader.py','leetcuda_rocm/build_extension.py')}
Path('artifacts/rocm/continuation_all/immutable_loader_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));assert r['passed']
