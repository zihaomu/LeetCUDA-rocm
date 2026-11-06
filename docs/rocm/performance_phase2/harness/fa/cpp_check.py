import json,subprocess
from pathlib import Path
from leetcuda_rocm.environment import ROOT,verify
from leetcuda_rocm.cross_validate import sha
folder=ROOT/'artifacts/rocm/performance_phase2/fa/packed_lds'
binary=folder/'cpp_bench'
sources=[ROOT/p for p in ('kernels/rocm/interview_attention_bench.cpp','kernels/rocm/interview_flash_attention_hip.cuh','kernels/rocm/interview_common_hip.cuh','kernels/rocm/bench_runtime.hpp','kernels/interview/flash_attn.cuh')]
env=verify()
command=['/opt/rocm/bin/hipcc','-O3','-std=c++20','--offload-arch=gfx1201','-DLEETCUDA_ROCM=1','-I'+str(ROOT/'kernels'),'-I'+str(ROOT/'kernels/rocm'),str(sources[0]),'-o',str(binary)]
subprocess.run(command,check=True)
fixture=ROOT/'artifacts/rocm/continuation_all/full_acceptance_immutable/cpp/fixtures.bin'
r=subprocess.run([str(binary),str(fixture),str(folder/'cpp_outputs.bin')],capture_output=True,text=True)
(folder/'cpp_checks.log').write_text(r.stdout+r.stderr)
checks=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{"id"')]
linked=subprocess.check_output(['ldd',str(binary)],text=True)
report=dict(environment=env,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},harness_sha256=sha(__file__),build_command=command,binary_sha256=sha(binary),fixture_sha256=sha(fixture),linked_libraries=linked,checks=checks,passed=r.returncode==0 and len(checks)==646 and all(c['passed'] for c in checks) and 'libtorch' not in linked)
(folder/'cpp.json').write_text(json.dumps(report,indent=2)+'\n');print(len(checks),report['passed']);assert report['passed']
