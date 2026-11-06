"""Run standalone suites while preserving historical ISA artifact paths."""
import json,sys
from pathlib import Path
from leetcuda_rocm import build_bench
from leetcuda_rocm.cross_validate import sha
output=Path(sys.argv[sys.argv.index('--output')+1])
original=build_bench.hip_isa
build_bench.hip_isa=lambda binary,unused:original(binary,output.parent/(output.stem+'_isa'))
build_bench.main()
report=json.loads(output.read_text());report['optimization_wrapper_sha256']=sha(__file__);output.write_text(json.dumps(report,indent=2)+'\n')
