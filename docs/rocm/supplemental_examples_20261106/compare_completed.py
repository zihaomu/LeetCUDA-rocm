from pathlib import Path
import json,torch,hashlib
root=Path('artifacts/rocm/continuation_all')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(folder,cuda_name,atol,rtol,output):
 p=root/folder;hp=p/'hip_primary.pt';cp=p/(cuda_name+'.pt');h=torch.load(hp,weights_only=True);c=torch.load(cp,weights_only=True);rows=[]
 for key in sorted(h.keys()&c.keys()):
  a,b=h[key],c[key];ok=torch.allclose(a.double(),b.double(),atol=atol,rtol=rtol)
  finite=torch.isfinite(a)&torch.isfinite(b);rows.append(dict(id=key,passed=ok,exact=torch.equal(a,b),max_abs=float((a.double()[finite]-b.double()[finite]).abs().max()) if finite.any() else None))
 report=dict(hip_sha256=sha(hp),cuda_sha256=sha(cp),checks=rows,common=len(rows),hip_only=len(h.keys()-c.keys()),cuda_only=len(c.keys()-h.keys()),atol=atol,rtol=rtol,passed=bool(rows) and all(x['passed'] for x in rows))
 (p/output).write_text(json.dumps(report,indent=2)+'\n');print(folder,cuda_name,'checks',len(rows),'fail',sum(not x['passed'] for x in rows),flush=True)
 return report
assert compare('cute_dsl','cuda_original',0,0,'cross.json')['passed']
compare('interview_fa','cuda_original',.003,.003,'cross_original.json')
p=root/'interview_fa'
if (p/'cuda_emptyfix.pt').exists():
 out=torch.load(p/'cuda_original.pt',weights_only=True);sources=['cuda_original.pt','cuda_slotfix.pt','cuda_emptyfix.pt']
 for name in sources[1:]:out.update(torch.load(p/name,weights_only=True))
 torch.save(out,p/'cuda_matched.pt')
 (p/'cuda_matched_manifest.json').write_text(json.dumps(dict(sources={name:sha(p/name) for name in sources},output_sha256=sha(p/'cuda_matched.pt'),policy='Original outputs, replacing mma by explicit slot/warmup fix and persist_d by explicit all-masked-row fix; original failures retained separately.'),indent=2)+'\n')
 assert compare('interview_fa','cuda_matched',.003,.003,'cross_matched.json')['passed']
