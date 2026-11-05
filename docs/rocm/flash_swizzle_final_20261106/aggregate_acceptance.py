from pathlib import Path
import json,hashlib,copy,os
root=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((root/name).read_text())
full,retest=read('frozen_final.json'),read('frozen_qk_retest.json')
assert full['passed'] and retest['passed']
for key in ('fixture_sha256','harness_sha256'):
 assert full[key]==retest[key]
assert full['builds']['hip']['source_sha256']==retest['builds']['hip']['source_sha256']
assert full['builds']['hip']['binary_sha256']==retest['builds']['hip']['binary_sha256']
assert sorted(x['hsaco_sha256'] for x in full['isa']['objects'])==sorted(x['hsaco_sha256'] for x in retest['isa']['objects'])
# New final measurements must inspect the exact immutable file loaded.
assert full['builds']['hip']['binary_sha256']==full['isa']['binary_sha256']
assert retest['builds']['hip']['binary_sha256']==retest['isa']['binary_sha256']
assert full['environment']['selected_device_pci_bus_id']==retest['environment']['selected_device_pci_bus_id']
name='flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qk'
assert len(retest['performance'])==54 and all(x['function']==name for x in retest['performance'])
assert {x['id'] for x in retest['performance']}=={x['id'] for x in full['performance'] if x['function']==name}
# Replace this entry's complete three rounds, never individual samples or cases.
out=copy.deepcopy(full)
out['performance']=[x for x in full['performance'] if x['function']!=name]+retest['performance']
out['performance_datasets']={
 'q_and_qkv':dict(report='frozen_final.json',sha256=sha(root/'frozen_final.json')),
 'qk':dict(report='frozen_qk_retest.json',sha256=sha(root/'frozen_qk_retest.json')),
 'policy':'Retain all three full rounds for Q/QKV. Replace all three QK rounds after the first run exceeded the fixed 5% stability gate at N1024 D64 stage1. Preserve frozen_acceptance.json and every original sample. No threshold or per-case sample selection.',
}
out['retest_telemetry']=retest['telemetry']
(root/'frozen_accepted.json').write_text(json.dumps(out,indent=2)+'\n')
if not (root/'frozen_accepted.pt').exists():os.link(root/'frozen_final.pt',root/'frozen_accepted.pt')
assert sha(root/'frozen_accepted.pt')==out['outputs_sha256']
print('aggregated',len(out['performance']),'samplesets')
