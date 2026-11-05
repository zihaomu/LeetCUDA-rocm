# U3：11个同步SIMT HGEMM迁移验收

日期：2026-11-06。本组完成后原接口 **229/229（100%）**，HGEMM **36/36**。
这是原公开计算接口的功能覆盖率；FFPA、Attention扩展、E5M2完整链路、独立构建及性能计划继续。

## 实现与契约

11个入口保留原naive/sliced、x4/x8、pack/BCF、双缓冲及8×8/16×8线程微分块。
FP16输入/输出，**明确升级FP32 FMA与运行状态**，只在写回时舍入FP16。
沿原K顺序计算，原FP16实现仍作为私有control；不声称逐位复现其累加舍入。

- naive：每线程一个输出，32×32线程块，原生有界读取。
- sliced：32×32×32 LDS分块，每线程一个输出。
- x4与pack：128×128×8，8×8线程微分块，分别保留half2/四元素输出组织。
- BCF：A在LDS转置，x4保留四象限微分块布局；packed LDS读取和输出宽度独立保留。
- x8：八元素LDS读取；K8、K16、K32双缓冲有不同LDS跨度，16×8微分块保留128线程分工。
- 双缓冲最后读取槽取 `(tile_count-1)&1`，支持奇数tile；两槽均实际使用。
- 原合法对齐整tile走各自kernel。尾部或未对齐A/B/C使用有界原生scalar kernel，明确标为扩展边界路径。
- 同设备连续FP16二维张量、正确形状、无输出别名、forward-only、32位索引及grid上限；当前stream和device guard。

原`half`临时数组被编译器放入额外LDS/scratch。最终使用原生半精度向量，
显式线程块上限，并控制K循环展开，使11个kernel都达到零spill/scratch，未放宽资源门限。
pack的源码组织差异保留；编译器可能合并相邻标量/短向量访问，不把名称中的x8误解为所有global load均128-bit。

## 验收

| 项目 | 结果 |
| --- | --- |
| 两张AMD卡 | 各 **891/891**：121共享输入、649边界/状态/运行时、121原FP16 control |
| 精度 | 原atol=rtol=0.01不变，独立FP64参考舍入FP16 |
| 原FP16 control | 121/121精确匹配CPU逐K舍入模型；其中22组未过FP64门限 |
| 5090原版 | 91/121过参考，30失败保留 |
| 5090仅槽位修复 | 99/121过参考，修复8组奇数tile问题；22组长K精度失败保留 |
| 专项跨平台 | 双方参考均通过的原版91组、修复版99组分别全部通过比较 |
| 全库AMD | **25模块、229接口、1262/1262** |
| 全接口共同输入 | **462/462**显式FP32 CUDA参考树与AMD比较通过；原FP16树460/462过参考 |
| ISA | 11个独立SIMT kernel均有FP32 FMA；LDS0–32KiB，VGPR22–172，全部零spill/scratch |

共享输入覆盖两种seed、非方阵、K32/96/256/4096、零值/抵消/互逆尺度。
AMD另覆盖每变体1/3/5个完整K tile、K8193尾部、空M/N/K、A/B/C未对齐、输入不变、输出canary、
精确FP32状态区分、非法dtype/rank/shape/stride/alias/grad、设备混用与guard，
整tile和尾部各8条stream、每条连续4次调用。

主卡物理GPU4/PCI83/HIP6，副卡物理GPU5/PCIa3/HIP7；锁定ROCm10.1/HIP7.16与Torch/Triton。
物理GPU3未参与计算。5090固定镜像；original快照不修改，槽位修复使用ported工作树。全库另建立明确标记的precision_reference工作树：
14个同步/异步SIMT入口升级FP32状态，其余CUDA实现保留。原ported全库460/462过参考，
一个异步16×8和一个同步x4 pack BCF在K256超差；报告和相同fixture完整保留。
相同输入在precision_reference与AMD均通过462组，验证的是升级后的精度契约；不改写原FP16失败。
生成脚本及两份CUDA参考源码/manifest随报告保存，CUDA cp.async与原tile/BCF分工仍保留。
CPU fixture、源码、输出和冻结ISA二进制hash逐项保留。fixture基于第一次229接口资格生成，
对应[生成时资格](hgemm_simt_20261106/fixture_qualification229.json)；后续只增加对照来源标记，冻结输入未重采样。

## 性能

每graph8调用，20ms×5组交替采样，同卡Torch.mm基线；33个配置均过精度和填NaN后的graph replay。
SIMT教学实现均慢于矩阵库基线，保留差距，不用不同GPU耗时之比宣称加速。

| 接口 | M=N=K | AMD ms | Torch/AMD |
| --- | ---: | ---: | ---: |
| `hgemm_naive_f16` | 256 | 0.014930 | 0.3259× |
| `hgemm_sliced_k_f16` | 256 | 0.015030 | 0.3234× |
| `hgemm_t_8x8_sliced_k_f16x4` | 256 | 0.023702 | 0.2049× |
| `hgemm_t_8x8_sliced_k_f16x4_pack` | 256 | 0.023634 | 0.2053× |
| `hgemm_t_8x8_sliced_k_f16x4_bcf` | 256 | 0.026639 | 0.1820× |
| `hgemm_t_8x8_sliced_k_f16x4_pack_bcf` | 256 | 0.026243 | 0.1837× |
| `hgemm_t_8x8_sliced_k_f16x8_pack_bcf` | 256 | 0.024320 | 0.1975× |
| `hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf` | 256 | 0.019700 | 0.2432× |
| `hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf` | 256 | 0.018312 | 0.2606× |
| `hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf` | 256 | 0.018914 | 0.2510× |
| `hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf` | 256 | 0.020499 | 0.2317× |
| `hgemm_naive_f16` | 1024 | 0.579860 | 0.0398× |
| `hgemm_sliced_k_f16` | 1024 | 0.537340 | 0.0468× |
| `hgemm_t_8x8_sliced_k_f16x4` | 1024 | 0.132287 | 0.1779× |
| `hgemm_t_8x8_sliced_k_f16x4_pack` | 1024 | 0.133564 | 0.1767× |
| `hgemm_t_8x8_sliced_k_f16x4_bcf` | 1024 | 0.136345 | 0.1731× |
| `hgemm_t_8x8_sliced_k_f16x4_pack_bcf` | 1024 | 0.136419 | 0.1745× |
| `hgemm_t_8x8_sliced_k_f16x8_pack_bcf` | 1024 | 0.128869 | 0.1825× |
| `hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf` | 1024 | 0.120833 | 0.1925× |
| `hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf` | 1024 | 0.114807 | 0.1980× |
| `hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf` | 1024 | 0.114961 | 0.2049× |
| `hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf` | 1024 | 0.113121 | 0.2098× |
| `hgemm_naive_f16` | 2048 | 4.730412 | 0.0323× |
| `hgemm_sliced_k_f16` | 2048 | 4.375032 | 0.0355× |
| `hgemm_t_8x8_sliced_k_f16x4` | 2048 | 0.936001 | 0.1630× |
| `hgemm_t_8x8_sliced_k_f16x4_pack` | 2048 | 0.940188 | 0.1588× |
| `hgemm_t_8x8_sliced_k_f16x4_bcf` | 2048 | 0.952056 | 0.1601× |
| `hgemm_t_8x8_sliced_k_f16x4_pack_bcf` | 2048 | 0.947011 | 0.1595× |
| `hgemm_t_8x8_sliced_k_f16x8_pack_bcf` | 2048 | 0.911139 | 0.1669× |
| `hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf` | 2048 | 0.892077 | 0.1700× |
| `hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf` | 2048 | 0.858278 | 0.1767× |
| `hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf` | 2048 | 0.854337 | 0.1783× |
| `hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf` | 2048 | 0.837467 | 0.1824× |

5090原版33个性能配置只有11个通过精度，另外22个时间保留为失败输入测量，不能当作合格结果。
早期生成器/启动参数编译失败、额外LDS和scratch候选完整保存在实验目录；最终资源检查通过后才正式入账。

## 证据与复现

- [主卡](hgemm_simt_20261106/hip.json)、[第二卡](hgemm_simt_20261106/hip_device1.json)
- [原CUDA](hgemm_simt_20261106/cuda_original.json)、[槽位修复CUDA](hgemm_simt_20261106/cuda_fixed.json)、[专项比较](hgemm_simt_20261106/comparison.json)
- [原FP16全库失败](hgemm_simt_20261106/qualified229_cuda_fp16.json)、[原FP16比较](hgemm_simt_20261106/qualified229_fp16_comparison.json)
- [全库](hgemm_simt_20261106/qualification229.json)、[462组对照](hgemm_simt_20261106/qualified229_comparison.json)、[源码审计](hgemm_simt_20261106/source_audit.json)

```bash
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.hgemm_simt_check fixtures --fixtures artifacts/rocm/continuation_all/u3/fixtures.pt
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.hgemm_simt_check run --fixtures artifacts/rocm/continuation_all/u3/fixtures.pt --output artifacts/rocm/continuation_all/u3/hip.json --performance
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.hgemm_simt_check run --device 1 --fixtures artifacts/rocm/continuation_all/u3/fixtures.pt --output artifacts/rocm/continuation_all/u3/hip_device1.json
```

复验前核对GPU占用、预约及PCI；大fixture、输出及二进制在两端专用artifacts目录。
