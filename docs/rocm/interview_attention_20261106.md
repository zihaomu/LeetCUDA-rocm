# interview 独立 FlashAttention：8个定义的ROCm验收

2026-11-06。`flash_attn.cuh`的8个GPU定义已有独立AMD实现与真实5090对照，interview功能覆盖可达53/53。原Python接口仍229/229，二者分母独立。新增公开计算入口`interview_flash_attention`，共7种计算调度；另1个定义是布局复制smoke。

## 实现与契约

| variant | 原路径 | AMD实现 |
| --- | --- | --- |
| mma | 手写MMA split-Q | cooperative预取；Q/K/V各自padding或XOR；WMMA |
| tma_ws | 手写TMA/WS | producer wave与consumer waves；K/V独立序号和槽位 |
| fa3 | 双消费者FA3 | 奇偶KV两组独立online状态，稳定合并 |
| cute | CuTe split-Q | GMMA地址布局、寄存器P转置、LDS输出 |
| cute_ws | CuTe WS | 上述布局加独立producer/consumer |
| cute_fa3 | CuTe FA3 | 上述布局加两组KV消费者及稳定合并 |
| persist_d | persistent-D | Q预缩放为FP16、阈值8条件重缩放、持久query CTA、GQA/右下因果 |
| copy smoke | CuTe TMA复制 | global→对应GMMA布局LDS→global |

输入/输出FP16，QK/PV/online状态FP32；手写原F16累加模式明确升级，不声称逐位保留原F16 MMA。CUDA参考使用上游的FP32模板选项。
多数路径D64/128；mma额外D32，persist-D额外D96。仅persist-D支持NQ≠NKV、GQA、右下对齐因果；所有计算路径支持显式scale及AMD有界尾块。
无dropout、任意mask和autograd；这些需求由已验收的`attention`训练API承担。

K深度1–4，WS的V深度1–2，其余V深度1；persist-D的K/V均1–2。真实槽位与提前量由运行时参数控制，固定最大LDS容量仍被保留。AMD FA3预填实际SK深度，原CUDA某些配置只提前1块，调度差异明确保留。
受64KiB LDS限制，AMD采用BR32/64与BC16/32/64；保持各路径分工。实际128位global读取至VGPR，等待后写LDS，并通过序号/释放获取同步消费，不冒充硬件TMA。
CuTe布局的9组坐标与冻结Cutlass主机参考全部一致。

## 验收结果

| 检查 | 结果 |
| --- | ---: |
| AMD物理GPU4 / PCI83 | 2578/2578 |
| AMD物理GPU0 / PCI03 | 2578/2578 |
| 最终格式整理后复验 | 2578/2578 |
| 独立HIP C++直接头调用 | 646/646，最终统一构建再次通过 |
| CUDA原版FP32模板 | 1218/1272；54失败保留 |
| CUDA basic索引/warmup修复 | 220/220 |
| CUDA persist-D全mask行修复 | 108/108 |
| 合并匹配参考与AMD输出 | 1280/1280 |
| 无padding额外对照 | AMD48；CUDA44；共同44/44 |
| 根包公开API/Graph | 14/14 |

2578=2442数值/Graph/copy+66流水线+54契约/设备+9布局+7显存检查。77组冻结CPU输入，SHA256=`3b0f58f9c0d532d7b2cdbc9631f8cc0bb8760aa360a92d4d2682fdd9e1e3a6fe`。
覆盖两种种子、随机/均匀/零V/大logits/抵消/重缩放/奇偶最大值、N17/65/137尾部、N2048、NKV4096、GQA、不同Q/K长度与因果空行。CPU FP64分query块计算参考；persist-D另复现原Q的FP16预缩放。
输出门限atol=rtol=0.003，LSE为2e-5；不放宽门限。Graph捕获后NaN覆盖输出，非默认stream与canary验证。生产者延迟注入、序号、最大领先深度和persistent CTA复用均实测；N4096检查没有额外N×N张量。

原CUDA失败分开处理：

- basic的48项失败来自stage3/4预取源地址`t+1`与目标槽`t+SK-1`错位；短N warmup还有越界风险。修复参考只改地址、warmup边界及首次wait。
- persist-D的6项失败来自NQ>NKV因果全mask行的`-inf - -inf`。修复参考仅在当前/历史max均为负无穷时将tile概率清零、保持状态。
- 原版169条case/stage不可运行记录包含共享内存超5090的101376B上限及已知unsafe warmup，未冒充通过。无padding补验另有2条资源不可运行记录。AMD这些配置由CPU参考和自身流水线审计覆盖。

## 性能与资源限制

两端各32组同卡Torch SDPA比较，原始交替Graph/event样本随JSON保存。AMD `T_SDPA/T_native`为**0.030–0.278**，CUDA原版0.248–0.842。修复basic另外重测6组；原版性能不代表所有输入正确。不同GPU耗时比不用于移植加速声明。

32个计算特化（16生产+16audit）均有真实WMMA和128位global读取，LDS≤65536B。生产特化中**14/16仍有spill**，最大scratch1408B；仅mma D32与fa3 D64无spill。条件向量写入改为无条件后消除D32私有alloca，32次shuffle替换64次概率转置。全展开、fence和禁用SLP候选均未解决大D资源问题，失败资源记录保留。因此此报告关闭功能缺口，**不宣称FA性能或资源优化达标**。

## 使用与复现

```python
from leetcuda_rocm import interview_flash_attention
out, lse = interview_flash_attention(q, k, v, variant="persist_d",
    stages_k=2, stages_v=2, causal=True, return_lse=True)
```

```bash
HIP_VISIBLE_DEVICES=6,2 scripts/rocm/run.sh python -m leetcuda_rocm.interview_attention_check run \
  --fixtures artifacts/rocm/continuation_all/interview_fa/fixtures.pt \
  --output artifacts/rocm/continuation_all/interview_fa/recheck.json --performance
HIP_VISIBLE_DEVICES=6 scripts/rocm/build_all_bench.sh --output artifacts/rocm/all_cpp
```

仅锁定ROCm10.1.0/HIP7.16/Torch2.14/gfx1201通过；物理GPU3禁止计算。源代码、镜像/PCI、输入/输出、二进制与ISA hash随报告保存。
[主卡](interview_attention_20261106/hip_primary.json)、[第二卡](interview_attention_20261106/hip_secondary.json)、[最终复验](interview_attention_20261106/hip_formatted.json)、[原CUDA](interview_attention_20261106/cuda_original.json)、[修复匹配对照](interview_attention_20261106/cross_matched.json)、[最终资源](interview_attention_20261106/formatted_resources.json)、[来源审计](supplemental_examples_20261106/source_audit.json)。
