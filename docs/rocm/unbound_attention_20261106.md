# 未绑定tiling-QK RR Attention兼容入口验收

2026-11-06。`load_unbound("flash_attn")`现提供`flash_attn_mma_stages_split_q_tiling_qk_acc_f32_rr`，补充的4个未绑定计算入口全部完成功能验收。原25模块229接口分母不变，interview独立53个kernel的进度仍单独统计。

## 复用依据

原`mma/others/flash_attn_mma_tiling_qk_F32F16F16F32_rr.cu`与`mma/basic/flash_attn_mma_tiling_qkv_F32F16F16F32.cu`，**整份文件**在去掉注释、空白并统一入口名称后完全一致。两份源文件均逐SHA核对不可变上游`3dbd618e703943e6761db05f29dd66cb7de94d13`。
规范化SHA256为`a37ebc3a46be4491c0d5cb14132bba25d69ac3d64a1772b2cf6fdd577b94dec6`；因此该兼容名称直接指向已验收的tiling-QKV原生AMD实现，加载时也检查源码等价性。
这不是新增一种计算算法；文档明确记录两名称的对应关系。保留真实Q/K维度分块、沿输出D分块的V搬运、单个PV累加片段以及原stage分工。

## 验收

| 项目 | 结果 |
| --- | ---: |
| AMD物理GPU4 / 0000:83:00.0 | 553/553 |
| AMD物理GPU0 / 0000:03:00.0 | 553/553 |
| 5090原CUDA rr源文件 | 262/262 |
| 共同输入跨平台输出 | 248/248，其中24项逐位一致 |
| AMD ISA | 14个D/stage特化，零scratch/spill，实际WMMA与128位加载 |
| 性能 | AMD和5090各18组同机SDPA对照 |

553=124普通输出+124 capture后NaN覆盖的Graph输出+14精确累加状态检查+291边界/设备/stream检查；CUDA采用同一62组冻结CPU fixture，含124普通和124Graph输出及14状态检查。
覆盖D32/64/96/128/256/512/1024，stage1/2，两随机种子，N128/256/384与N4096长序列、零V、均匀logits、大logits、抵消以及多batch/head。
AMD额外涵盖N1/17/63/65/127/129/257、空维度、half非16字节对齐、非法dtype/shape/alias/grad/device，以及6条非默认stream和设备guard恢复。
共同输出用实际FP16输入的CPU FP64参考，保持原Attention门限atol=rtol=0.003；边界专项使用独立Torch FP64参考。
**D>=256仍在每个128-key块后将运行输出舍入FP16**，14个精确状态测试明确区分该语义与全FP32状态，未静默升级。

ISA的VGPR为118–221，动态LDS为12–48KiB。共用实现的输入/QK/PV分块及stage1/2均保留，原CUDA仅进行当前stream适配。
18个性能配置采用32调用Graph、约40ms自适应测量、5轮交替event采样。本机SDPA/本实现比：AMD0.187–1.034，CUDA0.221–0.931；大D默认SDPA可能选择math后端，应按各形状阅读，不能用跨GPU耗时比表示移植加速。B1 Q/QK的固定性能门槛仍未闭合。

```python
from leetcuda_rocm import load_unbound
m = load_unbound("flash_attn")
m.flash_attn_mma_stages_split_q_tiling_qk_acc_f32_rr(q, k, v, out, 2)
```

输入输出均为FP16连续`[B,H,N,D]`同形状同设备，输出不共享输入存储，无autograd；stages>1映射2，否则1，延续原入口。仅gfx1201验收。

```bash
HIP_VISIBLE_DEVICES=6,2 scripts/rocm/run.sh python -m leetcuda_rocm.unbound_attention_check run \
  --backend hip --performance --fixtures artifacts/rocm/flash_tiled/fixtures.pt \
  --output artifacts/rocm/continuation_all/unbound_attention/recheck.json
```

镜像、框架、PCI、源码/fixture/输出/二进制hash在报告中冻结；物理GPU3禁止计算。
[主卡](unbound_attention_20261106/hip.json)、[第二卡](unbound_attention_20261106/hip_second.json)、[原CUDA](unbound_attention_20261106/cuda.json)、[跨平台](unbound_attention_20261106/comparison.json)、[资源](unbound_attention_20261106/resources.json)、[来源审计](unbound_attention_20261106/source_audit.json)。

继续interview独立FlashAttention、CuTe/CuTeDSL、swizzle、profiling及M7，整体迁移尚未完成。
