# GQA / mask / dropout / Attention反向验收

2026-11-06，锁定ROCm10.1/HIP7.16，R9700 gfx1201。**新增`leetcuda_rocm.attention`功能验收完成。**
原229个公开接口仍为229/229；独立AMD计算API由6增至7，两个权重量化辅助API另列。
原`flash_attention`前向接口的契约保留，本项为独立训练扩展。

## 接口与算法

```python
from leetcuda_rocm import attention
out = attention(q, k, v, mask=mask, causal=True,
                dropout_p=0.125, seed=17, offset=0)
out.backward(dout)
```

- Q为连续`[B,HQ,NQ,D]`，K/V为连续`[B,HK,NK,D]`；同GPU、同FP16/BF16，HQ为HK整数倍。
- D支持16/32/64/128/256；空B/NQ/NK、序列尾部和未对齐连续视图已验证。
- 因果位置为**左上对齐**：key索引≤query索引，允许NQ≠NK。
- bool mask的True表示保留；加性mask为有限值或负无穷，支持二维或可广播四维形状、非连续stride。全mask行返回0。
- dropout使用显式Philox4x32-10 seed/offset。相同参数重放相同掩码，包括graph；训练调用间由调用者推进offset。
  按完整`B*HQ*NQ*NK`坐标取counter，mask/causal不改变其他位置随机数。counter范围限制为uint32，seed为非负63位整数。
- 支持Q/K/V任意需要梯度的组合、第一阶反向；明确拒绝mask梯度，不提供二阶导数。输入和logits须有限，不在运行时扫描该数值前提。

Triton前向使用在线softmax和真实WMMA，P转输入dtype参与PV，FP32状态累加；只保存线性的LSE。
反向重算概率，分别分块计算delta、dQ、dK/dV；GQA的dK/dV由一个CTA循环所有对应query heads，确定性归并且无全局原子累加。
所有非空矩阵路径均检查实际WMMA，空序列路径明确为零填充。
不分配全局NQ×NK概率矩阵；用户传入的mask本身不计为算子临时空间。

## 验收

| 项目 | 结果 |
| --- | --- |
| AMD两卡 | 各377/377；含84组冻结输入的336份前向/梯度输出及运行时检查 |
| RTX5090同算法参考 | 373/373；少4项AMD双设备guard/混用检查 |
| 共同输入输出比较 | 336/336通过，134份逐位一致 |
| 公开API补验 | 62/62；自定义零/负/正scale、不同梯度需求组合、二阶导数拒绝 |
| 两端性能输入独立参考 | 各52/52，含长序列解析梯度 |
| 两端性能 | 各12个配置：FP16/BF16、三种形状，前向及前向+反向分别测量 |
| 工作空间 | 两端各7项通过；AMD N4096/D64/GQA额外5,373,952字节（约5.13MiB） |
| 原接口集成回归 | 25模块、229接口、1262/1262 |

固定门限：FP16 atol=0.003/rtol=0.01，BF16 atol=0.015/rtol=0.03；对独立CPU FP64数学输出和梯度比较。
不是将概率量化定义为离散函数后求零导数；采用标准Attention数学梯度的低精度近似，报告保留最大绝对误差与相对L2。
Philox由独立CPU整数轮函数精确核对4096个counter，覆盖高seed位与接近uint32上限；另测重复和推进offset。
覆盖全D、两个seed、三类mask/GQA/因果/dropout组合，空序列、长KV1025、零QK、全mask与大logits。
6条并发stream各3次调用，训练graph全部输出/梯度先填NaN再回放，跨设备拒绝与guard恢复均通过。

N4096的额外PyTorch工作空间等于线性输出/梯度/LSE/delta理论总和；对应密集FP16概率需要128MiB。
该项同时精确验证uniform attention的O=1、dQ=dK=0、dV=4，不仅计量内存。

CUDA比较明确为同一Triton算法的第二后端，**不是原仓库已有训练Attention的声明**；两端实现源码hash相同，CPU oracle独立。
AMD使用物理GPU4/PCI83/HIP6和GPU5/PCIa3/HIP7，物理GPU3未参与；CUDA功能用5090 GPU1、性能用GPU0，均在专用固定环境。

## 已解决的问题与性能差距

1. 原候选用已舍入O计算delta，在BF16单key下产生本应为零的dK，最大误差0.01904。改为从未舍入概率和dP重算delta，原门限下通过。
   这一方案增加反向重算，性能成本明确保留；没有对单key做掩盖性特判。
2. 原dK/dV query tile64在5090 D256需102400B共享内存，超过101376B。改为32后两端验证通过。
3. 空序列只写零，编译器移除矩阵指令；ISA gate限定实际非空计算，空路径另验证。
4. 性能脚本曾复用warmup的autograd节点造成graph默认stream引用失效。每次测量使用共享输入存储的新leaf view，保证节点属于当前capture stream；失败日志保留。

AMD实际记录186个编译特化，LDS最大32KiB；其中63个有spill，性能优化未关闭。
计时每graph8次，5组交替40ms样本；基线为同卡Torch SDPA，纯前向和前向+反向分开。
以下为Torch/本实现，超过1表示本机测得更快；不把不同GPU的耗时比作为优化收益。

| dtype | N/D/HK（HQ=4） | 前向 | 前向+反向 |
| --- | --- | ---: | ---: |
| float16 | 256/64/1 | 0.971× | 1.322× |
| float16 | 512/128/2 | 1.095× | 0.993× |
| float16 | 1024/256/4 | 0.892× | 0.421× |
| bfloat16 | 256/64/1 | 1.081× | 1.359× |
| bfloat16 | 512/128/2 | 1.087× | 1.115× |
| bfloat16 | 1024/256/4 | 0.904× | 0.378× |

D256训练仍明显慢于SDPA，后续继续减少spill和重算成本。小规模GQA观察到约1.32–1.36×，不外推到全部形状。

## 证据与复现

- [AMD主卡](attention_training_20261106/hip_recomputed.json)、[第二卡](attention_training_20261106/hip_device1.json)
- [5090](attention_training_20261106/cuda_recomputed.json)、[336份比较](attention_training_20261106/comparison.json)
- [AMD性能/空间](attention_training_20261106/performance.json)、[5090性能/空间](attention_training_20261106/cuda_performance.json)
- [公开API](attention_training_20261106/api.json)、[全库](attention_training_20261106/qualification229.json)、[源码审计](attention_training_20261106/source_audit.json)
- [原梯度失败报告](attention_training_20261106/hip_candidate.json)

```bash
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.attention_training_check fixtures --fixtures artifacts/rocm/attention/fixtures.pt
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.attention_training_check run --fixtures artifacts/rocm/attention/fixtures.pt --output artifacts/rocm/attention/hip.json --device 0 --guard-device 1
HIP_VISIBLE_DEVICES=6 scripts/rocm/run.sh python -m leetcuda_rocm.attention_training_performance --output artifacts/rocm/attention/performance.json
```

运行前核对GPU占用/预约。FFPA、整体独立构建、非绑定源码和性能收敛继续，不以本项功能完成结束整体任务。
