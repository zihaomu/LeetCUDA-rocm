# 第四轮：tiling-QK / tiling-QKV FlashAttention

2026-11-05。新增两个原始入口，覆盖率从 **168/229（73.36%）** 提升到
**170/229（74.24%）**，增加 0.87 个百分点。FlashAttention 从 2/29 增至
**4/29（13.79%）**；剩余 **43 个待迁移、16 个精度未验收**。
这是原 Python 绑定接口的功能覆盖率，不是整个仓库或性能验收完成率。

## 接口与实际实现

两个入口均通过 `leetcuda_rocm.load("flash_attn")` 调用：

| 原始入口 | AMD 实现的区别 |
| --- | --- |
| `flash_attn_mma_stages_split_q_tiling_qk_acc_f32` | Q/K 沿 D 分块；V 沿 key 维分成 16×OD 条带，同时保留该输出分片的 PV 累加器 |
| `flash_attn_mma_stages_split_q_tiling_qkv_acc_f32` | Q/K 沿 D 分块；V 沿输出列分成 BC×16 条带，一次仅保留一个 PV tile 的临时累加器 |

实现：[flash_attention_tiled.cu](../../kernels/rocm/flash_attention_tiled.cu)。
两种路径均使用 gfx1201 FP16-input / FP32-accumulator WMMA，有独立的 V 装载顺序和
LDS 布局。stage 1 装载当前块后计算；stage 2 使用 VGPR 预取、显式 load-counter
等待和两个 LDS 槽。保留原 stage 分派：`<=1` 映射到 1，`>1` 映射到 2。
该流水线是 NVIDIA `cp.async` 的 AMD 替代实现。

支持连续、同设备、同形状的 FP16 `q/k/v/out`，形状为 `[B,H,N,D]`，
**D=32/64/96/128/256/512/1024，两种 stage 都支持**。N 可有尾部，空 B/H/N 返回空结果。
输出不得与输入共享存储，不提供 autograd、causal、GQA、dropout 或任意 mask。
限制元素数小于 INT32_MAX、B×H 不超过 65535；非法 dtype、shape、stride 等拒绝执行。
此原入口不支持 BF16；已有独立 `leetcuda_rocm.flash_attention` 的 BF16 支持单独统计。

```python
from leetcuda_rocm import load

lib = load("flash_attn")
lib.flash_attn_mma_stages_split_q_tiling_qk_acc_f32(q, k, v, out, 2)
lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32(q, k, v, out, 1)
```

## 精度与资源取舍

原 CUDA 的 `acc_f32` 名称不表示全程 FP32：QK/PV 矩阵积和 softmax 的 max/sum
使用 FP32，P 在 PV 前存为 FP16。**D<256 的输出状态为 FP32；D>=256 每次 key
block 更新后都将输出状态舍入到 FP16**，最终除以分母并输出 FP16。本批保留这一约定，
没有新增精度升级项；清单中的显式累加精度升级仍为此前的 10 个 GEMM/GEMV 接口。
不同矩阵指令和 exp/归约实现仍有舍入差异，不承诺逐位等价。

专门构造 Q=K=0、32 个 key block、首块 V=1、其余 V=2^-11 的输入：
大 D 时逐块 FP16 状态应得到 **0.03125**，全程 FP32 状态会得到 **0.03173828125**。
两端各 28 个配置全部逐元素精确匹配独立 CPU 状态递推，避免宽松容差掩盖精度改变。

保留原 query/key tile：D<128 时 BR=BC=64，否则 BR=BC=128，因此大 D 的
状态舍入次数保持一致。为限制寄存器和 LDS，D>128 按 **128 列输出**分配独立 CTA，
各 CTA 重算完整 QK；D=1024 时重复 8 份 QK。这是可见的性能成本，不是原 CUDA
分块策略的完整复现。每个 CTA 仅保存局部 P tile，不产生全局 N×N 概率矩阵。

发布二进制中 **28/28 个特化使用 WMMA，均未发现 scratch 指令**。动态 LDS 从
12 KiB 到 48 KiB；D>=128 时 stage 1 为 40 KiB、stage 2 为 48 KiB。
14 个 stage-2 特化均发现线性指令序列中的 load → WMMA → wait；这只证明指令顺序，
不等于周期级重叠或吞吐提升证明。逐特化数据与二进制/汇编 hash 见
[AMD 专项报告](flash_tiled_20261105/hip.json)。

## 验证结果

| 验证 | 结果 |
| --- | --- |
| AMD 共享 CPU 输入，对 FP64 参考 | 248/248 通过 |
| AMD 尾部、空输入、参数契约、非默认 stream／并发 | 578/578 通过 |
| AMD 输出状态精度检查 | 28/28 通过 |
| AMD 专项合计 | **854/854** |
| RTX 5090 原 CUDA，对相同 FP64 参考 | 248/248 通过 |
| RTX 5090 输出状态精度检查 | 28/28 通过 |
| CUDA / AMD 对照 | **248/248 在容差内一致，24/248 逐位一致** |
| 新批次性能及填 NaN 后 graph replay | 两端各 36/36 通过数值与 replay 检查 |
| 最新完整 AMD 基础回归 | **24 模块、170 接口、931 项通过** |

共享输入覆盖两个随机种子、D 全集、N=128/256/384、batch/head 组合、零 V、
均匀 softmax、大 logits、正负抵消，以及 D=256/512/1024 的 N=4096 长序列。
边界覆盖 N=1/17/63/65/127/129/257、未对齐连续视图、stage 别名和六条并发 stream。
数值门限保留 `atol=rtol=0.003`；两端相对 FP64 的最大绝对误差均为约 0.001311，
两端输出最大绝对差为 0.0009765625。检查数不等于独立输入数或接口数。

原 CUDA 来自固定上游提交 `3dbd618e703943e6761db05f29dd66cb7de94d13`，
仅将 launch 接到当前 stream，用同一份冻结 fixture；没有修改 CUDA 算术算法。
输入、源码、头文件、构建产物、输出与测试程序均记录 SHA256。
完整 AMD 回归包含原来两个 shared 入口；已有历史报告保留。

证据：[CUDA 专项](flash_tiled_20261105/cuda.json)、
[逐项对照](flash_tiled_20261105/comparison.json)、
[完整 AMD 资格](qualification_20261105_flash_tiled.json)。

全部 **170 个接口**使用另一份共同 fixture 完成基础跨平台回归：CUDA 与 AMD 均
**344/344** 通过独立参考，**344/344** 输出在接口容差内一致，**220/344** 逐位一致。
全库 CUDA 使用 `ported/` 工作树，包含此前已公开的 CUDA 修复及精度调整；本批两项
专项使用上游 `original/`。有限形状回归不代替专项边界和长序列检查。
证据：[全库 CUDA](flash_tiled_20261105/qualified170_cuda.json)、
[全库 AMD](flash_tiled_20261105/qualified170_hip.json)、
[全接口逐项对照](flash_tiled_20261105/qualified170_comparison.json)。

## 性能观察

在相同冻结输入上，graph 每次 8 个调用、自定义实现与同机 Torch SDPA 交替测量、
每项取 3 组样本中位数。所有计时项均先通过独立参考和填 NaN 后 graph replay。
下表列出 tiling-QK、stage 2 的全部形状，单位为微秒：

| B×H×N×D | AMD 自定义 / μs | AMD SDPA / μs | 5090 原 CUDA / μs |
| --- | ---: | ---: | ---: |
| 1×4×128×32 | 10.78 | 4.92 | 4.10 |
| 1×4×256×64 | 18.52 | 7.65 | 10.26 |
| 1×4×512×64 | 33.06 | 14.78 | 19.07 |
| 1×4×1024×64 | 67.70 | 26.11 | 36.79 |
| 1×4×1024×128 | 102.10 | 28.47 | 60.23 |
| 1×4×256×256 | 35.96 | 24.28 | 32.87 |
| 1×4×512×256 | 66.44 | 43.39 | 61.20 |
| 1×4×512×512 | 121.15 | 53.45 | 168.54 |
| 1×4×512×1024 | 374.42 | 400.29 | 525.78 |

多数形状仍落后同机 SDPA；D=1024 这一组自定义实现略快，但采样确认该形状的
SDPA 使用 `aten::_scaled_dot_product_attention_math`。不能把这一结果解释为
超过 fused FlashAttention；其他形状的 SDPA 派发没有逐项采样。
两端完整 36 组原始样本均在 JSON 中，包括 tiling-QKV 和 stage 1；没有只保留获胜配置。
RTX 5090 与 R9700 的硬件、软件栈不同，跨卡耗时不作硬件归一化速度结论。

目前主要成本包括大 D 重复 QK、多个 LDS barrier、分片 V 装载与转置。
下一批优先保留原算法差异地实现 swizzle 及 Q/K 装载优化；随后推进 split-KV，
单独验证分区 softmax max/sum 和输出合并。其他未迁移项仍见
[支持清单](leetcuda_rocm10_operator_support.md)，本批没有把它们改成已支持。

## 设备与验证边界

最终 AMD 专项、性能、完整回归均使用 **PCI `0000:c3:00.0`，物理 GPU6，HIP mask 5**，
ROCm 10.1 / HIP 7.16 / Torch 2.14，镜像和具体版本见 JSON。选择时该卡无显存占用、
无活跃计算；主机其他卡有任务，未停止或修改任何其他任务。物理 GPU3 始终被排除。

首轮使用当时确认空闲的物理 GPU5 / PCI `0000:a3:00.0` / HIP mask 7，也通过 854 项。
实验期间预约文件更新为占用 GPU5；发现后停止在该卡安排后续任务，改用未预约的 GPU6，
重新运行完整专项与性能。首轮数据保留为
[初次运行记录](flash_tiled_20261105/hip_gpu5_initial.json)，本报告性能表仅使用最终 GPU6 数据。
[预约更新](flash_tiled_20261105/reservation_updated.json)、
[最终选卡前占用](flash_tiled_20261105/occupancy_gpu6_before.json)、
[回归后占用](flash_tiled_20261105/occupancy_after.json)均保留。

两张卡的专项都只暴露一张设备；本批尚未完成跨设备混用拒绝和 device guard 恢复的
双可见设备补测。不能将单卡 stream 检查代替这两项，也不外推到其他 AMD 架构。

## 复现

选卡前重新核对实际 PCI、占用和预约；以下 mask 只表示本次选择。大二进制、fixture
和 tensor 输出保存在本地 `artifacts/rocm/flash_tiled/` 及 5090 专用目录，Git 保存源码与 JSON。

```bash
export HIP_VISIBLE_DEVICES=5
scripts/rocm/run.sh python -m leetcuda_rocm.flash_tiled_check fixtures \
  --fixtures artifacts/rocm/flash_tiled/fixtures.pt
scripts/rocm/run.sh python -m leetcuda_rocm.flash_tiled_check run \
  --fixtures artifacts/rocm/flash_tiled/fixtures.pt \
  --output artifacts/rocm/flash_tiled/hip.json --performance

# 5090 专用目录，传入同一份 fixture 后：
LEETCUDA_CUDA_GPU=0 scripts/cuda/run.sh python -m leetcuda_rocm.flash_tiled_check run \
  --backend cuda --manifest artifacts/cross/pending_manifest.json --source-root original \
  --fixtures artifacts/flash_tiled/fixtures.pt --output artifacts/flash_tiled/cuda.json --performance

# 传回 CUDA JSON 和同名 .pt 后：
scripts/rocm/run.sh python -m leetcuda_rocm.flash_tiled_check compare \
  --cuda artifacts/rocm/flash_tiled/cuda_remote/cuda.json \
  --hip artifacts/rocm/flash_tiled/hip.json --output artifacts/rocm/flash_tiled/comparison.json
scripts/rocm/run.sh python -m leetcuda_rocm.smoke \
  --output artifacts/rocm/qualification_flash_tiled.json
```
