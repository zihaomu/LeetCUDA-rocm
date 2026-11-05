# 第五轮：Q / QK / QKV swizzle FlashAttention

2026-11-05。按[预先制定的 B1 计划](leetcuda_rocm10_next_batch_plan.md)，顺序实现三个
FP32 tiling-QKV swizzle 入口。**新增 3/3 个接口，功能 F0～F6 全部通过；性能 P 达标 1/3**。
原接口从 170/229（74.24%）增至 **173/229（75.55%）**，增加 1.31 个百分点；
FlashAttention 达到 **7/29（24.14%）**。剩余 **40 待迁移、16 精度未验收**。
QKV 三轮相对冻结父实现约 1.117×，Q/QK 功能支持、性能待优化。整个仓库迁移仍未完成。

## 接口与实现

公共前缀为 `flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_`：

| 后缀 | 实际改变的 LDS 操作数 | 模板 mask |
| --- | --- | ---: |
| `q` | Q | 1 |
| `qk` | Q、K | 3 |
| `qkv` | Q、K、转置后的 V | 7 |

三个入口共享算法模板，分别执行不同的地址排列。实际 GPU 实现位于
[flash_attention_tiled.cu](../../kernels/rocm/flash_attention_tiled.cu)，
通过 `leetcuda_rocm.load("flash_attn")` 调用，QK/PV 均使用 gfx1201 WMMA。
现有两个 shared、两个非 swizzle tiling 入口保留，新增接口没有替换其调用路径。

AMD 地址公式以 FP16 元素为单位：

```text
physical(row, col) = row * Cols + (col XOR (((row >> 2) & 1) << 3))
```

Q/K 在 `BR×16`、`BC×16` 面板中应用；V 先按 `BC×16 → 16×BC` 转置，再应用同一公式。
地址映射为双射且自身互逆，保留连续 8 个 FP16。新路径显式使用对齐的 128-bit
向量读取，Q/K 发布到 LDS 时也以向量写入；V 转置写入仍按其实际布局执行。
这是适合当前 AMD fragment 的实现，不照搬原 CUDA 的共享内存地址公式。

接口契约保持上一批约定：

- FP16 连续、同设备、同形状 `[B,H,N,D]`；输出不得与输入共享存储。
- D=32/64/96/128/256/512/1024；`stages<=1 → 1`，`stages>1 → 2`。
- QK/PV 与 softmax max/sum 为 FP32，P 转为 FP16；D<256 的输出状态为 FP32，
  **D>=256 在每个 key block 后舍入为 FP16**。
- Query/key tile 在 D<128 时为 64×64，否则为 128×128。D>128 每 CTA 只处理
  128 列输出，因此仍重复计算 QK；本批没有改变状态舍入频率。
- AMD 支持 N 尾部、空 B/H/N、未对齐但连续的输入；非法 dtype、stride、shape、
  autograd、跨设备、别名、索引／grid 超限均拒绝。
- Stage 2 使用真实双 LDS 槽、VGPR 预取及 load-counter 等待；没有全局 N×N 概率矩阵。
  本批不增加 BF16、causal、GQA、mask、dropout 或 backward。

```python
from leetcuda_rocm import load

lib = load("flash_attn")
lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_q(q, k, v, out, 1)
lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qk(q, k, v, out, 2)
lib.flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv(q, k, v, out, 2)
```

## 环境和证据身份

AMD 固定 ROCm 10.1.0、HIP 7.16.26385、Torch 2.14.0、Triton 3.8.0，
使用项目启动器及 gfx1201 独立缓存。镜像：
`sha256:a79c374ab2bc93c0e4358da6c7bda5cb84cf1308e8aff74fc99fc95c511cbd03`。

本批 `HIP_VISIBLE_DEVICES=2,1` 对应逻辑 0 / PCI `0000:03:00.0`（物理 GPU0）和
逻辑 1 / PCI `0000:43:00.0`（物理 GPU2）。主卡计时前确认空闲；第二张卡重复正确性。
物理 GPU3（PCI `0000:63:00.0`）未参与计算。全库单卡报告使用 `HIP_VISIBLE_DEVICES=2`。
报告记录 PCI、实际可见 mask、框架身份、温度、频率与占用采样；采样不构成长期独占保证。

5090 仍使用 `nvidia-5090x2` 上的
`/home/zihaomu/zihaomu_workspace/test_lap/leetcuda_rocm10`，本批选择 GPU0。
CUDA 镜像为 `sha256:42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee`。
新入口专项取固定上游 `3dbd618e703943e6761db05f29dd66cb7de94d13` 的 `original/` 源码，
仅统一 current-stream 启动方式。全接口回归取包含此前 CUDA 修复的 `ported/`，来源分开记录。

新入口沿用上一批的 62 组 CPU fixture，文件 SHA256：
`682f0cdfdee5a2d06b7970ada52335afe26cc77aa663985ceddc863759816c03`。
每个入口跑两种 stage；两端检查逐 tensor hash，记录源码、测试程序、辅助程序、
二进制、ISA 和输出 hash。冻结父实现来自提交 `66bb7fb852536d26cbb79bf1c122bb6c8c3b16e4`，
以其独立二进制加载，不用当前重新编译的父函数代替冻结性能基线。
多次链接的 `.so` 因 offload 对象排列不同可能具有不同 hash；最终四份 AMD 专项均保留
各自不可变二进制，提取出的设备对象 hash 集合一致。

## 功能验收 F0～F6

| 验收范围 | 最终结果 | 证据 |
| --- | --- | --- |
| AMD 主卡专项 | 1329/1329：共享 372、边界 879、状态 42、设备/stream 28、布局 8 | [hip.json](flash_swizzle_20261105/hip.json) |
| AMD 第二张卡专项 | 同一覆盖 1329/1329；共享输入 372 项均逐位匹配冻结父实现 | [hip_device1.json](flash_swizzle_20261105/hip_device1.json) |
| 5090 原 CUDA 专项 | 372/372 参考检查、42/42 状态检查，162 条三轮性能记录 | [cuda.json](flash_swizzle_20261105/cuda.json) |
| 新入口跨平台 | 372/372 在容差内一致，36/372 逐位一致 | [comparison.json](flash_swizzle_20261105/comparison.json) |
| 旧 shared 专项 | 538/538：共享 146、边界/契约/stream 392 | [shared_regression.json](flash_swizzle_20261105/shared_regression.json) |
| 旧 tiling 专项 | 858/858：共享 248、边界 582、原状态精度 28 | [tiling_regression.json](flash_swizzle_20261105/tiling_regression.json) |
| 四个既有 Attention 入口与冻结二进制 | 394/394 数值逐位一致；174/174 性能记录符合≤10%退化门槛，最坏约 3.45% | [existing_regression.json](flash_swizzle_20261105/existing_regression.json) |
| 全库 AMD 单卡 | 24 模块、173 原接口、937/937 检查通过，旧 170 接口无遗漏 | [基础资格](qualification_20261105_flash_swizzle.json) |
| 全接口 CUDA/AMD | 350/350 两端参考与跨平台容差通过 | [全接口对照](flash_swizzle_20261105/qualified173_comparison.json) |
| 最终源码/辅助程序/ISA/输出身份 | 201 项 hash 核对、108 个不同路径；四份 AMD 最终专项的设备对象 hash 集合一致 | [身份审计](flash_swizzle_20261105/source_artifact_audit.json) |

边界项较计划的双卡预计 873 项多六项：每入口新增两个显式索引/grid 上限拒绝检查。
42 个特化均有 WMMA 和资源记录。双设备检查覆盖旧四个和新增三个 Attention 入口。
完整批次判定及数据 hash 见 [batch_acceptance.json](flash_swizzle_20261105/batch_acceptance.json)。
主卡与第二张卡的 372 项共享输入均逐位匹配冻结父实现，主卡相对 FP64 的最大绝对误差约 0.001311。

共享输入检查要求输出有限，与独立 FP64 参考转成 FP16 后满足 `atol=rtol=0.003`，
同时保存相对 FP64 原值的最大绝对误差。AMD swizzle 与冻结非 swizzle 父实现要求逐位一致。
CUDA/AMD 只要求同一容差内一致，逐位数量单列。没有放宽容差或排除原有困难样本。

状态精度构造为 Q=K=0、32 个 key block、首块 V=1、后续 V=2^-11：
D>=256 必须精确得到 **0.03125**，不能变成全 FP32 状态的 **0.03173828125**。
42 项覆盖三个入口的全部 D×stage，旧 tiling 的 28 项状态检查另外保留。

布局探针直接编译生产源中的 `issue/publish/lds_index`，对 64/128 行、转置与否、
swizzle 与否共八种组合，核对实际 GPU LDS 的物理排列和逻辑回读。两个槽使用不同
行列标识，CPU 枚举所有地址检查有界、无冲突和互逆。这两种面板覆盖全部支持 D。
数值专项使用输出 canary；运行时检查包括六条并发 stream、foreign-current device、
guard 恢复及 graph replay 前填 NaN，避免把空图或错误 stream 计为通过。

## 性能和资源门槛 P

固定 B=1,H=4，九种 `(N,D)` 为 `(128,32)`、`(256,64)`、`(512,64)`、`(1024,64)`、
`(1024,128)`、`(256,256)`、`(512,256)`、`(512,512)`、`(512,1024)`；各跑 stage 1/2。
三个入口每轮 54 条，三轮共 162 条。每 graph 八次调用、五组至少 40 ms 样本，
交替测量新实现和冻结父实现，另配对测量新实现与 SDPA。逐形状保存 SDPA 派发。

预定门槛不变：每入口三轮均满足几何平均 `T_parent/T_new≥1.05`、任一配置
`T_new/T_parent≤1.10`；三轮中位数的相对极差≤5%；LDS≤48 KiB 且无 scratch spill。
不同硬件的绝对时间仅作观测，不当作 AMD 优化加速比。

| 入口 | 三轮几何平均加速（父实现 / 新实现） | 最坏新 / 父耗时 | 三轮稳定性 | 性能 P |
| --- | --- | ---: | --- | --- |
| q | 0.9886× / 0.9878× / 0.9878× | 1.2152× | 通过 | 待优化 |
| qk | 1.0173× / 1.0173× / 1.0171× | 1.2124× | 通过 | 待优化 |
| qkv | 1.1171× / 1.1178× / 1.1158× | 0.9809× | 通过 | 通过 |

**性能达标 1/3，整批性能未达标。** QKV 三轮达到 1.05× 门槛，所有配置均快于父实现，
D=64/128 的八个配置几何平均为 1.1143× / 1.1133× / 1.1137×，也达到 1.10× 冲刺目标。
Q 和 QK 仍未过几何平均及最大退化门槛；这两个入口保留功能支持、性能待优化状态。

- q 最坏点：N=128, D=32, stage=1，耗时为父实现的 1.2152×。
- qk 最坏点：N=128, D=32, stage=1，耗时为父实现的 1.2124×。

最终 42 个编译特化全部包含 WMMA，LDS 为 12～48 KiB，scratch 指令、VGPR spill 与
private segment 均为零。Q/QK/QKV 的 VGPR 范围分别为 120～246 / 118～222 / 118～248。
每个 stage 2 特化均观察到静态 load→WMMA→wait 次序；这不是周期级重叠测量。
D=64、stage 2 的三个新特化均恢复 37 条 `ds_load_b128` 和四条 `ds_store_b128`，
没有逐元素 `ds_load_u16_d16*`；地址映射仍分别作用于 Q、QK、QKV。

下面列出 QKV 全部 18 个配置的三轮中位数（ms）；完整 Q/QK/QKV 原始样本及五组采样
保存在 [AMD 报告](flash_swizzle_20261105/hip.json) 和
[原 CUDA 报告](flash_swizzle_20261105/cuda.json)，性能判定见
[acceptance.json](flash_swizzle_20261105/acceptance.json)。

| N | D | stage | AMD QKV | 冻结父实现 | AMD SDPA | 5090 原 QKV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 1 | 0.008439 | 0.008795 | 0.004858 | 0.003595 |
| 128 | 32 | 2 | 0.010041 | 0.010295 | 0.004786 | 0.003583 |
| 256 | 64 | 1 | 0.022757 | 0.024128 | 0.007338 | 0.010939 |
| 256 | 64 | 2 | 0.021425 | 0.022668 | 0.007301 | 0.010034 |
| 512 | 64 | 1 | 0.041143 | 0.043928 | 0.014006 | 0.020458 |
| 512 | 64 | 2 | 0.039251 | 0.041599 | 0.013996 | 0.018923 |
| 1024 | 64 | 1 | 0.088436 | 0.100613 | 0.025122 | 0.039600 |
| 1024 | 64 | 2 | 0.085400 | 0.095708 | 0.025049 | 0.036503 |
| 1024 | 128 | 1 | 0.128268 | 0.149506 | 0.029046 | 0.082646 |
| 1024 | 128 | 2 | 0.105065 | 0.131460 | 0.028841 | 0.063822 |
| 256 | 256 | 1 | 0.044642 | 0.046262 | 0.023222 | 0.042465 |
| 256 | 256 | 2 | 0.037769 | 0.046769 | 0.023115 | 0.032515 |
| 512 | 256 | 1 | 0.082318 | 0.086096 | 0.041227 | 0.081299 |
| 512 | 256 | 2 | 0.069999 | 0.086637 | 0.041466 | 0.061359 |
| 512 | 512 | 1 | 0.144390 | 0.163165 | 0.054335 | 0.153872 |
| 512 | 512 | 2 | 0.135476 | 0.166163 | 0.053815 | 0.112612 |
| 512 | 1024 | 1 | 0.458702 | 0.475668 | 0.407391 | 0.334669 |
| 512 | 1024 | 2 | 0.393089 | 0.470752 | 0.408764 | 0.237946 |

被测 D≤512 的 AMD SDPA 实际派发为 flash，D=1024 为 math。新 QKV 仍明显慢于大多数
同机 fused SDPA 配置；D=1024、stage 2 接近或略快于 math，不能据此宣称超过 fused FlashAttention。
较大的 D 仍重复 QK，Q/QK 的退化还需结合指令调度和 LDS 访问继续研究，当前未证明单一原因。

## 实验中解决的问题和保留的记录

1. 按 Q→QK→QKV 逐入口完成初版验证与原 CUDA 对照，各有 124 项共享输入、
   14 项状态检查、54 条三轮性能记录。初版使用 `col XOR ((row & 7)<<1)`，
   各自与冻结父实现数值一致，但性能几何平均约 0.885×、0.679×、0.677×。
   QKV 有八个特化出现 scratch spill。
2. 改为保留连续八元素的地址公式后，逐元素表达式仍使编译器生成窄 LDS 读取。
   该候选 QKV 性能约 0.522×，仍有 spill；保留该候选结果，没有把地址连续等同于
   编译器一定生成向量读取。最终实现对新路径显式表达 128-bit 对齐访问。
3. 测试探针第一次缩减编译范围时遗漏命名空间闭合，已修复；锁定容器不提供
   `rocm-smi`，运行内遥测改读 sysfs。两者属于验证工具问题，失败日志保留在 artifacts。
4. 一次测试重启时旧进程继续写日志，造成日志交叠；那次性能结果单独隔离。
   确认进程退出后重跑完整三轮，最终结果取独立日志和最终源码，不挑选最好一轮。

初版与中间候选只用于解释研究过程；最终支持声明使用最终源码的完整回归。
旧报告的 hash 不修改，已失败的 HGEMM/HGEMV/BF16 Reduce 精度证据不覆盖。

## 复现与后续工作

先检查 PCI、占用和预约，下面的 mask 仅代表本批机器实测对应关系。

```bash
# 在 LeetCUDA 根目录执行；沿用已冻结的 fixture 和父二进制
HIP_VISIBLE_DEVICES=2,1 scripts/rocm/run.sh python -m leetcuda_rocm.flash_swizzle_check run \
  --fixtures artifacts/rocm/flash_tiled/fixtures.pt \
  --frozen-baseline artifacts/rocm/flash_swizzle/baseline/tiling.json \
  --output artifacts/rocm/flash_swizzle/vector_hip.json --performance

HIP_VISIBLE_DEVICES=2,1 scripts/rocm/run.sh python -m leetcuda_rocm.flash_swizzle_check run \
  --device 1 --fixtures artifacts/rocm/flash_tiled/fixtures.pt \
  --frozen-baseline artifacts/rocm/flash_swizzle/baseline/tiling.json \
  --output artifacts/rocm/flash_swizzle/hip_device1.json

python3 -m leetcuda_rocm.flash_swizzle_acceptance \
  --hip artifacts/rocm/flash_swizzle/vector_hip.json \
  --output artifacts/rocm/flash_swizzle/acceptance.json

HIP_VISIBLE_DEVICES=2 scripts/rocm/run.sh python -m leetcuda_rocm.smoke \
  --output docs/rocm/qualification_20261105_flash_swizzle.json
```

大 fixture、输出 tensor、二进制和完整 ISA 留在两端专用 artifacts；JSON、源代码、
结论和复现命令进 Git。复现冻结性能比较需要原二进制及其锁定环境；新机器应先按
冻结提交重建并记录新的二进制 hash，不能将不同二进制的测量冒充本次重放。

后续按 B2 推进 `flash_attn_mma_stages_split_q_shared_kv_acc_f32_rr` 与
`flash_attn_mma_stages_split_q_shared_qkv_acc_f32_rr`。验收重点为真正减少寄存器
生命周期、VGPR/scratch 变化，以及保留 shared 路径的 FP32 输出状态。
本批 swizzle 的性能收敛继续独立跟踪；其他剩余项见[整体计划](leetcuda_rocm10_migration_plan.md)
和[逐接口清单](leetcuda_rocm10_operator_inventory.json)。
