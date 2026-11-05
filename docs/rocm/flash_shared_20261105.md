# 第三轮：两个原始 FlashAttention FP32 累加接口

本文保留第三轮记录；后续 tiling-QK/QKV 已完成，当前 170/229。见[第四轮报告](flash_tiled_20261105.md)。

2026-11-05。原始接口从 **166/229（72.49%）** 增至 **168/229（73.36%）**，
增加 2 个接口、0.87 个百分点；剩余 **45 个待迁移、16 个精度未验收**。
原 FlashAttention 模块从 0/29 增至 **2/29（6.90%）**，有支持接口的模块达到 24/25。

优先选择 FlashAttention，是因为此前它占 47 个待迁移接口中的 29 个。
本批先验证两种 LDS 复用方式，为后续 tiling、swizzle 和 split-KV 提供基础。
独立 `leetcuda_rocm.flash_attention` API 的历史结果没有直接计入原接口覆盖率。

## 接口与实现

两个名称均带 `flash_attn_mma_stages_split_q_` 前缀：

| 后缀 | 实际 AMD 实现 |
| --- | --- |
| `shared_kv_acc_f32` | Q 常驻独立 LDS 区域；K/V 共用可复用的 LDS 缓冲 |
| `shared_qkv_acc_f32` | Q 先经过 LDS 进入各 wave 的寄存器，原 Q 的 LDS 区域随后交给 K/V |

```python
from leetcuda_rocm import load

lib = load("flash_attn")
# q/k/v/out：同设备、连续、相同 [B,H,N,D] shape 的 FP16 tensor。
lib.flash_attn_mma_stages_split_q_shared_kv_acc_f32(q, k, v, out, 2)
lib.flash_attn_mma_stages_split_q_shared_qkv_acc_f32(q, k, v, out, 1)
```

实现位于 [flash_attention_shared.cu](../../kernels/rocm/flash_attention_shared.cu)。
QK 和 PV 均使用原生 gfx1201 FP16-input / FP32-accumulator WMMA。
softmax 的 max/sum 和输出累加状态均为 FP32，P 在 PV 之前转换为 FP16；
最终输出为 FP16。与这两个原 CUDA 入口的精度类型一致，未新增累加精度提升项。
不同矩阵指令、归约分组和 exp 实现仍可能产生舍入差异，不承诺逐位相同。

保留原有 stage 分派：`stages <= 1` 使用 stage 1，`stages > 1` 使用 stage 2。
stage 1 在 QK 完成后复用 K 的 LDS 区域装载 V；stage 2 使用两个 K/V 槽，
通过 `global_load_b128` 向 VGPR 预取 V，在 QK 后显式等待并写入 LDS。
也预取下一块 K；不将这种替代调度称为 NVIDIA `cp.async` 指令复现。

支持 D=32/64/96/128 的两种 stage，以及 D=256 的 stage 1；与原接口保持一致。
N 尾块、空 B/H/N 和未对齐的连续视图额外通过测试。拒绝错误 dtype/shape、
非连续布局、输出别名、梯度输入、超范围索引及不支持的 D/stage 组合。
本接口仅支持相同序列长度、无 mask 的前向；不包含 causal、GQA、dropout 或 backward。

AMD 调度差异明确保留：

- 大多数配置保持原 128 行 query tile；D=256 改为 64 行以满足 64 KiB LDS 上限。
- `shared_kv` 的 key tile 为 D<128 时 64 行，否则 32 行；`shared_qkv` 为 64 行。
- FP16 概率 P 经 LDS 交换以衔接两次 AMD WMMA；不会分配全局 N×N 分数矩阵。
- 两种 LDS 生命周期分别编译，绝非两个名字绑定同一通用 Attention 函数。

## 正确性与硬件证据

| 验证项 | 结果 |
| --- | ---: |
| 同一冻结 CPU 输入、独立 CPU FP64 attention 参考 | AMD / 原 CUDA 各 146/146 通过 |
| CUDA 与 AMD 输出在 atol=rtol=0.003 内一致 | 146/146；其中 12 项逐位一致 |
| AMD 尾块、空维度、两组 seed、未对齐视图、stage 映射、非法输入及并发 stream | 388/388 通过 |
| 全库单卡回归 | 24 模块、168 接口、927 项通过 |
| 性能形状的数值及非空 graph replay 检查 | 两端各 22/22 通过 |
| 对应 D/stage/LDS 策略的编译特化 | 18/18 包含 FP32 WMMA |

8 个 stage 2 特化均在汇编中出现 global load → WMMA → load wait 的静态顺序；
这不是逐周期重叠效率测量。16 个特化没有 scratch 指令；`shared_qkv` 的
D=128/stage2 和 D=256/stage1 分别出现 53、156 条静态 scratch 指令，性能问题仍待解决。
新增入口没有追加第二张 AMD 卡及设备切换验证，因为当时其他设备被占用或预约。

原 CUDA 来自固定上游快照 `3dbd618e703943e6761db05f29dd66cb7de94d13`。
专项构建只将 launch 适配到当前 stream，用填 NaN 后 graph replay 的结果检查
排除空图计时；源码、适配后源码、二进制、输入和输出均保存 SHA256。
这批 CUDA 146 项全部通过，未使用性能或输入筛选隐藏数值失败。

证据：[AMD 专项与逐特化 ISA](flash_shared_20261105/hip.json)、
[CUDA 专项](flash_shared_20261105/cuda.json)、
[逐项跨平台对照](flash_shared_20261105/comparison.json)、
[完整资格报告](qualification_20261105_flash_shared.json)。

全部 **168 个接口**再使用一份共同 fixture 完成两端回归：CUDA 和 AMD 均为
**340/340** 通过独立参考，**340/340** 在接口容差内一致，225 项逐位一致。
两端所记录的所有源码 hash 与提交文件一致；此基础回归不代替上面的专项边界测试。
证据：[CUDA](flash_shared_20261105/qualified168_cuda.json)、
[AMD](flash_shared_20261105/qualified168_hip.json)、
[全接口对照](flash_shared_20261105/qualified168_comparison.json)。

## 性能观察与下一批

使用相同 CPU fixture，固定 graph 每次 8 个调用、交替测量自定义实现和同机 Torch
SDPA、每项 3 组样本。采样的 SDPA 派发为 `aten::_scaled_dot_product_flash_attention`。
下面是当前实现的测量值，不是达到库性能的声明：

| B×H×N×D | 配置 | 自定义 / μs | 同机 SDPA / μs |
| --- | --- | ---: | ---: |
| 1×4×256×64 | shared-QKV，stage 2 | 27.81 | 7.77 |
| 1×4×1024×64 | shared-QKV，stage 2 | 95.62 | 26.40 |
| 1×4×1024×128 | shared-KV，stage 2 | 172.72 | 28.53 |
| 1×4×256×256 | shared-KV，stage 1 | 95.54 | 24.44 |

该卡驻留其他模型，计时属于共享环境观察，不能视为独占设备的性能验收。
不根据跨厂商硬件耗时宣称同规格速度，也不据此承诺稳定加速。

下一批优先 `tiling_qk_acc_f32` 和 `tiling_qkv_acc_f32`：通过逐块加载减少 LDS 和
寄存器占用，并覆盖更大的 head dimension。原 CUDA 的这两个入口在 D>=256
还会将输出累加状态存成 FP16，必须单独验证该精度约定，不能仅按函数名认定全程 FP32。
之后再推进 swizzle；split-KV 需要额外验证分区 softmax 状态合并。

## 设备身份核对与历史标签更正

本批最终测试使用 **PCI `0000:e3:00.0`（rocm-smi GPU7，HIP mask `4`）**。
检查时物理 GPU6 已被预约、GPU5 有驻留 guard，GPU0～4/7 有其他模型；
选择当时无活跃计算且显存余量足够的 GPU7，没有停止或修改其他任务。

本轮发现 HIP 序号不能直接当成物理 GPU 编号。例如当前 HIP `0` 对应 PCI
`0000:63:00.0`（禁止计算的物理 GPU3），HIP `6` 对应 `0000:83:00.0`。
项目环境验证现已在计算前查询所有可见设备的 PCI，并拒绝物理 GPU3；
实测 HIP mask `0` 在执行 workload 前被拒绝。共享启动器对 HIP 序号 `3` 的原限制保留。
只读 PCI 查询不执行 GPU kernel。

历史报告若仅含 `HIP_VISIBLE_DEVICES` 而未记录 PCI，其中的“物理 GPUx”标签
是未核实推定，不能用来证明物理卡身份或性能测量独占性。历史数值与原始 JSON 保留，
不以当前映射反推历史时刻；新的完整 168 接口回归记录明确 PCI 身份。
证据：[设备映射和禁用卡拒绝记录](flash_shared_20261105/device_identity.json)、
[当时占用快照](flash_shared_20261105/occupancy.json)。

## 复现

再次运行前重新核对 PCI、占用和预约；以下 mask 仅表示本次实际选择。

```bash
HIP_VISIBLE_DEVICES=4 scripts/rocm/run.sh python -m leetcuda_rocm.flash_shared_check fixtures \
  --fixtures artifacts/rocm/flash_shared/fixtures.pt
HIP_VISIBLE_DEVICES=4 scripts/rocm/run.sh python -m leetcuda_rocm.flash_shared_check run \
  --fixtures artifacts/rocm/flash_shared/fixtures.pt --output artifacts/rocm/flash_shared/hip.json --performance

# 将同一 fixture 传到 5090 专用目录后，使用固定上游快照 original/：
scripts/cuda/run.sh python -m leetcuda_rocm.flash_shared_check run --backend cuda \
  --manifest docs/rocm/cross_5090_20261105/pending56_manifest.json --source-root original \
  --fixtures artifacts/flash_shared/fixtures.pt --output artifacts/flash_shared/cuda.json --performance

# 传回 CUDA JSON 与同名 .pt，再比较：
HIP_VISIBLE_DEVICES=4 scripts/rocm/run.sh python -m leetcuda_rocm.flash_shared_check compare \
  --cuda artifacts/rocm/flash_shared/cuda_remote/cuda.json \
  --hip artifacts/rocm/flash_shared/hip.json --output artifacts/rocm/flash_shared/comparison.json
```
