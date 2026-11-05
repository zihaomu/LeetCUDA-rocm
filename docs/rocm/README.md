# LeetCUDA ROCm 10 适配

此入口将原仓库的基础 CUDA C++ 算子编译为 HIP 扩展；执行的仍是原始
LeetCUDA 算法，不会把自定义 kernel 替换成 PyTorch 实现。PyTorch 仅作为
测试参考。当前目标是 **ROCm 10.1.0 / Radeon AI PRO R9700 / gfx1201**。
尚未完成整个仓库的 NVIDIA 专用 kernel 迁移。

2026-11-05 第七轮 B3：**186/229（81.22%）**，新增11个MMA/CuTe HGEMM，
明确采用FP32累加、保留FP16 I/O。双卡各2947项、全库1048项与376组全接口对照通过。
原CUDA的64项长K精度失败和40项stage5资源错误单独保留；AMD36个特化无scratch。
当前27待迁移、16精度未验收，继续B4。见[B3报告](mma_hgemm_20261105.md)。

2026-11-05 第六轮 B2：**175/229（76.42%）**，新增两个shared rr Attention入口，
原Attention达到9/29。两卡各660项专项、191组原CUDA/AMD对照、全库971项与
全接口354组对照全部通过。FP32输出状态保持不变；大D寄存器压力降低，性能仍需优化。
当前38待迁移、16精度未验收，正在继续B3。主卡PCI a3（物理GPU5/HIP7），复验卡PCI43（物理GPU2/HIP1）。
见[B2结果](flash_rr_20261105.md)、[资格报告](flash_rr_20261105/qualification175.json)、
[实时进度](leetcuda_rocm10_live_progress.md)。

2026-11-05 第五轮历史实测：**24 个模块、173 个原始接口、937 项检查通过**，
覆盖 **173/229=75.55%**。按计划新增 Q/QK/QKV 三个 FP32 swizzle FlashAttention，
原 FlashAttention 达到 7/29；两张 AMD 卡各 1329 项专项、372 组原 CUDA/AMD 对照通过。
全接口 350/350 组共享输入通过两端参考及容差检查。42 个特化全部使用 WMMA、
LDS≤48 KiB、无 scratch spill；性能 **1/3 达标**，QKV 相对冻结父实现约 1.117×，
Q/QK 功能支持但性能待优化。当前 **40 待迁移、16 精度未验收**。
见[本轮实现与验收](flash_swizzle_20261105.md)、
[最新资格报告](qualification_20261105_flash_swizzle.json)及[下一批安排](leetcuda_rocm10_next_batch_plan.md)。
主卡为 PCI `0000:03:00.0`（物理 GPU0 / HIP mask 2），复验卡为 `0000:43:00.0`（物理 GPU2 / HIP mask 1）。

2026-11-05 第四轮历史实测：**24 个模块、170 个原始接口、931 项检查通过**，
覆盖 **170/229=74.24%**。新增 tiling-QK / tiling-QKV 两个原 FlashAttention 入口，
支持 D 到 1024，保留大 D 的 FP16 输出累加状态。专项 **854 项**通过，
原 CUDA 与 AMD 的 **248 组**共享输入全部通过独立参考及跨平台容差检查。
全接口 344/344 组共享输入在 CUDA/AMD 两端通过参考及跨平台容差检查。
28 个特化使用 WMMA、无 scratch spill，性能多数仍落后同机 SDPA。
见 [本轮实现与限制](flash_tiled_20261105.md)及
[历史资格报告](qualification_20261105_flash_tiled.json)。该批剩余 **43 待迁移、16 精度未验收**。
最终使用 PCI `0000:c3:00.0`（物理 GPU6 / HIP mask 5），重新测量前核对为空闲且未预约。

2026-11-05 第三轮历史实测：**24 个模块、168 个原始接口、927 项检查通过**，
覆盖 **168/229=73.36%**。新增两个原 FlashAttention FP32 累加入口，分别实现
shared-KV 和 shared-QKV 的 LDS 复用及 stage 1/2 调度。专项 534 项通过，
原 CUDA 与 AMD 的 146 组共享输入全部通过参考和跨平台容差检查。
见 [本轮实现、性能限制与后续计划](flash_shared_20261105.md)和
[历史资格报告](qualification_20261105_flash_shared.json)。该批剩余 45 个待迁移、16 个精度未验收。

第三轮按 PCI 确认使用物理 GPU7 / HIP mask 4，并补充物理 GPU3 的 PCI 拒绝校验。
HIP 序号与物理 GPU 序号不相同；历史报告若未记录 PCI，其物理卡标签视为未核实推定。
该轮该卡驻留其他模型，性能数据属于共享环境观察，尚未达到成熟库水平。

2026-11-05 第二轮历史实测：**23 个模块、166 个原始接口、923 项检查通过**，
覆盖 **166/229=72.49%**。本轮新增 9 个 WMMA HGEMM 接口，包括不同分块、
双缓冲、静态／动态 LDS 多 stage 路径；明确采用 FP16 输入/输出、FP32 累加。
专项 1465 项通过，166 个接口的 336 组共享输入在 5090 与 AMD 上全部通过，
两端输出均在接口容差内一致。详见 [本轮实现与性能](wmma_hgemm_20261105.md)和
[历史资格报告](qualification_20261105_wmma.json)。该批尚有 47 个待迁移及 16 个精度未验收接口。
该批记录 HIP mask 6，未记录 PCI；单卡检查数不能与此前双卡报告直接比较。

此前矩阵 HGEMV 批次达到 157/229、890 项；矩阵 HGEMV 同样明确提升为 FP32 WMMA
累加，两张 AMD 卡分别完成 66、63 项扩展检查。保留
[上一轮 5090 对照和剩余方案](cross_5090_20261105.md)及
[历史资格报告](qualification_20261105.json)。

2026-11-03 历史批次：**23 个模块、156 个原始接口、888 项检查通过**（物理 GPU0），
见 [历史资格报告](qualification_20261103_async.json)。在此前布局转置与 BLAS 批次的
153 个接口基础上，新增 3 个异步 SGEMM 入口，覆盖率为 156/229=68.12%。

- [异步 SGEMM](async_sgemm_20261103.md)：三个原始入口使用 AMD VGPR 预取和双 LDS
  缓冲，两张卡各 213 项检查、18 组性能比较通过；已验证实际加载／计算重叠，无 scratch spill。

- [布局转置](layouts_20261103.md)：25/25 接口；第二张卡 324 项通过、33 组性能对比。
- [BLAS 后端](blas_20261103.md)：hipBLAS/rocBLAS；170 项扩展检查、9 组性能对比，
  第二张卡的 SGEMM/HGEMM 217 项通过；该批新增 NN/TN 两个 HGEMM BLAS 计算接口。
- [前批研究](continuation_20261103.md)：Merge Attention States 与同步 SGEMM 已通过；
  11 个同步 FP16 累加 HGEMM 的 FP64 精度门限仍未通过，保持研究状态。
- [HGEMV CuTe 替代研究](hgemv_research_20261103.md)：3 个候选已运行，长 K 精度未通过；
  该历史批次未在默认模块发布；矩阵 HGEMV 后续已用 FP32 累加通过，另两个仍精度未验收。
- [FP8 完整链路](fp8_pipeline_20261103.md)：原生 HIP E4M3 动态量化、四种 scale 模式、
  普通与 producer/consumer 调度、离线权重；Python 与专用 C++ 共用实现。功能通过，
  性能仍待优化；原 C++ FP8 不在 229 的分母中。
- [FP4 完整链路](fp4_pipeline_20261103.md)：保留 NVFP4 E2M1/E4M3 打包及两级量化，
  精确展开为 FP16 后执行 WMMA；四种模式、两种 tile、两种调度和离线权重均验证。
  使用 FP16 矩阵指令，性能仍需优化；同样单独统计，不增加原始接口的通过数量。

2026-11-02 历史实测：**21 个模块、135 个函数变体、316 项检查通过**（物理 GPU0）。
另有 [48 项非默认设备检查](device1.json) 在物理 GPU2 / 逻辑 GPU1 通过，覆盖
histogram、reduce、dot product 的设备分配与非默认 stream。3 个 BF16 自身
累加变体明确排除，详见下方限制。CUDA 路径本机未执行回归测试。

新增：[AMD 矩阵单元 GEMM 和 FlashAttention 前向](matrix_attention.md)，提供独立
`gemm` / `flash_attention` 接口、WMMA 指令检查和性能对比。第一批 21 模块报告
保留为历史验证记录；这两个独立 AMD 实现单独验证，不计入原始接口的通过数。

## 启动和验证

从仓库根目录运行：

```bash
# 先核对实际 PCI、占用和预约；5 仅为本轮选择的 HIP 序号，不保证长期空闲
export HIP_VISIBLE_DEVICES=5
# 首先验证共享环境，再验证项目环境并运行数值测试
./scripts/rocm/run.sh

# 仅编译选定模块
./scripts/rocm/run.sh python -m leetcuda_rocm.build elementwise softmax sgemm

# 只验证选中的模块
./scripts/rocm/run.sh python -m leetcuda_rocm.smoke --modules elementwise softmax sgemm

# 同时记录 HIP event 耗时（包含 Python 调度，部分原接口还有分配/CPU 同步）
./scripts/rocm/run.sh python -m leetcuda_rocm.smoke --bench

# 查看项目实际运行环境
./scripts/rocm/run.sh python -m leetcuda_rocm.environment
```

本工作区默认调用 `/home/zihaomu/zihaomu_workspace/rocm10-launch.sh`；可通过
`ROCM10_LAUNCHER` 指定相同锁定环境的启动器。依赖已存在于锁定容器中，无需
升级 Torch 或安装 CUDA toolkit。选卡前检查 PCI、占用和预约；物理 GPU3
（PCI 0000:63:00.0）禁止计算。当前 HIP mask 0 会映射到该卡，因此会被项目环境检查拒绝。

- 项目锁：[`environment.lock.json`](../../scripts/rocm/environment.lock.json)。
- ROCm release 为 **10.1.0**；其 HIP component 是 **7.16.26385**，不能用
  HIP 版本是否以 `10` 开头判断 ROCm release。
- Python **3.13.13**、Torch **2.14.0+rocm10.1.0**、Triton 的完整 package
  version 和镜像 SHA256 均严格验证。
- 编译使用 C++20。JIT 缓存位于 `.cache/rocm10/<环境指纹>/torch/`，源代码
  内容另外参与模块名和构建目录指纹；生成的 HIP 文件不写回 `kernels/`。
- 运行结果写入 `artifacts/rocm/qualification.json`。失败返回非零退出码；
  报告记录 GPU、镜像、版本、源文件 hash、每个函数的误差与容差。
- 使用新的 Python/Torch/ROCm/GPU 配置时必须另建 profile 并重新验证。

## 当前迁移范围

| 模块 | ROCm 入口包含的实现 |
| --- | --- |
| elementwise | FP32 / FP16 标量与向量化加法 |
| relu, gelu, sigmoid, swish, hardswish, hardshrink, elu | 原标量和 packed 激活实现 |
| softmax | per-token、safe、online，FP32 / FP16 |
| layer_norm, rms_norm | FP32 / FP16、不同累加精度、packed |
| reduce | FP32 / FP16 / BF16 输入 FP32 累加 / FP8 / INT8 reduction |
| dot_product | FP32 / FP16 dot product |
| rope, embedding, histogram | 原生 kernel |
| sgemv, hgemv | k16 / k32 / k128；新增 `hgemv_tensor_core_cute`，明确采用 FP32 WMMA 累加 |
| sgemm | 10 个同步 FP32 kernel、3 个 AMD VGPR 预取变体，以及 hipBLAS FP32 GEMM |
| hgemm | hipBLAS FP16 NN/TN（COMPUTE_16F），以及 9 个 AMD WMMA 变体（FP32 累加）；共 11/36 个接口 |
| flash_attn | shared 2、tiling 2、FP32 Q/QK/QKV swizzle 3，共 7/29；FP16 I/O，tiling/swizzle 在 D>=256 保留 FP16 输出状态；swizzle 性能 1/3 达标 |
| mat_transpose | 25 个接口：14 个原实现及 11 个显式 HIP 布局变体 |
| nms | IoU ballot + 单 block 顺序消除 |
| merge_attn_states | FP32/FP16/BF16 状态合并、可选 LSE、空状态和输入契约检查 |

最新通过情况以 [`qualification_20261105_flash_swizzle.json`](qualification_20261105_flash_swizzle.json) 为准，
[`qualification.json`](qualification.json) 保留 2026-11-02 的历史结果。两种规模、
非默认 stream、独立数值参考用于基础资格检查；这不是所有 shape 的支持承诺。
低精度累加变体使用报告中单独记录的容差，不等同于 FP32 累加精度。

BF16 自身累加在随机输入测试中超过精度门限，因此保守地排除该类 3 个变体
（`bf16_bf16`、`bf16x2_bf16`、`bf16x8_pack_bf16`）；原 CUDA 教学路径保留。
例如 `bf16_bf16` 在 `[16,1024]` 输入上出现约 0.395 的绝对误差（1.86%），
没有通过放宽测试容差将它标为合格。BF16 输入请使用名称含 `_f32` 的累加版本。

以下入口仍为 NVIDIA 专用，**未迁移**：

- `kernels/interview/build.sh` 的整套 C++ benchmark；FP8/FP4 头文件已有 AMD 分支和
  `scripts/rocm/build_bench.sh {fp8,fp4} --run --performance` 专用入口（选择其中一个名称）。
- `kernels/hgemm` 中剩余的 MMA/PTX、CuTe、标量异步路径，`ws-hgemm` 与
  `flash-attn` 剩余 25 个绑定变体，以及未绑定 WGMMA/TMA 和 FFPA；9 个 WMMA HGEMM 已迁移。
  FP8 E4M3 与 NVFP4 完整链路已单独接入。
- 原目录下的 Python 大规模性能脚本、TensorRT、Nsight 示例。

四个原 Triton 示例已接入 ROCm：vector-add、fused-softmax、layer-norm（含反向）、
merge-attn-states。两张 AMD 卡和 5090 各 195 项检查通过，108 组跨平台输出/梯度
对照通过；性能仍待优化。见 [全仓对照范围与未完成项](cross_5090_20261105.md)。

后续应按 kernel 单独替换为 AMD WMMA/MFMA、HIP/rocBLAS 或 Triton 实现，保留
算法标识和独立测试；不能用宏把 NVIDIA 指令静默变成空操作。当前未声称达到
原 NVIDIA benchmark 的性能。

完整的 [算子支持清单](leetcuda_rocm10_operator_support.md)、
[229 项 JSON 清单](leetcuda_rocm10_operator_inventory.json) 和
[迁移计划](leetcuda_rocm10_migration_plan.md) 随本分支保存。

## Python 低层接口

由 `run.sh` 启动的 Python 可以使用：

```python
import torch
from leetcuda_rocm import load

lib = load("elementwise")
a = torch.randn(8, 256, device="cuda")
b = torch.randn_like(a)
c = torch.empty_like(a)
lib.elementwise_add_f32x4(a, b, c)
torch.testing.assert_close(c, a + b)
```

ROCm PyTorch 仍使用 `torch.cuda` / `device="cuda"`。这些是原教程的**低层
接口**，没有 autograd、通用 shape dispatcher 或完整输入检查。调用方必须保证：

- tensor 位于当前逻辑 GPU，同一设备、正确 dtype、连续存储且满足向量对齐；
  多 GPU 使用前用 `torch.cuda.device(...)` 选择输入设备。
- 遵守原函数的 tile/pack/block 限制。不能给无尾部保护的向量化 kernel 传
  任意长度。参考 `leetcuda_rocm/smoke.py` 中的已验证形状。
- 不传空 tensor、负 embedding/histogram 下标、越界 embedding 下标或超大
  NMS 输入。当前资格报告不覆盖这些输入。

适配层明确将原 kernel launch 放到 PyTorch 当前 stream，内部新分配放到
当前逻辑设备。当前保留原来的 32-lane 算法假设，**不支持 wave64 GPU**；
加载器拒绝未验证的架构，而不是直接在 MI 系列上沿用这些假设。

同一环境下可用 `--device` 选择逻辑 GPU，例如物理 GPU0/2 均可用时：

```bash
HIP_VISIBLE_DEVICES=0,2 ./scripts/rocm/run.sh python -m leetcuda_rocm.smoke \
  --device 1 --modules histogram reduce dot_product --output artifacts/rocm/device1.json
```

## 设计依据

AMD 的 [ROCm 10.1 HIP 迁移指南](https://rocm.docs.amd.com/projects/HIP/en/docs-10.1.0/how-to/hip_porting_guide.html)
要求人工处理架构相关代码和 warp 假设；PyTorch 的
[HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) 保留 CUDA API
命名。本实现使用 PyTorch 自带 HIPify，并补齐其尚未映射的 FP8 header 和
转换符号。HIP sync intrinsic 的 mask 使用 64-bit 类型；gfx1201 仍按 32-lane
执行。ReLU half2 使用两个标量 half max，FP16 GELU 使用 float tanh 避免指数
商溢出。测试还修正了原 BF16/FP16 packed reduction 在 FP32 累加路径中
提前舍入的问题，以及矩形转置的 block 索引和 SGEMM 中的残留补丁字符。
不存在 ROCm 7 回退路径。

11 个 `mat_transpose_cute_*` AMD 替代接口支持 FP32 连续二维输入与转置形状输出，
包括零维长度和尾部；逐位保留 NaN、无穷与正负零。向量化接口要求 16 字节基地址对齐，
拒绝别名、autograd 和混合设备，内部 device guard 保持调用方设备。其余原转置接口
仍遵守原 tile/pack 限制。

BLAS 接口的形状、精度与 TN 打包约定见 [BLAS 文档](blas_20261103.md)。
