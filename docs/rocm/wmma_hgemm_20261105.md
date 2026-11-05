# 第二轮：9 个原始 WMMA HGEMM 接口迁移

2026-11-05。新增 **9 个原始接口**，总数由 **157/229（68.56%）** 增至
**166/229（72.49%）**，增加 3.93 个百分点。剩余 **47 个待迁移、16 个精度未验收**。
HGEMM 模块由 2/36 增至 11/36；全部仓库仍未完成迁移。

这批接口采用 **FP16 输入/输出、FP32 WMMA 累加**。原 NVIDIA 实现使用 FP16
累加，因此这是一项明确标注的精度提升，不能声称逐位复现原 FP16 累加。
包括此前矩阵 HGEMV 在内，当前 166 个合格接口中有 **10 个**明确记录累加精度变化。

## 新增接口与实际实现

下表名称均带 `hgemm_wmma_` 前缀；可通过 `leetcuda_rocm.load("hgemm")` 调用。
每项均有对应的编译配置与实际分块，未将所有名称指向一个通用 GEMM。

| 名称后缀 | block 输出 M×N | 每 warp 输出 | 加载和存储策略 |
| --- | ---: | ---: | --- |
| `m16n16k16_naive` | 16×16 | 16×16 | 单 wave32，直接从 global 构造 WMMA fragment |
| `m16n16k16_mma4x2` | 64×32 | 16×16 | 8 个 wave32，单缓冲静态 LDS |
| `m16n16k16_mma4x2_warp2x4` | 128×128 | 32×64 | 8 个 wave32，每 wave 2×4 个 AMD WMMA tile |
| `m16n16k16_mma4x2_warp2x4_dbuf_async` | 128×128 | 32×64 | VGPR 预取、显式 load wait、双 LDS 缓冲 |
| `m32n8k16_mma2x4_warp2x4_dbuf_async` | 128×128 | 64×32 | 保留原 warp 输出区域，用 16×16 AMD atom 重组原 32×8 atom |
| `m16n16k16_mma4x2_warp2x4_stages` | 128×128 | 32×64 | 静态 LDS ring，stage 2～5，3D block swizzle |
| `m16n16k16_mma4x2_warp2x4_stages_dsmem` | 128×128 | 32×64 | 动态 LDS ring，stage 2～5，3D block swizzle |
| `m16n16k16_mma4x2_warp4x4_stages_dsmem` | 256×128 | 64×64 | 动态 LDS ring，stage 2～5，3D block swizzle |
| `m16n16k16_mma4x4_warp4x4_stages_dsmem` | 256×256 | 64×64 | 16 个 wave32，动态 LDS ring，stage 2～4；stage 4 用满 64 KiB |

`stages` 决定实际 LDS ring 深度和预取距离，`swizzle` / `swizzle_stride` 决定
3D grid 的 N 分区；不忽略这些参数。静态 ring 的 stage 2 与固定双缓冲自然复用同一
特化，20 个被测调用配置对应 **19 个不同的 GPU kernel 特化**。

AMD 没有使用 NVIDIA `cp.async`。对齐的块使用 `global_load_b128` 将数据预取到
VGPR，完成当前 tile 计算后通过 `s_wait_loadcnt` 等待并写入 LDS。尾块和未对齐
连续视图使用带边界检查的加载。B 在 LDS 中按 AMD fragment 的 K 向量布局存放，
没有沿用 NVIDIA 的 padding 数量。

调用示例：

```python
from leetcuda_rocm import load

lib = load("hgemm")
# a[M,K], b[K,N], out[M,N]：同卡连续 FP16 tensor，无 autograd，无输出别名。
lib.hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(a, b, out)
lib.hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(
    a, b, out, 3, True, 256
)
```

支持 M/N/K 尾块及空维度，K=0 输出零。非法 stage、开启 swizzle 时非正 stride、
不匹配 shape/dtype、非连续输入、输出别名和梯度输入被拒绝；不会静默选择另一个 stage。
原 CUDA 的部分未支持参数会回退 stage 2，AMD 的显式报错约定在这里单独说明。

## 正确性、指令与并发证据

本轮 ROCm 10.1 / gfx1201 使用物理 **GPU6**。GPU0～5、7 存在其他任务，预约记录
占用 GPU5；检查后选择空闲 GPU6，未使用 GPU3，也未抢占其他任务。
这批没有新增第二张 AMD 卡的设备切换验证，不能套用上一批其他算子的双卡结果。

| 验证项 | 本轮结果 |
| --- | ---: |
| 新增接口与 CUDA 共用的 CPU fixture，独立 CPU FP64 参考 | AMD 200/200 通过 |
| 两个 seed、尾块、空维度、K=8193、抵消、动态范围、未对齐连续视图、stage/swizzle、非法输入、并发 stream | 1265/1265 通过 |
| 三种规模的同机 Torch 配对性能测试，含数值门限 | 60/60 通过数值检查，记录实际性能差距 |
| 清空输出后执行 captured graph 的结果检查 | 两端各 60/60 通过 |
| GPU kernel 特化指令检查 | 19/19 有 FP32 WMMA，均无 scratch spill |
| 完整原始接口回归 | 23 模块、166 接口、923 项通过 |

完整回归的检查数受可见设备数影响。本轮仅暴露 GPU6；此前 890 项报告暴露两卡，
其中额外的设备切换和混卡拒绝检查不会在单卡环境中运行。所有报告保留实际设备身份。

19 个特化中，16 个多缓冲特化在反汇编中存在 `global_load_b128 → WMMA → load wait`
的指令顺序；这证明预取指令未被简单移到计算之后，不代表进行了逐周期重叠效率测量。
全部特化合计 194 条 WMMA 静态指令、124 条 128 位 global load；无 scratch load/store。

证据：[AMD 专项报告](wmma_hgemm_20261105.json)、
[逐特化 ISA](wmma_hgemm_20261105/isa.json)、
[完整资格报告](qualification_20261105_wmma.json)。

## 全部 166 个接口跨平台回归

使用同一份冻结 CPU fixture，在 RTX 5090 和 R9700 上分别运行 23 个模块，
CUDA 使用本分支修复后的源码。两端 **336/336** 通过独立数值参考，
**336/336** 输出在对应接口容差内一致，最终源码复验 **222/336** 逐位一致；无缺失或执行错误。
该基础回归与下面的长 K WMMA 专项是两套测试，基础回归通过不抵消专项精度失败。

证据：[CUDA](wmma_hgemm_20261105/qualified166_cuda.json)、
[ROCm](wmma_hgemm_20261105/qualified166_hip.json)、
[逐项对照](wmma_hgemm_20261105/qualified166_comparison.json)。

## 5090 对照定位的两个不同问题

同一份 10 组 CPU 输入展开为 200 个函数/配置用例，逐张量验证 SHA256。
原源码取自上游固定提交 `3dbd618e703943e6761db05f29dd66cb7de94d13`，
工具链和 RTX 5090 环境沿用 [上一轮锁定配置](cross_5090_20261105.md)。

1. **原版双缓冲收尾错误。** 两个 `...dbuf_async` 在 K=80（5 个 K16 tile）时仍读取
   `s_a[1]` / `s_b[1]`，应读 `(NUM_K_TILES - 1) & 1`。两组 seed 共 4 个失败用例，
   绝对误差约 1.9～2.0。已在共享 CUDA 源码修复，并在 5090 复验，4/4 转为通过。
2. **长 K 的 FP16 累加误差。** K=4096 的 40 个用例在原版和修复版 CUDA 上均未通过
   `atol=rtol=0.01`，AMD FP32 累加全部通过。它与缓冲区选择错误是不同问题。

| 200 个专项用例 | 原版 CUDA | 修复末缓冲后的 CUDA | AMD FP32 累加 |
| --- | ---: | ---: | ---: |
| 对独立 FP64 参考通过 | 156 | 160 | 200 |
| 与 AMD 在接口容差内一致 | 156 | 160 | — |
| 与 AMD 逐位一致 | 0 | 0 | — |

CUDA 失败结果保留，未删除长 K 输入或放宽门限。清单中的 CUDA 验收字段也使用这批
扩展输入，因此“全部选定 CUDA 用例通过”的接口数从上一轮 210 降至 201；这是增加了
9 个接口的长 K 覆盖，并非 AMD 新增算子回归失败。229 个原始接口仍都有 CUDA 运行证据。

另一个测试工具问题是原始 kernel 固定使用默认 stream，导致在非默认 stream 捕获
CUDA Graph 时得到空图。新实验构建仅将 launch 改为当前 stream，记录修改后源码 hash；
通过“填 NaN 后 replay 得到原输出”验证后才发布计时。首轮空图时间不用于性能结论。

证据：[原版 CUDA](wmma_hgemm_20261105/cuda_original.json)、
[修复版 CUDA](wmma_hgemm_20261105/cuda_fixed.json)、
[原版对照](wmma_hgemm_20261105/comparison_original.json)、
[修复后逐项对照](wmma_hgemm_20261105/comparison_fixed.json)。

## 性能仍有差距

固定大小 graph、每 graph 8 次调用、交替测量自定义与同机 Torch，3 组样本。
两端输入与输出 dtype 相同，但内部累加精度不同，硬件与框架版本也不同。
1024³/2048³ 的原 CUDA FP16 累加未通过本轮数值门限，不能仅凭其耗时宣称公平的精度等价速度比。

下面列出本次被测 AMD 配置中各规模的最低耗时；没有自动改变用户选择的入口或 stage。

| M=N=K | AMD 配置 | kernel / ms | 同机 Torch / ms | 现状 |
| --- | --- | ---: | ---: | --- |
| 256 | `mma4x2`，64×32 block | 0.00881 | 0.00487 | 约慢 1.81 倍 |
| 1024 | `mma4x2_warp4x4_stages_dsmem`，stage 5 | 0.05980 | 0.02462 | 约慢 2.43 倍 |
| 2048 | 同上，stage 5 | 0.30583 | 0.14809 | 约慢 2.07 倍 |

功能资格已通过，性能尚未达到成熟库。后续优化方向为 B 的 LDS 访问布局、提高 BK、
降低每 tile 的同步成本；不能在优化时悄悄改变原入口的几何与 stage 约定。

## 复现

```bash
HIP_VISIBLE_DEVICES=6 scripts/rocm/run.sh python -m leetcuda_rocm.wmma_check fixtures \
  --fixtures artifacts/rocm/wmma/fixtures.pt
HIP_VISIBLE_DEVICES=6 scripts/rocm/run.sh python -m leetcuda_rocm.wmma_check run \
  --fixtures artifacts/rocm/wmma/fixtures.pt --output artifacts/rocm/wmma/hip_final.json --performance
python3 -m leetcuda_rocm.wmma_isa_check \
  --report artifacts/rocm/wmma/hip_final.json --output artifacts/rocm/wmma/isa_summary.json

# 在 5090 专用目录，原始源码快照与旧 pending_manifest.json 均保留：
scripts/cuda/run.sh python -m leetcuda_rocm.wmma_check run --backend cuda \
  --manifest docs/rocm/cross_5090_20261105/pending56_manifest.json \
  --source-root ported --fixtures artifacts/wmma/fixtures.pt \
  --output artifacts/wmma/cuda_fixed.json --performance

# 传回 CUDA JSON 与同名 .pt 后：
HIP_VISIBLE_DEVICES=6 scripts/rocm/run.sh python -m leetcuda_rocm.wmma_compare \
  --cuda artifacts/rocm/wmma/cuda_remote/cuda_fixed.json \
  --hip artifacts/rocm/wmma/hip_final.json --output artifacts/rocm/wmma/comparison_fixed.json
```

上述 GPU6 是本批实际选择；再次运行前应重新检查占用和预约，不把该设备写死为长期空闲。
