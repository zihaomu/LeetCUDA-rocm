# 未绑定HGEMM三入口验收（ROCm 10）

2026-11-06。三个已存在于上游源码但未纳入原pybind列表的入口现已可用，**单独统计3/4未绑定计算入口**；原25模块229接口分母保持不变。

| 入口 | 原累加状态 | AMD机制与资格 |
| --- | --- | --- |
| `hgemm_wgmma_m64n128k16_stages_tma_ws_tn` | FP32 | 1 producer wave + 4 consumer waves，实际2–5槽环形流水线，FP16 WMMA/FP32累加 |
| `hgemm_wgmma_m64n128k16_f16acc_stages_tma_ws_tn` | FP16 | 明确升级FP32累加，与上行共享同一AMD算法；原FP16状态另有诊断 |
| `hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_swizzle` | FP16 | 明确升级FP32；128×128×16、8 waves、原每K16 XOR布局、2–5槽预取 |

WGMMA以64×128×32子块适应AMD 64KiB LDS限制。保留独立生产者/消费者和acquire/release序列计数；没有模拟不存在的TMA/WGMMA硬件指令。
TN swizzle为新增Kind 5，保持BK=16，区别于interview的BK=64及寄存器ping-pong路径。

## 结果

| 验收 | 结果 |
| --- | ---: |
| AMD物理GPU4 / PCI 0000:83:00.0 | 560数值/Graph + 36边界/设备 + 4流水线审计通过 |
| AMD物理GPU0 / PCI 0000:03:00.0 | 同上全部通过 |
| 原CUDA TN及WGMMA数学参考 | 384/432；TN长K原FP16累加48项失败保留 |
| 显式FP32 CUDA TN及WGMMA数学参考 | 432/432 |
| 共同输入跨平台输出 | 432/432，0项逐位一致 |
| 修改公共HGEMM头的回归 | interview全部477项通过 |
| ISA | 20个编译特化，全部真实WMMA、128位加载；零scratch/spill |

560项含432个共享普通/Graph输出、72个tail/alignment输出、24个stage fallback输出、32个累加状态输出。
16种FP32/FP16状态控制各有普通及Graph检查，已含在560项内，不重复计数。
状态控制精确区分1与1.03125；FP16诊断采用每K16将FP32 WMMA状态舍入FP16，不能等同NVIDIA内部舍入顺序。
四种stage的延迟消费者审计均完成12个K块，实际领先深度分别为2/3/4/5，无超时。
最大VGPR210，最大LDS61540B（stage5），验证槽位循环复用。

CPU FP64独立参考取实际FP16输入，输出比较使用既有HGEMM绝对误差0.01、相对误差0；未放宽门限。
公共fixture为既有262144个FP32数值文件，使用offset0/8192（不是两个独立随机种子），较长输入循环使用；两端逐输入SHA核对。
普通计算及capture后NaN覆盖的Graph replay均检查输出前后canary。涵盖K64/192/512/1024/2048、N2304及swizzle stride256/512/2048。
AMD额外覆盖(65,137,80)、(1,3,17)、(129,257,129)尾部、偏移1个half的非16B对齐输入，以及不同当前设备、流和输入设备。

## CUDA参考边界

5090的sm120实际编译拒绝两种原WGMMA浮点指令，见[FP32探测](unbound_gemm_20261106/wgmma_fp32_probe.log)、[FP16探测](unbound_gemm_20261106/wgmma_fp16_probe.log)。这两个入口使用CPU FP64与cuBLAS FP32累加数学参考，**没有声称5090执行了原WGMMA算法**。
TN以不可变上游`3dbd618e703943e6761db05f29dd66cb7de94d13`编译，原版仅改当前stream。另一份显式参考只将两处MMA累加及其寄存器改为FP32、在原epilogue前转回FP16，搬运、stage和swizzle均保留。
原版K512/1024/2048的48项精度失败继续保留；明确FP32参考全部通过。
首次二进制归档的Python相对路径错误也保留，修复后CUDA重跑；该修复不影响AMD计算，AMD报告使用归档的初版验收脚本作来源核对。

## 性能

两端各24配置（3入口×4 stages×2形状），本机`torch.mm`作独立基线，固定FP32累加选项。
采用32次调用Graph、约40ms自适应采样、5轮交替event测量；每项同时检验结果。
AMD WS两名称的基线/本实现比为0.093–0.150，TN为0.156–0.244，当前仍明显落后成熟库。CUDA WGMMA列是数学参考自身的计时，不能解读为原WGMMA性能。
本轮完成功能和机制验收；不据此宣称M7性能达标。

## 调用与重跑

```python
from leetcuda_rocm import load_unbound
mod = load_unbound("hgemm")
# A[M,K], C[M,N]；B形状[K,N]但存储物理连续BT[N,K]，延续上游契约。
mod.hgemm_wgmma_m64n128k16_stages_tma_ws_tn(a, b, c, 5, True, 512)
```

FP16连续、有限值、无autograd，同设备、C独立存储，正尺寸及32位索引/合法grid。非法dtype、shape、device、stride、alias被拒绝。
stages不在2..5时保留上游fallback（WGMMA为3，TN为2），开启block swizzle必须正stride。当前仅gfx1201；构建应在Graph capture之前完成。

```bash
HIP_VISIBLE_DEVICES=6,2 scripts/rocm/run.sh python -m leetcuda_rocm.unbound_gemm_check run \
  --backend hip --performance \
  --fixtures artifacts/rocm/continuation_all/interview/fixtures.bin \
  --output artifacts/rocm/continuation_all/unbound_gemm/recheck.json
```

镜像、Torch、HIP、PCI、输入/源码/二进制hash随报告冻结，物理GPU3禁止计算。
[主卡](unbound_gemm_20261106/hip.json)、[第二卡](unbound_gemm_20261106/hip_second.json)、[原CUDA](unbound_gemm_20261106/cuda_original.json)、[FP32参考](unbound_gemm_20261106/cuda_fp32.json)、[比较](unbound_gemm_20261106/comparison.json)、[资源](unbound_gemm_20261106/resources.json)、[源码审计](unbound_gemm_20261106/source_audit.json)。

下一个未绑定Attention已确认与原tiling-QKV整份源码除名称/注释外一致，将独立验收兼容入口。其他计划继续执行，整体迁移尚未完成。
