# 第二批原始接口适配与 HGEMM 精度研究

日期：2026-11-03。环境继续锁定为 ROCm 10.1.0 / R9700 / gfx1201。

## 本轮可用结果

新增 4 个通过验证的原始计算接口：

| 模块 | 新增原始接口 | 说明 |
| --- | --- | --- |
| Merge Attention States | `merge_attn_states_cuda` | 合并两段 attention 输出和 log-sum-exp，支持 FP32/FP16/BF16 |
| SGEMM | `sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf` | 同步 FP32 双缓冲 |
| SGEMM | `sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf` | 同步 FP32 双缓冲 |
| SGEMM | `sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf` | 同步 FP32 双缓冲 |

基础资格验证现为 **22 个模块、139 个原始接口、387 项检查通过**。
按原有 229 个 Python 扩展计算接口统计，覆盖率从 **58.95% 提升到 60.70%**。
已有两个独立 AMD WMMA GEMM / FlashAttention 接口另计。

加载器支持一个扩展使用多个源文件，全部源文件内容参与缓存 hash；新增报告记录
每个源文件的 hash。原有 21 模块重新执行回归，历史报告保留不覆盖。
完整结果见 [GPU0 资格报告](qualification_20261103.json)。

## Merge Attention States

原始名称保留 `merge_attn_states_cuda`，通过 ROCm 加载器构建为 HIP。
输入／输出状态形状为 `[tokens, heads, head_size]`，LSE 为 FP32 `[heads, tokens]`。
状态须连续、16 字节对齐，head_size 须为 pack size 的倍数：FP32 为 4，FP16/BF16 为 8。
允许零 tokens，heads 和 head_size 必须为正；支持可选 `output_lse=None`。
输出不可与输入共享存储，接口拒绝梯度输入、混合设备和不匹配的 shape/dtype/layout。

两个状态的权重由各自 LSE 决定，使用稳定的指数归一化。沿用原接口约定：
**正负无穷 LSE 均表示空状态**。修复了两段都为空时的 NaN：返回零输出、负无穷 LSE；
空状态中的 NaN 输出不会污染有效状态。LSE 中的 NaN 不在支持契约内。

测试包括：三种 dtype、非整齐 token 数、D=8/24/128、可选 LSE、非默认 stream、
单边／双边空状态、极端有限 LSE、非法输入拒绝，以及从独立 FP64 attention
计算两段状态后合并，与完整 attention 对照。另用 FP64 `logaddexp` 作独立合并参考。

物理 GPU2／逻辑 GPU1 补测还验证输入设备与调用者当前设备不同时的 device guard、
当前设备恢复和混合设备拒绝。合计 **87 项检查通过**，见
[第二张卡报告](continuation_device1_20261103.json)。

```python
import torch
from leetcuda_rocm import load

lib = load("merge_attn_states")
prefix = torch.randn(7, 3, 128, device="cuda", dtype=torch.bfloat16)
suffix = torch.randn_like(prefix)
prefix_lse = torch.randn(3, 7, device="cuda", dtype=torch.float32)
suffix_lse = torch.randn_like(prefix_lse)
output = torch.empty_like(prefix)
output_lse = torch.empty_like(prefix_lse)
lib.merge_attn_states_cuda(
    output, output_lse, prefix, prefix_lse, suffix, suffix_lse
)
```

## 同步 GEMM 的修复与限制

新增 SGEMM 的源文件同时包含同步和异步实现。ROCm 构建只启用同步部分，
带 `cp.async` PTX 的异步 kernel 与 host 实例化通过条件编译排除，没有将异步指令改为空操作。
修复同步双缓冲尾轮固定读取 buffer 1 的问题，改为依据 K tile 数选择最终缓冲区。

新增三个 SGEMM 变体通过 K tile 数为 1/2/3/5/8 的检查，以及已有两组矩形输入检查。
FP32 参考为 FP64 matmul 舍入到 FP32，`atol=rtol=2e-4`。
这些仍是低层分块接口，要求连续输入及有效 tile 对齐；测试覆盖 M=128/256、N=256，
K 为 16 的倍数，不提供任意尾部尺寸保证。

本轮以正确性和精度为验收目标，没有为这些原始变体测量性能或声明加速。
CUDA 路径未执行回归；修改涉及共享源文件中的数值边界与缓冲区索引。

## HGEMM：已经能运行，但当前精度门限未通过

11 个普通同步 HGEMM 候选已成功编译执行，同样修复了双缓冲尾轮索引。
它们保留原始 **FP16 累加** 语义；与已有 `leetcuda_rocm.gemm` 的 FP32 累加不同。

研究结果：

- 三个随机种子、同一组 256×256×256 输入对照全部 11 个变体，共 33 项。
- **33/33 项逐位匹配独立模拟的逐步 FP16 FMA 累加结果**，表明这些用例的差异
  来自预期的 FP16 累加舍入，而不是 HIP 翻译破坏了计算。
- 相对 FP64 matmul 的最大绝对误差约 **0.01880**；11 个变体各自都至少有一个
  用例未通过 `atol=0.01, rtol=0.01` 的当前精度门限。
- 另有 **55/55 项**单 tile、奇偶 tile 数等边界检查通过。

因此，这 11 个变体标为“已运行、精度未验收”，**不进入默认支持清单或默认加载器**。
这不是 GPU 无法实现 HGEMM 的证据，也不是对所有 FP16 累加用途的否定。
需要当前已验收的 FP16/BF16 矩阵乘时，使用已有的 FP32 累加 AMD WMMA `gemm` 接口。
是否另设 FP16 累加精度契约应单独评估，不能将此批结果记为已通过当前门限。

完整用例、误差、舍入模拟结果及源码 hash 见
[HGEMM 研究报告](hgemm_research_20261103.json)。研究命令在存在精度失败时返回非零，属于预期结果。

## 复现及剩余方向

在 LeetCUDA 根目录：

```bash
# 当前全部基础接口；输出 22 模块 / 139 接口 / 387 项检查
./scripts/rocm/run.sh python -m leetcuda_rocm.smoke

# 第二张卡，仅检查本轮两个模块
HIP_VISIBLE_DEVICES=0,2 ./scripts/rocm/run.sh python -m leetcuda_rocm.smoke \
  --device 1 --modules merge_attn_states sgemm \
  --output artifacts/rocm/continuation/device1.json

# 未验收的 HGEMM 精度研究；预期因精度门限返回非零
./scripts/rocm/run.sh python -m leetcuda_rocm.research_hgemm
```

当前 229 个原始接口中仍有 **76 个尚未移植、14 个精度未验收**。
76 个待移植接口包括 66 个矩阵指令／CuTe 实现、6 个异步搬运实现、4 个 cuBLAS 绑定；
14 个精度未验收接口包括本轮 11 个 FP16 累加 HGEMM 和此前 3 个 BF16 累加 Reduce。

功能层面的后续重点仍是 FP8/FP4 GEMM、FFPA、Attention 的 GQA/backward/dropout/任意 mask，
以及独立 C++ benchmark 接入。其中 FP8/FP4、FFPA 位于独立 C++ 实现范围，不计入 229 的分母。
现有 FlashAttention 前向已经可用，长序列因果模式还需要性能调优。
