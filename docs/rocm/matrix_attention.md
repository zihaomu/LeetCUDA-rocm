# AMD 矩阵单元 GEMM 与 FlashAttention 前向

新增两个独立 AMD 实现：`leetcuda_rocm.gemm` 和
`leetcuda_rocm.flash_attention`。它们保留 LeetCUDA 分块 GEMM 和 split-Q /
在线 softmax 的算法思路，使用锁定环境中的 Triton 编译到 gfx1201。
这两个接口实际执行自定义 GPU kernel，不调用 PyTorch matmul / SDPA 作为实现。
原有 CUDA WMMA/PTX/CUTLASS 变体的接口没有被替换。

## 运行

在仓库根目录：

```bash
# 完整数值、指令检查及性能对比
./scripts/rocm/run.sh python -m leetcuda_rocm.matrix_attention_check

# 只检查正确性和 AMD 矩阵指令
./scripts/rocm/run.sh python -m leetcuda_rocm.matrix_attention_check --correctness-only

# 复现小范围分块配置比较（先确认 GPU0 空闲）
./scripts/rocm/run.sh python scripts/rocm/probe_matrix_tiles.py
```

结果写入 `artifacts/rocm/matrix_attention/report.json`，完整生成汇编位于同目录
`isa/`。随代码保存的结果快照是 [matrix_attention.json](matrix_attention.json)。
环境严格锁定为 ROCm 10.1.0 / HIP 7.16 / Torch 2.14 / Triton 3.8 / R9700
`gfx1201`，详细身份及源代码 SHA256 记录在报告中。

```python
import torch
from leetcuda_rocm import gemm, flash_attention

# 由 scripts/rocm/run.sh 启动 Python
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
w = torch.randn_like(x)
y = gemm(x, w)

q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = flash_attention(q, k, v, causal=True)
```

## 支持范围与实现

| 项目 | 支持范围 |
| --- | --- |
| GEMM | `A[M,K] @ B[K,N]`，FP16/BF16 输入，FP32 累加，输入 dtype 输出 |
| GEMM 布局 | 连续或有 stride 的二维输入，包括转置；支持 M/N/K 尾部 |
| Attention | `Q[B,H,NQ,D]`、`K/V[B,H,NK,D]`，FP16/BF16 |
| Head dimension | 16、32、64、128 |
| Mask | 无 mask，或含对角线的 causal；causal 要求 NQ=NK |
| 布局与输出 | 支持 strided 输入、当前 stream、可选预分配连续 `out` |
| 当前不支持 | dropout、任意 attention mask、GQA、backward、空维度、跨设备输入 |

接口拒绝带 `requires_grad` 的输入和与输入共享存储的输出，不会静默给出错误梯度。
索引偏移和输出元素数限制在有符号 32 位范围内；没有对非有限输入提供特殊语义。
这些限制由公开接口检查；以 `_` 开头的 launch 函数仅供内部验证使用。

GEMM 小矩阵使用 64×64×32 tile / 4 warps，`min(M,N)>=512` 时使用
128×128×64 tile / 8 warps。Attention 使用 64×64 的 query/key tile，保留每行
最大值、归一化分母和输出累加值；在遍历 K/V 时进行在线重标定，不将 NQ×NK
分数矩阵写到全局内存。QK 和 PV 都使用矩阵单元，softmax 状态为 FP32，PV 的
概率输入转换为 FP16/BF16。因而与全 FP64 attention 不要求逐位一致。

资格脚本检查所有采样配置的汇编，必须包含对应的指令：

```text
v_wmma_f32_16x16x16_f16
v_wmma_f32_16x16x16_bf16
```

报告还记录汇编/HSACO hash、寄存器数、spill 和 shared memory 用量。
这使“使用 AMD 矩阵单元”成为实测检查项，而不是只依据 `tl.dot` 推断。

## 验证方法

- 正确性参考：GEMM 使用独立 FP64 matmul；Attention 显式构造 FP64 分数矩阵、
  softmax 和 PV。参考仅用于测试。
- 覆盖 FP16/BF16、小尺寸与 tile 尾部、转置输入、大 tile、非默认 stream、
  causal/非 causal、交叉注意力、零 scale、较大 logits，以及非法输入拒绝。
- 断言对照舍入到输出 dtype 的参考值，同时记录相对原始 FP64 参考的绝对误差和
  相对 L2 误差；具体 `atol`/`rtol` 保存在每个用例中。
- 性能形状再与对应的 PyTorch 输出核对；GEMM 对照 `torch.mm(out=)`，
  Attention 对照默认派发的 `scaled_dot_product_attention`。
- 单独用 CPU profiler 记录 SDPA 实际派发算子，profiling 不在计时区间。

性能通过 HIP graph 回放计时，预热并排除 JIT/库初始化，5 轮交替测量实现和
基线，每轮约 40ms，取中位数。测试重复使用相同输入，属于热缓存 GPU 执行时间，
不含 Python 调用成本；未锁定 GPU 时钟。SDPA 没有 `out` 参数，其输出分配在
graph capture 中完成。不要将接近 1 的小幅差异解读为稳定的端到端加速。

## 实测结果（2026-11-02）

物理 GPU0 上 **68/68 项正确性与接口检查通过**：32 项 GEMM、26 项 Attention、
10 项非法输入拒绝。另完成 24 组性能形状的输出对照和计时，28 个采样编译配置
全部包含预期的 FP16/BF16 WMMA 指令，均未发生寄存器 spill。
所有 Attention 性能基线的实际派发均为
`aten::_scaled_dot_product_flash_attention`。

独立 FP64 参考检查中，GEMM 的最大相对 L2 误差分别为 FP16 `3.47e-4`、
BF16 `1.67e-3`，Attention 分别为 `2.89e-4`、`2.43e-3`。
这里是相对未舍入 FP64 参考的误差，包含输出 dtype 的舍入误差。

以下时间单位为 **微秒**，越小越好；倍率为“PyTorch 时间 / 本实现时间”，
大于 1 表示本实现较快。GEMM 尺寸按 M×N×K，Attention 按 B×H×N×D。

| 算子 / 尺寸 | dtype | 本实现 | PyTorch | 倍率 |
| --- | --- | ---: | ---: | ---: |
| GEMM 2048×2048×2048 | FP16 | 158.91 | 155.36 | 0.98× |
| GEMM 2048×2048×2048 | BF16 | 159.80 | 157.00 | 0.98× |
| Attention 1×8×1024×64，非 causal | FP16 | 28.46 | 29.17 | 1.02× |
| Attention 1×8×1024×64，非 causal | BF16 | 31.32 | 29.91 | 0.96× |
| Attention 1×8×256×64，causal | FP16 | 5.72 | 11.53 | 2.01× |
| Attention 1×8×256×64，causal | BF16 | 7.91 | 13.50 | 1.71× |
| Attention 1×8×2048×64，causal | FP16 | 90.62 | 74.55 | 0.82× |
| Attention 1×8×2048×64，causal | BF16 | 108.75 | 75.13 | 0.69× |
| Attention 1×8×1024×128，causal | FP16 | 47.51 | 39.57 | 0.83× |
| Attention 1×8×1024×128，causal | BF16 | 51.86 | 41.43 | 0.80× |

2048³ GEMM 分别达到约 **108.1 / 107.5 TFLOPS**。全部 GEMM 用例倍率为
0.90–1.07×；非 causal Attention 为 0.96–1.19×；causal 为 0.69–2.01×。
短序列的优势不能外推到长序列：BF16、N=2048 的 causal 延迟比基线高约 45%。
完整 24 组数据、每轮时间、数值误差及 ISA hash 见
[GPU0 结果快照](matrix_attention.json)。

另在空闲的物理 GPU2（`HIP_VISIBLE_DEVICES=0,2`，逻辑设备 1）复测，
**68/68 项检查通过**，见 [第二张卡结果](matrix_attention_device1.json)。
复现命令：

```bash
HIP_VISIBLE_DEVICES=0,2 ./scripts/rocm/run.sh \
  python -m leetcuda_rocm.matrix_attention_check --device 1 --correctness-only \
  --output artifacts/rocm/matrix_attention/device1/report.json
```

这些结果验证了 ROCm 10 上重写矩阵单元 GEMM 与融合 Attention 的可行性，
也说明后续难点主要在具体变体的重写和性能调度。长序列 causal 的 tile、
对角线分阶段计算与 occupancy 值得继续测量；当前没有 profiler 证据将差距归因于
某个单一瓶颈。本次只覆盖 R9700/gfx1201 的上述前向范围，不代表原库全部
CUDA 变体、其他 AMD 架构或训练路径已完成适配。

## 设计参考

实现依据分块矩阵乘法与 FlashAttention 在线 softmax 算法；相关说明见
[Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)、
[Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。
AMD WMMA 的架构背景见
[AMD WMMA layout](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.amd.AMDWMMALayout.html)。
