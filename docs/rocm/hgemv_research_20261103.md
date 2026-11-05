# HGEMV CuTe 变体研究：2026-11-03

三个候选均已在 ROCm 10.1.0 / R9700 gfx1201 编译和运行，**尚未验收，不在默认模块发布**。
这使原接口清单中的“待适配／未验证”从 62 降为 59，“精度未验收”从 14 增为 17；
通过数仍为 **153/229（66.81%）**。未将研究实现计为成功移植。

实现：`kernels/rocm/hgemv_layouts.cu`；驱动：`leetcuda_rocm/research_hgemv.py`。
原 CUDA/CuTe 源码保留。三个实现分别为每行一 wave 的标量 half FMA、8 元素 packed
half2 FMA，以及每 wave 16 行的 AMD FP16 WMMA。packed 实现显式初始化每个 tile 的
half2 累加器，修正原 CuTe 源码中未初始化局部累加器的问题。

| 原始接口 | FP64 精度检查通过／总数 | 独立 FP16 算术模拟相等／总数 | 当前结论 |
| --- | ---: | ---: | --- |
| hgemv_f16_cute | 26/30 | 30/30 | 标量 FMA 与 XOR reduction 模型数值相等；长 K 精度未过门限 |
| hgemv_f16x8_cute | 18/21 | 21/21 | tile 内 half2、tile 间 half 累加与 XOR reduction 模型数值相等；K=8192 精度未过门限 |
| hgemv_tensor_core_cute | 24/30 | 9/30（每 16 项精确求和后 RNE 假设） | WMMA 实际累加／舍入还需研究，不能宣称已证明算法语义一致 |

共 3 个种子、81 项检查；门限保持 `atol=0.003, rtol=0.003`。
输入为 FP16 正态随机数乘 0.25，M=0/1/3/17/64/65/129，K 覆盖 0/1/8/19/128/256/257/1024/8192；
packed 只运行 K 为 8 的倍数的有效形状。矩阵单元版本在 K=1024 和 K=8192 的三组种子
都未通过该门限，最大绝对误差约 0.1660。

机器码中确认存在 `v_wmma_f16_16x16x16_f16`，不是标量或 PyTorch matmul 替代。
独立模型同时试验了分块 RNE、分块向零舍入、两种逐项 half FMA 顺序；都未完全匹配
矩阵单元版本，因而不能将其误差全部归因于已知的某一种舍入模式。
AMD/Clang 提供该 FP16 WMMA builtin，但内建名称本身不能证明本实现满足数值要求。
参见 [Clang builtin 定义](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html#builtin-amdgcn-wmma-f16-16x16x16-f16-w32-gfx12)
和 [AMD RDNA4 布局说明](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/)。

完整数据、源码与机器码身份见 [研究 JSON](hgemv_research_20261103.json)。
本阶段未发布性能合格声明，也未将容差放宽。后续需要核对矩阵单元的累加顺序／精度，
并区分保持原 FP16 算术的教学实现与另行标识的 FP32 累加实现。

```bash
./scripts/rocm/run.sh python -m leetcuda_rocm.research_hgemv
# 当前预期返回 1：FP64 精度门限未通过，结果仍保存到 artifacts/rocm/hgemv/research.json。
```
