# ROCm 10 全范围统一功能验收

2026-11-06。统一入口已完整执行，**15组算子/构建验证共15,163项通过**，另8步生成CPU夹具。原25模块229接口仍为1262项；测试项数量不作为算子数量。
本报告对应固定二进制加载器后的最终全量复验，关闭统一构建与功能回归工作。固定B1三入口性能门槛也已[全部通过](flash_swizzle_final_20261106.md)。

## 一条命令执行

```bash
# 先核对实际PCI、占用和预约；例示mask对应本机物理GPU0和GPU2。
HIP_VISIBLE_DEVICES=2,1 scripts/rocm/qualify_all.sh \
  --output artifacts/rocm/full_acceptance

# 只构建和运行独立C++验证；可执行文件不链接PyTorch。
HIP_VISIBLE_DEVICES=2 scripts/rocm/build_all_bench.sh \
  --output artifacts/rocm/all_cpp
```

启动器验证固定ROCm10.1.0镜像、HIP7.16、Torch2.14/Triton3.8、gfx1201及允许的PCI。
完整入口要求两张允许的GPU做混用拒绝/guard恢复；物理GPU3禁止计算。新机器须重新资格验证，不能将gfx1201结论直接外推其他架构。
每组单独保存JSON、日志、source/fixture/二进制hash；失败立即退出，完整成功才写总manifest为passed。
`--skip-cpp`只执行Python组，其范围会写入manifest；不代表已执行独立C++。

## 本次结果

| 验证组 | 通过项数 |
| --- | ---: |
| 原25模块、229接口 | 1262 |
| 独立矩阵单元GEMM/FA | 68 |
| FP8 E4M3 / E5M2 | 756 / 756 |
| FP4完整链及契约 | 1541 |
| 原Triton示例 | 195 |
| GQA/mask/dropout/backward | 377 |
| FFPA三路径 | 2755 |
| interview独立FA | 2578 |
| vector/profile/transpose | 254 |
| CuTeDSL三路径 | 158 |
| swizzle GEMM五定义 | 403 |
| 未绑定Attention / GEMM | 553 / 600 |
| 独立C++九组 | 2907 |
| **合计** | **15163** |

复用已有输出的precision/state诊断没有再次计入总数。报告[summary.json](unified_acceptance_20261106/full/summary.json)列出计算口径，[manifest.json](unified_acceptance_20261106/full/manifest.json)记录每一步命令和结果。

C++九组为base/SGEMV、SGEMM、HGEMM、FP8 E4M3、FP8 E5M2、FP4、FFPA、独立FA、其他GPU示例；对应370/234/477/93/93/165/729/646/100项。FP8/FP4专项还有更广的Python契约/量化测试，不把C++数量误当成全部覆盖。
book/tests的31个章节已逐文件映射到这些已验收实现：[映射JSON](supplemental_examples_20261106/book_mapping.json)。ch01是设备能力/roofline，ch20–23是主机布局或静态检查；ch22的device static-check函数没有运行时launch，不新增GPU算子。原book CUDA构建命令保留，AMD使用上述统一入口。

## 5090对照与完整范围

新一轮同精度CUDA参考和AMD均执行229接口、462组共同输入；所有参考检查与共同输出门限通过。CUDA使用明确标注的`precision_reference`树，原低精度树历史460/462及扩大专项失败保持原结论。
[AMD](unified_acceptance_20261106/full/qualified229_final_hip.json)、[5090同精度参考](unified_acceptance_20261106/full/qualified229_final_cuda.json)、[逐模块比较](unified_acceptance_20261106/full/qualified229_final_comparison.json)。

| 独立统计范围 | 功能覆盖 |
| --- | ---: |
| 原公开计算接口 | 229/229，100% |
| interview独立GPU定义 | 53/53，100% |
| 真实但未绑定入口 | 4/4，100% |
| CuTe/CuTeDSL/swizzle/profiling GPU定义 | 24/24，100% |
| 原Triton示例 | 4/4，100% |

范围间有算法/源码复用，不能相加作为去重算子数。原229中60项明确升级累加精度、3项升级TF32输入精度；相关原精度失败保留。独立公开计算API9个，另2个权重量化辅助接口。

## 已验证替代与实际限制

CUDA TMA/warpgroup使用明确的AMD载入、序号同步和WMMA实现；NVFP4使用精确FP16展开及WMMA，未声称原生FP4指令。CuTeDSL使用对应HIP算法后端，Nsight使用rocprofv3流程。5090无法执行Hopper WGMMA，该部分采用独立CPU及显式标注的GPU数学参考，不声称运行原WGMMA。

仅有限测试覆盖的功能通过，不表示所有形状/架构已验证。FP8/FP4、FFPA WS、训练Attention D256及独立FA仍有性能差距；独立FA14/16生产特化有spill。profiling四类指令计数器返回0而未通过资格，trace和SQ_WAVES已验证。这些限制逐项保留在专项报告中。
M7中明确量化的B1三轮目标已在独立报告验收；本报告的功能数字不替代性能结果。

可在不运行GPU的情况下检查归档数量、报告SHA、当前源码与固定二进制记录：`python3 scripts/rocm/audit_final_reports.py`。该审计不会替代真实GPU验收。
