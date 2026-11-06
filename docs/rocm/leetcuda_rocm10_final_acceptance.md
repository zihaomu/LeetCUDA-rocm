# LeetCUDA ROCm 10 计划最终验收

2026-11-06。计划中的算子实现、独立构建、跨平台验证和明确量化的性能门槛已完成；M0–M7及B1–B7、U1–U3关闭。本轮最后新增8个interview FlashAttention定义和24个补充GPU定义，最终统一验收15组、**15163项全部通过**。

后续[第二阶段性能优化](performance_phase2_results.md)已完成：对冻结旧AMD实现，独立FA约1.202×、FP8完整在线约1.577×、FFPA WS约1.073×、D256前向+反向约1.087×，均通过固定配置三轮门槛。四个受影响实现已重新专项验收；下述15163项仍指先前完整运行，未冒充本次重新执行。FP4另见[独立研究目标](fp4_research_goal.md)。

| 独立统计范围 | 总数 | 已支持 | 未验收 | 比例 |
| --- | ---: | ---: | ---: | ---: |
| 原公开计算接口 | 229 | 229 | 0 | 100% |
| interview独立GPU定义 | 53 | 53 | 0 | 100% |
| 真实但未绑定计算入口 | 4 | 4 | 0 | 100% |
| CuTe/CuTeDSL/swizzle/profiling示例定义 | 24 | 24 | 0 | 100% |
| 原Triton示例 | 4 | 4 | 0 | 100% |

这些范围存在算法和源码复用，不能相加称为去重算子总数。原229个接口跨25个模块，60个明确升级累加精度、3个明确升级输入精度，原精度失败证据保留。另提供9个独立计算API与2个权重量化辅助接口。

## 最终证据

- **原接口**：1262项全库检查，5090同精度CUDA参考与AMD的462组共同输入全部通过；原CUDA低精度树与显式修复/精度升级参考分别记录。
- **统一构建**：九组不链接PyTorch的HIP C++可执行文件2907项通过，原book/tests全部31章建立源码与验证入口映射。
- **新增FlashAttention**：双AMD各2578项、C++646项；修复参考与AMD共同1280份输出通过，原CUDA54项缺陷单独归档。
- **其他24定义**：向量/profile/转置、CuTeDSL、swizzle双AMD每卡分别254/158/403项，跨平台168/66/198份共同输出全部通过。
- **B1性能**：Q约1.178×、QK约1.217×、QKV约1.114×，三入口均满足三轮固定门槛；42特化零spill、LDS≤48KiB；旧四Attention入口394正确性和174性能记录无超过10%回退。
- **复现审计**：共享加载器改为先在子进程编译，再首次加载按SHA冻结的二进制；最终全库重新执行。来源、夹具、实际加载文件和ISA的哈希可核对。

详细证据：[统一验收](unified_acceptance_20261106.md)、[B1最终性能](flash_swizzle_final_20261106.md)、[新增FA](interview_attention_20261106.md)、[补充24定义](supplemental_examples_20261106.md)、[逐接口库存](leetcuda_rocm10_operator_inventory.json)、[补充范围清单](leetcuda_rocm10_supplemental_scope.md)。

## 仍然存在的限制

功能覆盖完成不表示所有workload都达到成熟ROCm库性能。第二阶段已有上述收益，但独立FA仍有13/16生产特化spill（最大scratch752B）；FFPA WS的LDS占用及训练Attention的dQ/dKV仍有优化空间。第二阶段门槛衡量对旧AMD实现的改进，不代表已追平成熟库。FP4仍采用FP16展开计算，未开始新一轮性能实验。

AMD没有原样执行NVIDIA TMA/WGMMA/CuTe编译器：采用已验证的AMD载入、同步、WMMA及对应算法后端。NVFP4精确展开为FP16后使用WMMA，未声称原生FP4矩阵指令。5090不能执行Hopper WGMMA，该部分明确采用独立数学参考。

profiling的trace和SQ_WAVES已验证；四类指令计数器返回0，未通过资格，指令类型用ISA佐证。Nsight/TensorRT工具本身及原CUDA专用CLI不声明已移植，AMD使用项目提供的验证与profiling入口。仅对已测ROCm10.1.0/gfx1201及报告所列形状、精度、接口契约提供资格结论，其他AMD架构尚未验证。

## 使用与交付

```bash
cd /home/zihaomu/zihaomu_workspace/leetcuda_workspace/LeetCUDA
# 例示本机允许的物理GPU0和GPU2；使用前核对占用和预约。
HIP_VISIBLE_DEVICES=2,1 scripts/rocm/qualify_all.sh --output artifacts/rocm/full_acceptance
```

工作区文档镜像位于`/home/zihaomu/zihaomu_workspace/leetcuda_workspace/doc`。5090实验目录为`/home/zihaomu/zihaomu_workspace/test_lap/leetcuda_rocm10`；原CUDA快照保持独立，修改参考与原始失败均保留。
交付分支：[zihaomu/LeetCUDA-rocm 的 rocm10](https://github.com/zihaomu/LeetCUDA-rocm/tree/rocm10)。完整实验记录见[实时进度](leetcuda_rocm10_live_progress.md)。

初次实现与完整验收证据提交：[`8da0434`](https://github.com/zihaomu/LeetCUDA-rocm/commit/8da0434)。第二阶段在`87edbab`基线上修改四个实现，新证据和源码审计见[第二阶段报告](performance_phase2_results.md)，沿用`rocm10`分支交付。
