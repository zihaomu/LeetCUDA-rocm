# ROCm 10 补充范围与剩余验收清单

2026-11-06。原 Python 扩展接口 **229/229** 已完成功能验收；下面内容单独统计，不加入该分母。
计数表示源码定义/路径覆盖，不表示原 CUDA ABI 兼容或工程工作量比例。性能目标独立判定。

## interview 独立 C++

| 头文件 | 原 GPU kernel 定义数 | 功能 | 5090共享输入对照 |
| --- | ---: | --- | --- |
| base.cuh | 19 | 完成 | 完成 |
| sgemv.cuh | 3 | 完成 | 完成 |
| sgemm.cuh | 4 | 完成 | 完成 |
| hgemm.cuh | 5 | 待完成 | 待完成 |
| flash_attn.cuh | 8 | 待完成 | 待完成 |
| ffpa_attn.cuh | 3 | 完成 | 完成；原CUDA同步缺陷及显式修复分别保留 |
| FP8头文件 | 5 | 完成，E4M3/E5M2 | 完成 |
| FP4头文件 | 6 | 完成 | 完成；1560项共同输出全部逐位一致 |
| 合计 | **53** | **40/53（75.47%）** | **40/53（75.47%）** |

base/SGEMV：[双卡和5090各370项、跨平台370项](interview_base_20261106.md)。
FFPA：[三种Split-D实现](ffpa_20261106.md)。FP8：[两种格式完整链路](fp8_e5m2_20261106.md)。
FP4：[完整链跨平台验收](fp4_cross_20261106.md)。逐头文件数据见[JSON清单](leetcuda_rocm10_supplemental_scope.json)。
这些头文件也供book/tests复用；后续统一AMD构建入口显式映射测试范围，不声称原CUDA命令直接适用于HIP。

## 其他必须项

| 范围 | 实际入口/文件 | 状态与方案 |
| --- | --- | --- |
| 未绑定HGEMM WGMMA | `hgemm_wgmma_m64n128k16_stages_tma_ws_tn`、`hgemm_wgmma_m64n128k16_f16acc_stages_tma_ws_tn` | 待完成；AMD WMMA与真实生产者/消费者替换Hopper指令，分别记录原精度与升级精度；5090不能执行的专有指令记录真实编译/运行限制 |
| 未绑定HGEMM TN swizzle | `hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_swizzle` | 待完成；保留TN布局、stage及swizzle分工 |
| 未绑定tiling-QK Attention | `flash_attn_mma_stages_split_q_tiling_qk_acc_f32_rr` | 待完成；D32/64/96/128/256/512/1024及stage1/2专项验收 |
| CuTe向量加 | `kernels/cutlass/cute/vector_add.cu` | 待完成；保留半精度多元素线程布局 |
| CuTeDSL SGEMM | `sgemm.py`、`mma_atom.py`、`pipelining.py` | 待完成；分别实现标量、SIMT矩阵分块、三级异步流水线；原例不是TF32矩阵指令，首例FP16输出舍入需明确记录 |
| swizzle示例 | `hgemm_mma_swizzle.cu`、`mma_simple_swizzle.cu`、`mat_trans_swizzle.cu` | 待完成；分别验证plain/swizzle坐标映射、矩阵运算与转置 |
| Nsight演示 | `elementwise.cu`、`relu.cu` | 待完成；移植GPU操作并提供实际可运行的AMD profiling流程，Nsight工具本身不移植 |
| 统一构建与整体回归 | Python/C++/示例、环境与证据入口 | 待所有范围接入后执行 |
| B1固定性能门槛 | Q/QK；QKV已达标 | Q/QK仍待优化；3轮×18配置、几何均值≥1.05、单项回退≤10%、稳定性≤5%，保持原父实现及失败证据 |
| 其他性能差距 | FP8/FP4、FFPA WS、训练Attention D256、B2/B4 spill | 继续研究并复验；不能用功能通过替代性能达标 |

四个历史SGEMM stage2/stage2_offset/stage3/stage3_offset名称只有声明，没有实现，不计为待移植的已存在算子。
CPU布局绘图、book排版和空transformer目录不属于GPU算子。

每次新增结果更新此清单及[实时进度](leetcuda_rocm10_live_progress.md)。整体迁移尚未完成。

SGEMM补充：[四路径双卡/5090各234项及跨平台234项通过](interview_sgemm_20261106.md)。
