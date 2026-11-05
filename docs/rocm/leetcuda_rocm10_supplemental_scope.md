# ROCm 10 补充范围与剩余验收清单

2026-11-06。原 Python 扩展接口 **229/229** 已完成功能验收；下面内容单独统计，不加入该分母。
计数表示源码定义/路径覆盖，不表示原 CUDA ABI 兼容或工程工作量比例。性能目标独立判定。

## interview 独立 C++

| 头文件 | 原 GPU kernel 定义数 | 功能 | 5090共享输入对照 |
| --- | ---: | --- | --- |
| base.cuh | 19 | 完成 | 完成 |
| sgemv.cuh | 3 | 完成 | 完成 |
| sgemm.cuh | 4 | 完成 | 完成 |
| hgemm.cuh | 5 | 完成 | 4条CUDA源码参考；WGMMA为CPU及5090数学参考 |
| flash_attn.cuh | 8 | 完成 | 完成；原版54失败与显式修复参考分开 |
| ffpa_attn.cuh | 3 | 完成 | 完成；原CUDA同步缺陷及显式修复分别保留 |
| FP8头文件 | 5 | 完成，E4M3/E5M2 | 完成 |
| FP4头文件 | 6 | 完成 | 完成；1560项共同输出全部逐位一致 |
| 合计 | **53** | **53/53（100%）** | **52条CUDA源码参考 + 1条数学参考** |

base/SGEMV：[双卡和5090各370项、跨平台370项](interview_base_20261106.md)。
FFPA：[三种Split-D实现](ffpa_20261106.md)。FP8：[两种格式完整链路](fp8_e5m2_20261106.md)。
FP4：[完整链跨平台验收](fp4_cross_20261106.md)。逐头文件数据见[JSON清单](leetcuda_rocm10_supplemental_scope.json)。
这些头文件也供book/tests复用；统一AMD构建入口已显式映射全部31章测试范围，不声称原CUDA命令直接适用于HIP。

## 其他必须项

| 范围 | 实际入口/文件 | 状态与方案 |
| --- | --- | --- |
| 未绑定HGEMM WGMMA | `hgemm_wgmma_m64n128k16_stages_tma_ws_tn`、`hgemm_wgmma_m64n128k16_f16acc_stages_tma_ws_tn` | 完成；真实stage2–5的WMMA生产者/消费者，5090 WGMMA编译拒绝与数学参考分列 |
| 未绑定HGEMM TN swizzle | `hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_swizzle` | 完成；独立BK16/TN XOR布局，stage2–5及swizzle stride均已验收 |
| 未绑定tiling-QK Attention | `flash_attn_mma_stages_split_q_tiling_qk_acc_f32_rr` | 完成；整份原源码与tiling-QKV除名称/注释外一致，兼容入口双卡/5090与状态语义专项通过 |
| CuTe向量加 | `kernels/cutlass/cute/vector_add.cu` | 完成；双AMD及原CUDA、逐位对照，保留多元素线程布局 |
| CuTeDSL SGEMM | `sgemm.py`、`mma_atom.py`、`pipelining.py` | 完成；标量/SIMT/三槽流水线分别验收，CUDA72项与66份逐位对照；首例FP16输出舍入保留 |
| swizzle示例 | `hgemm_mma_swizzle.cu`、`mma_simple_swizzle.cu`、`mat_trans_swizzle.cu` | 完成；5 GEMM+3 transpose定义，双卡/CPU/CUDA、Graph、布局、资源与性能分别验收 |
| Nsight演示 | `elementwise.cu`、`relu.cu` | 完成12 GPU定义及rocprofv3 trace/SQ_WAVES；4类指令计数器未验收；Nsight工具本身不移植 |
| 统一构建与整体回归 | Python/C++/示例、环境与证据入口 | 统一C++9组2907项与31章映射完成；全功能15组15163项最终回归通过 |
| B1固定性能门槛 | Q/QK/QKV全部达标 | 三入口三轮约1.178/1.217/1.114倍；42特化零spill且LDS≤48KiB，原父实现及失败证据保留 |
| 其他性能差距 | FP8/FP4、FFPA WS、训练Attention D256、B2/B4 spill | 差距与原因已记录，后续可继续优化；没有声称达到未设定的统一性能目标 |

四个历史SGEMM stage2/stage2_offset/stage3/stage3_offset名称只有声明，没有实现，不计为待移植的已存在算子。
CPU布局绘图、book排版和空transformer目录不属于GPU算子。

每次新增结果更新此清单及[实时进度](leetcuda_rocm10_live_progress.md)。计划算子与必须验收项已完成，见[最终验收](leetcuda_rocm10_final_acceptance.md)。

SGEMM补充：[四路径双卡/5090各234项及跨平台234项通过](interview_sgemm_20261106.md)。

HGEMM补充：[五路径双卡各477项、显式FP32 CUDA参考及比较各336项](interview_hgemm_20261106.md)。36个AMD编译特化零scratch/spill；原CUDA108项精度失败及WGMMA在5090编译限制均保留。

未绑定HGEMM：[3入口双卡各600项独立检查、跨平台432项](unbound_gemm_20261106.md)。未绑定范围3/4，原229分母不变；20个编译特化零spill，性能差距保留。

未绑定Attention：[等价入口及双卡各553项验收](unbound_attention_20261106.md)。未绑定范围现4/4闭合，原229分母不变；interview剩8条FlashAttention及其他补充范围继续。

最新补充：[独立FA 8定义](interview_attention_20261106.md)、[其他GPU 24定义](supplemental_examples_20261106.md)。interview现53/53=100%，其他24/24=100%，未绑定4/4=100%；分别统计。上文末尾历史进度以本次结果更新，整体M7必须项也已完成。
