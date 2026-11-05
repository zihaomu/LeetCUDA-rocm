# 补充范围持续验收记录

2026-11-06。原 Python 接口229/229的分母保持不变。本轮32个补充GPU定义均已完成验收，来源审计102项、CUDA归档审计223项及全库15163项通过。B1三个入口固定性能门槛均已通过。见[最终验收总表](leetcuda_rocm10_final_acceptance.md)、[统一15163项验收](unified_acceptance_20261106.md)和[B1最终性能](flash_swizzle_final_20261106.md)。

| 范围 | 已完成证据 | 限制/正在处理 |
| --- | --- | --- |
| interview FlashAttention 8定义 | 两张AMD各2578项；整理后再2578项；独立C++646；CUDA修复参考共同输出1280/1280 | 原CUDA54失败保留；169条原资源/安全不可运行记录；AMD14/16生产特化spill；性能仅SDPA的0.030–0.278倍 |
| CuTe向量1 + profiling12 + 转置3定义 | 双AMD各254；原CUDA224；168份输出全部逐位一致；17性能配置；整理后254通过 | 原CUDA转置尾块barrier不安全，AMD明确扩展；profiling只有trace/SQ_WAVES通过 |
| CuTeDSL GEMM 3定义 | 双AMD各158；整理后158；原CUDA72；66份共同输出逐位一致；各12性能配置；三个原Python脚本AMD入口通过 | 使用独立HIP后端；不声明NVIDIA DSL编译器已移植；旧固定stream性能数据无效并保留 |
| swizzle GEMM 5定义 | 双AMD各403独立检查；整理后403；CUDA FP32参考198；198份比较通过；各20性能配置；11特化零spill | 显式FP32累加升级；原CUDA66/198精度失败保留 |
| 统一独立C++ | 9组2907/2907；不链接Torch；31章源码映射已建立 | 最终格式整理及固定加载器后全量重建通过 |
| B1 Q/QK/QKV | 三轮约1.178/1.217/1.114倍，全部过固定门槛，42特化无spill | Q/QK采用线性V padding8和LDS阶段复用；失败候选及不稳定首测保留 |

## 已定位问题与实际方案

- FA basic CUDA：stage3/4源地址`t+1`与目标槽`t+stage-1`不匹配；短序列warmup另有越界风险。仅修改预取索引、warmup边界和初次等待的独立参考220项通过。
- FA persist-D CUDA：全mask行在`-inf - -inf`后生成NaN；仅将该tile概率清零、保持running state的独立参考108项通过。原版48+6失败及修复版分别保存。
- FA资源：部分原CUDA深stage需要超过5090的101376字节共享内存，按case列为不可运行。AMD使用较小分块与CPU FP64和可运行共同配置验收。
- AMD FA：条件向量状态写入造成LLVM私有alloca，无条件写回后D32归零；全展开、调度栅栏、禁用SLP未消除大D spill，候选与二进制保留。
- CuTe向量：双方编译器将原表达式融合成两次FP16 FMA，初版逐操作舍入CPU参考不符合实际算术。冻结误报后按独立FP64两次FMA舍入重建参考，仍保持逐位门限。
- 转置：AMD采用全线程barrier、有界载入/存储，分别保留plain、padding33、XOR LDS地址；原CUDA对照只执行完整tile。
- CuTeDSL：保留naive的最终FP16舍入、128×128×8 SIMT及padding4、三级无padding流水线。CUDA计时函数必须每次取得current stream；新增3条跨capture stream的NaN投毒检查后重测，先前43–109倍无效性能不得引用。
- swizzle GEMM：原FP16累加长K误差不能通过门限，明确升级FP32；普通/XOR、小/大tile、LDS/寄存器输出机制分别保留。
- HIPify：不支持带换行的限定kernel名，格式整理后一次编译失败已保存；binding改用局部using声明，最终复验通过。

## Profiling与依赖

`HIP_VISIBLE_DEVICES=6 scripts/rocm/profile_examples.sh artifacts/rocm/profile_examples`
实测12个kernel各20次，240个目标dispatch，trace记录时间/寄存器/LDS/scratch。
SQ_WAVES两次采样均逐kernel匹配理论波数；VALU/WAVE32_VALU/SALU/SMEM计数均返回0，标记当前环境未验收，不能据此声称没有指令。真实指令类型以冻结ISA佐证。
`profile_examples.sh 输出目录 counters`复现已验证的wave采集。Nsight工具本身不移植。

5090 CuTeDSL4.4.2依赖位于项目私有`.cache/cute-dsl-4.4.2`；版本与wheel hash保存在`cute_dsl_install.json`。
依照[NVIDIA官方安装说明](https://docs.nvidia.com/cutlass/4.4.2/media/docs/pythonDSL/quick_start.html)，target安装需额外将`nvidia_cutlass_dsl/python_packages`加入PYTHONPATH；首轮导入失败保留。

证据根目录：`artifacts/rocm/continuation_all/`。正式清单、来源审计和全库回归已更新，所有计划算子及必须验收项完成；性能限制继续保留。
