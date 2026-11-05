# LeetCUDA ROCm 10 持续迁移进度

最近更新（UTC）：2026-11-05T15:17:07+00:00。任务状态：**执行中**。

用户要求：完成后续计划中的所有算子任务，完成前持续推进；每次关键实验结束即时追加本文件。仓库镜像与工作区 `doc` 同步。

## 最近正式验收的状态

- 基线提交：`5b7926c7d339356eee6a1531ec9af8e844dc81e7`。
- 原接口：**186/229（81.22%）**；待迁移 **27**，精度未验收 **16**。
- 已完成 B1：三个 swizzle 功能支持；性能 1/3 达标。
- 当前执行：B3验收完成，186/229；推进B4的20个Attention。低精度QK失败已复现，QK FP32/PV FP16研究通过992项，原CUDA审计运行中。

## 固定工作队列

| 工作包 | 数量 | 状态 | 验收重点 |
| --- | ---: | --- | --- |
| B2 shared rr | 2 | 功能完成，性能待优化 | 真实寄存器生命周期变化、VGPR/scratch、FP32 输出状态、双卡和原 CUDA 对照 |
| B3 HGEMM MMA/CuTe | 11 | 功能完成，显式FP32升级／性能待优化 | 原算术模型、tile/TN/stage/swizzle、长 K 与奇数 tile |
| B4 其余 Attention | 20 | 进行中，低精度模型与QK升级实验 | 低精度模型、布局/分工，split-KV、Os2g、CuTe 的独立机制 |
| B5 异步 HGEMM | 3 | 待执行 | 真实异步流水线；区分原精度失败与搬运错误 |
| B6 TF32 SGEMM | 3 | 待执行 | TF32 输入舍入、实际矩阵路径、数学模式 |
| B7 WS-HGEMM | 1 | 待执行 | producer/consumer 分工和同步 |
| 既有精度未验收 | 16 | 待执行 | 保留失败证据，验证原语义或明确记录精度升级；门限不放宽 |
| 性能收敛 | 单独统计 | 待执行 | B1 Q/QK 等未达标路径；功能数量不等于性能达标数量 |
| 整体计划补充范围 | 不计入229 | 待执行 | FFPA、Attention 扩展、FP8/FP4 性能、独立 C++ 等已有计划范围，按各自契约验收 |

## 尚未验收的逐接口清单

| 模块 | 接口 | 正式状态 |
| --- | --- | --- |
| flash_attn | `flash_attn_mma_stages_split_kv` | pending |
| flash_attn | `flash_attn_mma_stages_split_q` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_kv` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_qkv` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_tiling_qk` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_tiling_qkv` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_kv_swizzle_q` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_kv_swizzle_qk` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_kv_swizzle_qkv` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_qkv_swizzle_q` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_qkv_swizzle_qk` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_qkv_swizzle_qkv` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_tiling_qk_swizzle_q` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_tiling_qk_swizzle_qk` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_tiling_qk_swizzle_qkv` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_tiling_qkv_swizzle_q` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_tiling_qkv_swizzle_qk` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_tiling_qkv_swizzle_qkv` | pending |
| flash_attn | `flash_attn_cute` | pending |
| flash_attn | `flash_attn_mma_stages_split_q_shared_qkv_Os2g` | pending |
| hgemm | `hgemm_naive_f16` | numerical_rejected |
| hgemm | `hgemm_sliced_k_f16` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k_f16x4` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k_f16x4_pack` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k_f16x4_bcf` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k_f16x4_pack_bcf` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k_f16x8_pack_bcf` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf` | numerical_rejected |
| hgemm | `hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf` | numerical_rejected |
| hgemm | `hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async` | pending |
| hgemm | `hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async` | pending |
| hgemm | `hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async` | pending |
| hgemv | `hgemv_f16_cute` | numerical_rejected |
| hgemv | `hgemv_f16x8_cute` | numerical_rejected |
| reduce | `block_all_reduce_sum_bf16_bf16` | numerical_rejected |
| reduce | `block_all_reduce_sum_bf16x2_bf16` | numerical_rejected |
| reduce | `block_all_reduce_sum_bf16x8_pack_bf16` | numerical_rejected |
| sgemm | `sgemm_cublas_tf32` | pending |
| sgemm | `sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages` | pending |
| sgemm | `sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem` | pending |
| ws_hgemm | `ws_hgemm_naive_cute` | pending |

## 实时实验记录

- **2026-11-05T14:24:05+00:00**：恢复持续迁移。核对代码干净，当前分支 rocm10；物理 GPU0（PCI 03，HIP 2）和 GPU2（PCI 43，HIP 1）空闲，预约状态 released；物理 GPU3 禁用。5090 两卡空闲。开始 B2 源码/契约审计。

实验期间只增加有最终证据的正式支持数量；编译成功、候选通过或单卡结果会单独记录。

- **2026-11-05T14:25:37+00:00**：B2 契约审计完成：原 CUDA rr 使用较短的 PV 临时寄存器生命周期。AMD 候选在 D>64 时将每 CTA 的输出状态限制为 64 列，真实减少同时存活的 FP32 状态；保持 KV/QKV 不同 LDS 复用，明确记录输出分片导致重复 QK。QKV D=256/stage2 需将 key tile 调整为32以满足64KiB LDS限制。下一步编译、数值和VGPR/scratch实测。

- **2026-11-05T14:28:17+00:00**：B2 候选代码已写入：两个独立 rr 导出、64列 FP32 输出分片、QKV D256/stage2 的合法 LDS 配置。正在首次编译和基础检查；正式支持数仍为173。专项将覆盖全部19个 D×stage 特化、长序列、精确 FP32 状态、canary、双设备、原CUDA共享输入与三轮性能。

- **2026-11-05T14:31:18+00:00**：B2 首次模块检查18/18通过（包含既有7个和新增2个Attention入口）。正式专项增加N=4096、两种种子、全D范围，并对D256/stage2的QKV rr单独覆盖；当前正在冻结CPU共享输入。已将原非rr路径的输出偏移编译期固定为0，避免给旧路径增加运行时开销。

- **2026-11-05T14:34:27+00:00**：B2 主卡专项通过：191项共享输入、414项边界/契约、19项精确FP32状态、36项设备/stream，共660项；69条三轮性能记录已生成。原CUDA同样191项参考+19项状态通过。正在提取VGPR/scratch与同输入跨平台对照，随后第二张卡复验。

- **2026-11-05T14:37:41+00:00**：B2 首轮资源结果：KV D256/stage1 的VGPR 243→154；QKV D128/stage2 的12个VGPR spill已消除，D256/stage1 spill从127降到32。但独立输出CTA重复搬运/QK导致明显性能退化，部分三轮数据波动。保留首轮全部结果，改进为同一CTA内两组warp协作处理输出分片、共同装载Q/K/V，减少重复搬运并保留实际寄存器缩减；正式数量不提前增加。

- **2026-11-05T14:39:25+00:00**：预约复核：物理GPU4/6已被minimax任务预约，避开；GPU0/1/2上出现其他服务，原主卡计时不作为独占验收。改用空闲物理GPU5（PCI a3，HIP7）重新配对测量冻结父实现；CUDA侧保持GPU0。第二版使每CTA统一搬运Q/K/V、每线程FP32状态分半，P只由一组warp写入，避免重复写入竞争。

- **2026-11-05T14:45:14+00:00**：B3 源码审计发现实际资源约束：x2 的 K tile=32，stage5 原 tile 需要80KiB；CuTe 128×256 tile 在stage3/4需要72/96KiB，超过本卡64KiB。将保留真实流水线深度和布局类型，对超限配置明确缩小AMD CTA tile，专项覆盖原有效stage及TN非方阵输入。静态stage入口原版只实现2/3/4，其他值映射2，已纠正初始假设。

- **2026-11-05T14:47:56+00:00**：B2 第二版主卡660/660通过，5090原版191项共享输入+19项状态通过。三轮稳定：KV同契约配置几何平均0.9624/0.9620/0.9625×，QKV 0.9957/0.9958/0.9955×；D256改善1.09/1.33×，D128仍退化。KV D256 VGPR243→155；QKV D256 spill127→13，D128/stage2 spill12→0；D256/stage2仍有29 spill。保留未达性能/无spill目标的明确记录，正在第二卡及旧入口回归。

- **2026-11-05T14:50:30+00:00**：B2 第二张卡660/660通过；191/191组原CUDA/AMD跨平台对照通过（12组逐位一致）。完整回归24模块175接口971项通过，较旧单卡937项包含新增4项Attention及双设备契约检查30项（转置22、HGEMM4、SGEMM2、merge2）；新增2项尚待全接口354组对照和旧路径性能回归完成后正式入账。B3的11个候选已写入独立文件，正在隔离编译，不影响本轮正式加载器。

- **2026-11-05T14:53:17+00:00**：B3 独立编译和首轮探测11/11入口通过，包含非方阵65×129×97、长K4097及所有有效stage；尚未加入正式库存。B2旧入口394项与冻结二进制逐位一致；174条计时中1条（KV D32/stage2，第二轮）出现11.07%退化，其余通过。保留该完整失败测量，检查三轮稳定性并重新测量，不挑选单个最好样本。

- **2026-11-05T14:57:23+00:00**：B2正式入账：175/229=76.42%，剩38待迁移+16精度未验收。双卡各660、原CUDA共享191、完整回归971、全接口354组均通过。旧路径完整重测394项逐位一致、174条计时全部在10%以内；首轮1条异常及全部样本保留。库存和doc镜像已同步。B3首次扩展运行在ISA导出时遇到报告路径错误，数值检查已完成；修正路径后重新运行，不把未落盘试跑算正式结果。

- **2026-11-05T14:58:45+00:00**：B2提交0c9b786已生成，正在推送fork/rocm10。B3扩大候选运行通过360共享输入、2587边界/状态/设备检查、108性能记录；明确采用FP32累加，原FP16 CUDA长K审计仍在编译执行。继续集成B3，不停止在B2交付。

- **2026-11-05T15:02:13+00:00**：B3原CUDA扩大测试发现新问题：带padding的三个x2实现stage5实际申请104960字节，而原封装仅设置98304字节，5090实测报invalid argument；这是CUDA原版资源错误。保留原失败日志，正式报告将30项对应共享用例明确记为无法执行，不伪造数值通过。AMD按64KiB重设tile的stage5已通过。其余CUDA配置继续测量，原FP16长K失败与资源错误分开记录。

- **2026-11-05T15:05:07+00:00**：B3继续审计确认第四个CUDA资源失败：TN swizzle x4的stage5需要102400字节，也超过98304设置与5090的101376上限。原版两次失败日志均保留；总计40项共享用例明确记录资源拒绝。AMD全部stage5有效，资源改写范围已在代码中落实。

- **2026-11-05T15:08:09+00:00**：B3原版CUDA审计完整落盘：360项中256通过、64项因K4096的原FP16累加精度失败、40项因stage5资源超限拒绝；每项失败均保留。AMD最终候选360/360及2587边界通过，36个特化全部真实WMMA、LDS≤64KiB、无scratch。rr对比x2在stage2/3/4的VGPR分别163→136、155→128、164→137，生命周期优化有效。

- **2026-11-05T15:08:56+00:00**：B3双卡各2947项专项通过（360共享+2587边界/状态/设备），CUDA/AMD原始对照256组在门限内；其余64精度失败+40资源失败与预期原版问题一致。全库集成首次揭示旧gemm_checks把所有B都按NN参考计算，3个TN入口15项错误；已按原TN存储契约修复独立参考并将MMA的K tile显式设为16/32，保留首轮失败报告后重跑全库。

- **2026-11-05T15:15:49+00:00**：B3验收闭环完成：全库24模块186接口1048项、全接口376组原输入对照通过。准备正式更新到186/229。B4首个研究结论：8个低精度tiling入口，原FP16 QK/PV模型在大logits出现32/992失败，最大误差0.524；仅把QK累加升级FP32、保留FP16 PV与原输出状态后992/992通过。两版源码/结果均保存，下一步原CUDA验证该精度限制并实现其余布局/分工。

- **2026-11-05T15:17:07+00:00**：正式库存更新到186/229=81.22%，剩27待迁移+16精度未验收；21个接口明确记录累加精度升级。B3源码、最终harness、全库adapter hash审计通过，文档镜像已更新，准备提交。实时MD脚本现在从正式库存刷新比例和待验收清单，避免进度文字落后于证据。
