# LeetCUDA ROCm 10 算子支持清单

更新日期：2026-11-06。当前基础资格报告已在 ROCm 10.1.0 / R9700 / gfx1201 上重新验证；
此前 AMD WMMA GEMM / FlashAttention 的独立报告保留其 2026-11-02 身份。

设备标签更正：本轮发现 HIP 序号与物理 GPU 序号不一致。历史报告未记录 PCI 时，
其“物理 GPUx”标签视为未核实推定；数值结果保留，但不证明物理卡身份或独占性。
最新回归使用PCI `0000:83:00.0`（物理GPU4/HIP6），第二张完整复验卡为`0000:a3:00.0`（物理GPU5/HIP7）；项目保留禁用物理 GPU3 的 PCI 校验。

## 1. 总量、适配比例与可行性

### 1.1 当前比例

**按原有 PyTorch 扩展公开计算接口统计：共 229 个，已验证支持 229 个，
适配比例为 100.00%；目前未验收 0 个，占 0.00%。**

新增接口 `hgemv_tensor_core_cute` 明确把累加状态从 FP16 提升为 FP32，仍使用
AMD WMMA，保持 FP16 输入/输出。它属于数值精度改善后的迁移，不代表逐位复现
NVIDIA FP16 MMA。上一批另增加 9 个明确采用 FP32 累加的原 WMMA HGEMM 接口；229 个中共有60个记录了累加精度变化。
B6另有3个TF32入口显式增加输入残差补偿，FP32累加类型不变；原TF32语义control独立通过。
B4的19个MMA Attention只把QK累加升级FP32，PV和原输出状态类型保留；CuTe保持FP32。B1三个FP32 swizzle保留原精度。

| 状态 | 数量 | 占 229 个原始接口的比例 | 含义 |
| --- | ---: | ---: | --- |
| 已适配并通过验证 | **229** | **100.00%** | 25 个模块主卡 1262 项通过；本轮SIMT HGEMM专项两卡各891项通过；性能独立判定 |
| 尚未适配／尚未验证 | **0** | **0.00%** | B1–B7待迁移集合已关闭；精度失败另列 |
| 已运行，但精度门限未通过 | **0** | **0.00%** | 原精度失败保留，显式升级后通过 |
| **合计** | **229** | **100.00%** | **229 + 0 + 0 = 229** |

按扩展模块统计，共 **25 个**：原始公开接口全部通过当前测试的 **25 个（100%）**，
部分通过的 **0 个（0%）**，尚无原始接口验收通过的 **0 个（0%）**。
所有25个模块的原公开接口均通过当前测试；
**25/25 = 100%** 的模块覆盖率，不能代替 **229/229 = 100.00%** 的接口适配比例。

另有 **8 个独立 AMD 计算接口**：`gemm`、`flash_attention`、`attention`、`ffpa_attention`、`fp8_gemm`、
`fp8_gemm_prepacked`、`fp4_gemm`、`fp4_gemm_prepacked`，以及 2 个权重量化辅助接口
`quantize_fp8_weight`、`quantize_fp4_weight`。
这些接口单独统计：前两者没有逐一实现原版 HGEMM／FlashAttention 的同名变体约定；
FP8/FP4 原实现属于独立 C++ 范围，原本没有计入 229 个 Python 接口。

新增 [FP8 完整链路与独立 C++ 验证](fp8_pipeline_20261103.md)：
保留四种量化粒度、producer/consumer 执行方式和离线 B；实际使用 FP8 WMMA。
2026-11-06新增E5M2完整链路，默认E4M3兼容；两格式双卡各756项、480份跨平台输出通过，见[E5M2报告](fp8_e5m2_20261106.md)。
功能通过，性能仍需优化；不因此增加原始接口的通过数量。

新增 [FP4 完整链路与独立 C++ 验证](fp4_pipeline_20261103.md)：
保留 NVFP4 打包、两级 scale 和四种模式，在寄存器中精确展开为 FP16，再使用
FP16 WMMA/FP32 累加；支持普通／producer-consumer 调度、默认／宽 tile 与离线 B。
这是 NVFP4 数据格式的 AMD 实现，未声称使用原生 FP4 矩阵指令；性能仍待优化。

### 1.2 第十四轮：11个同步SIMT HGEMM

U3完成后 **229/229（100%）**，HGEMM **36/36**。原公开接口已全部功能验收，整体补充范围继续。
11种原SIMT tile、pack、BCF和双缓冲差异保留，明确采用FP32 FMA与状态。
两卡各891项、全库1262项、462组同精度CUDA参考/AMD比较通过，原FP16全库460/462；11个kernel全部零spill/scratch，LDS≤32KiB。
原CUDA91/121、仅槽位修复版99/121过参考，22项长K精度失败保留。
性能仍慢于Torch矩阵库，见[U3报告](hgemm_simt_20261106.md)。

### 1.2 第十三轮：三个BF16 Reduce

U2完成后 **218/229（95.20%）**，Reduce **20/20**，剩11个同步SIMT HGEMM。
保留scalar/x2/x8读取与thread派发，明确升级补偿FP32状态及确定性两级归约。
双卡各252项、全库1185项及440组共同输入对照通过；101项源码审计通过。
原BF16 control与原CUDA均有6组精度失败，原结论保留；升级路径通过严格FP32门限。
x8中等规模约1.13倍Torch，其他被测配置仍慢，见[U2报告](reduce_precision_20261106.md)。

### 1.2 第十二轮：两个scalar/packed HGEMV

U1完成后 **215/229（93.89%）**，HGEMV **6/6**，剩11个HGEMM和3个Reduce精度工作包。
保留scalar与8元素packed的独立分工，明确升级FP32乘加/状态/归约；双卡各150项、全库1179项及434组共同输入对照通过。
原半精度control 30项精确匹配舍入模型，其中11项精度失败；新路径全部通过原0.003门限。
packed在三个被测规模为同卡Torch约1.42–2.02倍，scalar最大规模仍略慢。
另修复CUDA packed局部状态未初始化；原版及修复版失败均保留。见[U1报告](hgemv_vector_20261106.md)。

### 1.2 第十一轮：WS-HGEMM

B7完成功能验收后达到 **213/229（93.01%）**，原pending集合为0，仍有16个精度未验收入口。
真实独立生产者/消费者wave使用三槽LDS、序号原子同步和WMMA；受控消费者延迟下生产者领先3个tile。
双卡各131项、全库1175项、全接口430组对照通过；97项源码审计通过。
明确升级FP32累加；原CUDA两组K4096失败保留，双方通过的13组专项比较全部通过。
性能仍慢于Torch；继续精度工作包及整体补充范围。见[B7报告](ws_hgemm_20261106.md)。

以下为历史批次记录，其数量和“下一步”属于相应批次当时状态；当前状态以上表为准。

### 1.2 第十轮：三个TF32 SGEMM

B6完成后达到 **212/229（92.58%）**，SGEMM **17/17**。
两个矩阵入口使用BF16 WMMA分解TF32，保留静态/动态LDS、stage与swizzle；
BLAS入口执行三个显式量化乘积。**TF32x3残差补偿是明确的输入精度升级**，不宣称原生TF32或普通FP32别名。
双卡各883项、全库1173项、428组全接口对照通过，7个矩阵特化无spill且有真实预取次序。
AMD独立门限0.0002；原CUDA及跨平台沿用此前TF32门限0.005，不能互换。
当前剩1个WS接口待迁移及16个精度未验收；性能和原CUDAgraph限制见[B6报告](tf32_sgemm_20261106.md)。

### 1.2 第九轮：三个异步HGEMM

B5完成后达到 **209/229（91.27%）**，HGEMM **25/36（69.44%）**。
三种SIMT微分块保留真实VGPR预取和双槽LDS，明确升级FP32累加。
双卡各216项、全库1109项、全接口422组对照通过；106项源码hash审计通过。
原CUDA末尾固定槽错误已单独修复，原版29项失败与修复后21项精度失败保留。
性能已测，预取相对同步约1.00–1.09倍，仍慢于Torch；见[B5报告](async_hgemm_20261106.md)。
当前继续B6 TF32、B7 WS及16个精度工作包。

### 1.2 第八轮：剩余20个原Attention入口

B4完成后原FlashAttention达到 **29/29（100%）**。覆盖split-Q/KV、shared/tiling及12个低精度swizzle、Os2g与CuTe，保留各自布局和分工；19项显式升级QK累加。

- 双卡各6677项；全库206接口1088项；全接口416组跨平台对照通过。
- 原CUDA实际执行1759项：1707通过、52大logits精度失败；24项原D96索引错误及16项原shape域之外分别记录。
- 双方参考通过的1707组跨平台对照全部通过，107组逐位一致。
- 210个WMMA特化、LDS≤64KiB；24个有scratch spill，性能仍待优化。
- 旧9入口929项逐位一致，机器指令段完全一致；控制GC和增加采样后的完整420条计时均在10%以内，前两轮失败样本保留。

三项swizzle-QKV使用转置V[B,H,D,N]，不能按普通V调用；详细契约和性能见[B4报告](flash_low_20261105.md)。下一步B5三个异步HGEMM。

### 1.2 第七轮：11个MMA/CuTe HGEMM

B3新增10个MMA及1个CuTe HGEMM，原HGEMM达到 **22/36（61.11%）**。
这11个入口明确升级到FP32累加、保持FP16输入/输出；实际保留各变体的K16/K32、stage、静态/动态LDS、rr、宽读取、swizzle和TN存储机制。

- 双卡各2947项专项；全库186接口、1048项；全接口376组跨平台对照全部通过。
- 36个特化全部使用WMMA、LDS≤64KiB、无scratch；rr降低实际VGPR数量。
- 原CUDA扩大专项：256通过、64项K4096精度失败、40项stage5资源拒绝。AMD全部通过；不隐去原版失败。
- 性能已测，仍落后同机Torch库。详见[B3结果与逐接口机制](mma_hgemm_20261105.md)。

其后B4已经完成功能验收，当前继续B5。

### 1.2 第六轮：两个 shared rr Attention 入口

B2 新增 shared-KV / shared-QKV 两个 `acc_f32_rr`，原 Attention 达到 **9/29（31.03%）**。
保留 FP32 QK/PV、max/sum 和输出状态。D>64 由同一 CTA 内两组 warp 共享搬运、分担输出状态。

- 两卡各660项专项通过；原CUDA/AMD的191组共享输入全部通过参考与跨平台容差。
- 全库175接口、971项通过；354组全接口跨平台对照全部通过。
- 19个rr特化均使用AMD WMMA。KV D256寄存器243→155；QKV D256/stage1 spill127→13。
- 性能已测、仍需优化：D256约1.09–1.33倍加速，D128变慢；保留逐配置三轮原始数据。
- 旧shared专项538项、冻结旧路径394项通过；旧路径完整重测174条计时均在10%退化线内，首轮1条失败记录保留。

详见[B2报告](flash_rr_20261105.md)。B3随后已完成功能验收。

### 1.2 第五轮：三个 swizzle FlashAttention 入口

按预定 Q→QK→QKV 顺序，新增三个 `tiling_qkv_acc_f32_swizzle_*` 入口，
原 FlashAttention 模块达到 **7/29（24.14%）**，本批没有更改累加精度。

- 两张 AMD 卡各 **1329 项**通过：372 项共享输入、879 项边界、42 项状态精度、28 项设备／stream、8 项实际 LDS 布局。
- 原 CUDA 与 AMD 的 372/372 项独立参考及跨平台容差检查通过；36 项逐位一致。AMD 与冻结父实现 372 项均逐位一致。
- 全库 **24 模块、173 接口、937 项**通过；350/350 组全接口共同输入对照通过。
- 42 个特化使用真实 WMMA，LDS≤48 KiB，全部无 scratch spill；保留 Q、QK、QKV 实际布局差异。
- 性能 **1/3 达标**：QKV 相对冻结父实现三轮几何平均约 **1.117×**；Q/QK 功能支持，性能待优化，最坏退化约 21%。
- 旧四个 Attention 入口 394 项数值逐位一致、174 条性能记录均在 10% 退化门槛内；旧 shared/tiling 专项另有 538/858 项通过。

详见[本批实现与验收](flash_swizzle_20261105.md)和[计划执行结果](leetcuda_rocm10_next_batch_plan.md)。两个shared rr随后已完成B2验收。

#### 第四轮：tiling-QK / tiling-QKV FlashAttention

新增 `flash_attn_mma_stages_split_q_tiling_qk_acc_f32` 和
`flash_attn_mma_stages_split_q_tiling_qkv_acc_f32`；原 FlashAttention 模块达到 **4/29**。
支持 D=32/64/96/128/256/512/1024、stage 1/2。两条 V 分块路径分别实现，
D>=256 的 FP16 输出状态舍入经过专门验证，没有静默提升精度。

- AMD 专项 **854/854**：248 项共享输入、578 项边界／契约／并发、28 项存储精度检查。
- 原 CUDA 同样通过 248 项独立 FP64 参考和 28 项存储精度检查；跨平台 248/248 在容差内一致。
- 全库 **24 模块、170 接口、931 项**通过；344/344 组全接口共享输入通过两端参考及跨平台容差检查。
- 28 个特化均使用 WMMA、未发现 scratch spill，最大 LDS 48 KiB。
- 两端各 36 组性能测量及 graph replay 检查；多数形状仍慢于同机 SDPA。
- D>128 按 128 列输出分片并重复计算 QK，这是控制资源占用的代价；其后的三个 FP32 swizzle 入口已在第五轮完成。

接口、数据和限制见 [本轮 tiling FlashAttention 报告](flash_tiled_20261105.md)。

#### 第三轮：两个 shared FlashAttention 接口

新增 `flash_attn_mma_stages_split_q_shared_kv_acc_f32` 和
`flash_attn_mma_stages_split_q_shared_qkv_acc_f32`；原模块达到 2/29。
两个入口分别实现 KV 与 QKV 的 LDS 复用，stage 1/2 有不同的实际缓冲和预取行为。

- 专项 146 个共享 CPU 输入用例、388 项边界／契约／并发检查通过。
- 原 CUDA 与 AMD 各 146/146 通过独立 FP64 参考，146/146 跨平台输出在容差内一致。
- 全部 168 接口的 340 组基础共享输入再次通过两端独立参考及跨平台容差检查。
- 18 个 GPU 特化使用 WMMA；两个较大 shared-QKV 配置出现 scratch spill，性能未达标。
- 两端各完成 22 组性能与 graph replay 检查；AMD 使用有模型驻留的共享卡，计时不作为独占性能验收。
- 该批提出的 tiling-QK / tiling-QKV 已在第四轮完成；后续 swizzle 批次已完成，继续按计划推进 shared `rr` 等剩余项。

接口契约、限制、性能和设备标签更正见 [本轮 FlashAttention 报告](flash_shared_20261105.md)。

#### 第二轮：新增 9 个 WMMA HGEMM 接口

基础 WMMA、block/warp tiling、双缓冲以及静态／动态 LDS 多 stage 路径已接入。
从 157 增至 **166/229（72.49%）**；HGEMM 模块由 2/36 增至 11/36。

- AMD 专项：200 个共享输入用例、1265 项边界／参数／并发检查全部通过，60 组性能及 graph replay 检查。
- 19 个 GPU 特化确认 FP32 WMMA，均无 scratch spill；记录的 HIP mask 为 6，未记录 PCI，没有第二张 AMD 卡验证。
- 原 CUDA 两个双缓冲入口的奇数 tile 收尾错误已修复并在 5090 复验。
- K=4096 的 40 项 CUDA FP16 累加仍未过精度门限；AMD FP32 累加通过，未声称逐位等价。
- 当前完整单卡回归为 23 模块、166 接口、923 项；检查数与此前双卡报告不直接相加。
- 全部 166 个接口的 336 组共享输入在 CUDA / ROCm 上均通过独立参考，两端输出 336/336 在容差内一致、最终源码复验 222/336 逐位一致。
- 性能仍待优化：2048³ 本批较好配置约 0.306 ms，同机 Torch 约 0.148 ms。

接口列表、精度变化及报告见 [第二轮 WMMA 迁移](wmma_hgemm_20261105.md)。

### 1.3 上一轮：5090 对照与矩阵 HGEMV（2026-11-05）

- 使用同一份 CPU 输入文件，在 RTX 5090 CUDA 13.0 与 R9700 ROCm 10.1 上运行，
  记录输入、源码、二进制和镜像 SHA256。CUDA 上游基线来自固定提交，不静默修正。
- 原 17 个精度未验收接口共 168 个跨平台用例：14 个接口的所有用例逐位一致；
  另外 2 个 HGEMM 的 `K=32` 用例触发原 CUDA 末缓冲错误；矩阵 HGEMV 存在指令累加差异。
- 3 个 BF16 Reduce、标量/packed HGEMV，以及 HGEMM 的长 K 精度损失，在 CUDA 也会出现。
  因此“精度未验收”不能解释为“AMD 无法执行这些算法”；严格精度门限保持不变。
- 矩阵 HGEMV 使用 FP32 WMMA 状态后，共享输入 15/15 通过；两张 AMD 卡分别完成
  66、63 项边界、输入契约、并发 stream 与设备检查。完整回归 23 模块、157 接口、890 项通过。
- 性能已测但仍需优化：`M=4096,K=8192` 约 0.311 ms，同机 `torch.mm` 约 0.078 ms。
  这项正确性解决方案不是性能达标声明。

- 当时 157 个接口的 318 个同输入用例，在修复后的 CUDA 与 ROCm 上全部通过；318/318
  跨平台输出在接口容差内一致，221/318 逐位一致。
- 56 个待迁移接口也已完成 5090 基线：53 个全部通过，3 个异步 HGEMM 有长 K 精度失败。
  全部 229 个原始计算接口都有 CUDA 运行证据；CUDA 成功不增加 AMD 支持数。
- 四个 Triton 示例已修复并验证：两张 AMD 卡与 5090 各 195 项通过，108 组公开输出/
  梯度对照全部通过。独立 C++ 的 10 个 CUDA 运行分组已跑完，但没有整体 AMD 验收通过。

详见 [5090 对照验收报告](cross_5090_20261105.md)。
其中保留第一轮剩余 72 个接口的原因；该处的剩余数量属于历史阶段；当前原229接口已全覆盖，补充范围继续。

#### 上一批：新增 3 个异步 SGEMM 接口（2026-11-03）

从 **153/229（66.81%）增至 156/229（68.12%）**，增加 **1.31 个百分点**。
当时全套回归为 **23 模块、156 接口、888 项检查通过**。

- 新增 `sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async`、
  `sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async`、
  `sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async`。
- 保留双 LDS 缓冲、三种线程输出布局和 FP32 FMA 顺序；AMD 使用 VGPR 预取及显式
  load-counter 等待。它是原 `cp.async` 数据流的软件流水线替代，不是直接入 LDS 指令复现。
- 两张卡各 **213 项检查通过**，其中数值检查逐位匹配对应同步版本；另有 **18 组性能比较**。
  三个变体的发布汇编均确认加载与计算重叠、无 scratch spill 和设备函数调用。
- 8×8 变体在本次 2048³ 上约 0.854 ms，比 FP32 PyTorch 基线快 1.369×；其他变体仍有退化。

详见 [异步 SGEMM 迁移报告](async_sgemm_20261103.md)。

#### 上一批：布局转置与 BLAS 新增 14 个接口

累计从 **139/229（60.70%）增至 153/229（66.81%）**，增加 **6.11 个百分点**。
当时全套回归为 **23 模块、153 接口、837 项检查通过**。

- 11 个布局转置入口完成，见下方转置批次记录。
- 新增 `sgemm_cublas`、`hgemm_cublas_tensor_op_nn`、`hgemm_cublas_tensor_op_tn`，
  由 hipBLAS/rocBLAS 实现。该历史批次 HGEMM 默认模块发布这 2 个 BLAS 计算接口，本轮已增加 9 个 WMMA。
- BLAS 扩展检查 **170 项通过**、性能比较 **9 组**；第二张卡的 SGEMM/HGEMM 回归
  **217 项通过**。保留 TN 原始打包布局与普通 `COMPUTE_16F` 数学模式。
- TF32 模式探测返回不支持，仍保留为待实现项；未用 FP32 别名冒充 TF32。

详见 [BLAS 迁移报告](blas_20261103.md)。

另完成 3 个 HGEMV 候选的 81 项研究检查，发现长 K 精度失败，因此保持未验收。
标量与 packed 共 51 项匹配独立 FP16 算术模型；矩阵版本确认使用真实 FP16 WMMA，
但其内部累加／舍入仍需研究。见 [HGEMV 研究](hgemv_research_20261103.md)。

#### 本轮第一批：本轮进展：11 个布局转置接口完成

- 已通过接口从 **139/229（60.70%）增至 150/229（65.50%）**，增加 **4.80 个百分点**。
- Matrix Transpose 的 **25/25 个原始接口**已通过当前测试；新实现保留寄存器、LDS、
  行／列拷贝、向量化和 swizzle 的区分，使用显式 HIP 布局替代 CuTe 依赖。
- 完整回归 **22 模块、150 接口、661 项检查通过**；物理 GPU1 的转置检查 **324 项通过**。
- 新增 **33 组性能对比**，覆盖 3 个规模 × 11 个变体。最佳版本快于 PyTorch `copy_(x.T)`，
  部分教学版本仍较慢；未把该基线对比解释成所有 AMD 转置实现的性能上限。

详见 [布局转置迁移报告](layouts_20261103.md)。

#### 上一批：本轮进展：15 个常规 HIP 候选的实际结果

上次为 **135/229 = 58.95%**。本轮对 15 个常规 HIP 候选进行了研究：

- **4 个新增通过**：Merge Attention States 1 个接口、同步 SGEMM 3 个接口。
- **11 个转为精度未验收**：同步 HGEMM 均已编译运行，但未通过当前 FP64 参考精度门限。
- 基础完整回归为 **22 模块、139 接口、387 项检查通过**；物理 GPU2 上新增模块
  的补充检查 **87 项通过**。
- 新增接口覆盖率提高 **1.75 个百分点**，没有将能够编译但精度未通过的接口计为成功。

HGEMM 的关键研究结果：11 个变体 × 3 个随机种子，共 **33/33 项逐位匹配独立模拟的
FP16 FMA 累加结果**；另 **55/55 项**分块边界检查通过。但这些变体相对 FP64 的误差
最多约 0.01880，各自都至少有一项未通过 `atol=0.01, rtol=0.01`。
这说明这些用例的差异来自 FP16 累加舍入，不能误判为 HIP 翻译破坏了计算。
它们保留为研究实现，不在默认加载器中发布。需要当前已验收的半精度矩阵乘，
可使用已有的 FP32 累加 AMD WMMA `gemm`。

详见 [本轮实现及研究报告](continuation_20261103.md)。

### 1.3 未支持的原因，以及“不能适配”的含义

| 原因分类 | 数量 | 分布 | 当前结论 |
| --- | ---: | --- | --- |
| 原公开接口尚未功能验收 | **0** | 无 | 229个均通过当前测试；60个显式累加升级、3个输入精度升级 |
| 不在229分母内的补充范围 | 单独统计 | 独立构建/非绑定源码（E5M2、训练Attention与FFPA功能已完成） | 仍需继续实现与验收，详见第4节 |

当前实测证明通过当前资格门限的是 229 个原始接口，另加 8 个独立 AMD 计算接口。
**目前没有证据证明某个算子的数学功能永久无法在 ROCm 上实现。**
“原 NVIDIA 实现不能原样运行”“尚未适配”“已运行但精度未验收”是不同状态；
也不能承诺全部变体能保持原有精度语义及性能通过同一门限。

架构与依赖判断依据本地源码。AMD 的
[HIP 迁移说明](https://rocm.blogs.amd.com/software-tools-optimization/hipify/README.html)
指出内联 PTX、特定 intrinsic 和硬编码依赖需要额外处理；
[CUTLASS 官方说明](https://docs.nvidia.com/cutlass/latest/overview.html)
列出的目标为 NVIDIA GPU。这些资料说明改写原因，不证明数学功能不能迁移。

### 1.4 统计分母与范围

229 的统计规则：扫描 `kernels/` 下的 `PYBIND11_MODULE`，统计绑定块中的
`TORCH_BINDING_COMMON_EXTENSION` 和 `m.def` 计算接口，剔除注释、句柄管理函数，
在模块内按名称去重。包含可选条件编译分支，因而不是某一次构建必然导出的接口数。

- FlashAttention 的 29 个接口含 `BUILD_FLASH_ATTN_MMA_OTHERS` 下的 3 个可选接口。
- 不展开模板参数、stage、tile、dtype、shape，也不把测试用例计为新接口。
- 不计 `init_cublas_handle`、`destroy_cublas_handle`。
- **229 不包括没有 Python 绑定的 CUDA kernel、独立 C++ benchmark、原有 Triton／
  CuTe DSL 示例及第三方依赖。93.89% 不能写成整个仓库所有 GPU 实现的完成率。**
- 每个接口的名称、状态、原因、绑定位置与源码 hash，见
  [逐接口清点数据](leetcuda_rocm10_operator_inventory.json)。
- “通过”以测试覆盖的输入和当前 R9700/gfx1201 平台为限，不代表任意 shape、
  其他 AMD 架构或完整训练路径均已支持。

## 2. 原有算子的逐模块比例

| 算子／模块 | 原接口总数 | 已通过 | 待适配 | 精度失败 | 适配比例 | 当前支持范围 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Elementwise Add（`elementwise`） | 6 | 6 | 0 | 0 | 100.00% | FP32/FP16 标量及向量化加法 |
| ReLU（`relu`） | 6 | 6 | 0 | 0 | 100.00% | 标量与 packed 实现 |
| GELU（`gelu`） | 6 | 6 | 0 | 0 | 100.00% | 标量与 packed 实现 |
| Sigmoid（`sigmoid`） | 6 | 6 | 0 | 0 | 100.00% | 标量与 packed 实现 |
| Swish（`swish`） | 6 | 6 | 0 | 0 | 100.00% | 标量与 packed 实现 |
| HardSwish（`hardswish`） | 6 | 6 | 0 | 0 | 100.00% | 标量与 packed 实现 |
| HardShrink（`hardshrink`） | 6 | 6 | 0 | 0 | 100.00% | 标量与 packed 实现 |
| ELU（`elu`） | 6 | 6 | 0 | 0 | 100.00% | 标量与 packed 实现 |
| Softmax（`softmax`） | 9 | 9 | 0 | 0 | 100.00% | per-token、safe、online 等 FP32/FP16 实现 |
| LayerNorm（`layer_norm`） | 8 | 8 | 0 | 0 | 100.00% | FP32/FP16、不同累加精度与 packed 实现 |
| RMSNorm（`rms_norm`） | 9 | 9 | 0 | 0 | 100.00% | FP32/FP16、不同累加精度与 packed 实现 |
| Reduce Sum（`reduce`） | 20 | 20 | 0 | 0 | 100.00% | 所有原入口通过；3个BF16自身累加变体明确升级补偿FP32 |
| Dot Product（`dot_product`） | 5 | 5 | 0 | 0 | 100.00% | FP32/FP16 点积 |
| RoPE（`rope`） | 3 | 3 | 0 | 0 | 100.00% | 旋转位置编码 |
| Embedding（`embedding`） | 6 | 6 | 0 | 0 | 100.00% | 查表及向量化实现 |
| Histogram（`histogram`） | 2 | 2 | 0 | 0 | 100.00% | 标量与向量化直方图 |
| SGEMV（`sgemv`） | 3 | 3 | 0 | 0 | 100.00% | FP32 矩阵向量乘，k16/k32/k128 变体 |
| HGEMV（`hgemv`） | 6 | 6 | 0 | 0 | 100.00% | k16/k32/k128、WMMA及scalar/packed CuTe均通过；后三个明确升级FP32累加 |
| SGEMM（`sgemm`） | 17 | 17 | 0 | 0 | 100.00% | 10同步、3预取、1普通BLAS及3个TF32x3入口；后3个明确升级输入精度 |
| Matrix Transpose（`mat_transpose`） | 25 | 25 | 0 | 0 | 100.00% | 原 14 个实现及 11 个显式 HIP 布局变体 |
| NMS（`nms`） | 1 | 1 | 0 | 0 | 100.00% | IoU ballot 与单 block 顺序消除 |
| 原版 HGEMM（`hgemm`） | 36 | 36 | 0 | 0 | 100.00% | 2个BLAS、9个WMMA、11个MMA/CuTe、3个异步及11个同步SIMT通过；低精度状态升级明确记录 |
| WS-HGEMM（`ws-hgemm`） | 1 | 1 | 0 | 0 | 100.00% | 独立生产者/消费者、三槽LDS与WMMA；显式FP32累加升级 |
| 原版 FlashAttention（`flash-attn`） | 29 | 29 | 0 | 0 | 100.00% | 全部原入口功能通过；19个MMA入口明确升级QK累加，保留PV和输出状态；性能独立记录 |
| Merge Attention States | 1 | 1 | 0 | 0 | 100.00% | FP32/FP16/BF16，含空状态、可选 LSE 和设备／stream 检查 |
| **合计：25 个扩展模块** | **229** | **229** | **0** | **0** | **100.00%** | 其中60个累加精度升级、3个TF32输入精度升级；8 个独立 AMD 计算接口另列 |

“全部接口通过”也仅表示通过现有测试集，仍须遵守原实现的 dtype、形状、连续性、
对齐和 tile/pack 限制。已支持的基础模块调用入口为 `leetcuda_rocm.load("模块名")`。

## 3. 新增的 AMD 原生实现

| 算子 | 公共接口 | 已支持功能 | 当前限制 |
| --- | --- | --- | --- |
| FP16/BF16 GEMM | `leetcuda_rocm.gemm` | FP32 累加、AMD WMMA、转置及 strided 输入、M/N/K 尾部、可选预分配输出 | 二维矩阵乘、正尺寸、前向计算；不提供 backward |
| FlashAttention 前向 | `leetcuda_rocm.flash_attention` | FP16/BF16、FP32 累加、AMD WMMA、在线 softmax、因果/非因果、strided 输入；非因果模式支持 NQ≠NK | D 仅支持 16/32/64/128；因果模式要求 NQ=NK；无 GQA、dropout、任意 mask、backward |
| GQA/训练Attention | `leetcuda_rocm.attention` | FP16/BF16 D16/32/64/128/256、GQA、布尔/加性mask、显式Philox dropout、Q/K/V一阶梯度；左上因果支持NQ≠NK | 连续输入；mask无梯度；不支持二阶导数；D256有性能差距，见[专项报告](attention_training_20261106.md) |
| FP8 GEMM 完整链路 | `leetcuda_rocm.fp8_gemm` | BF16 → E4M3/E5M2 动态量化 → FP8/BF8 WMMA/FP32 累加 → BF16；四种 scale 模式；普通／producer-consumer 调度 | 连续 BF16；M>0、N%8=0、K%16=0；FP32 scale/inverse 必须处于正常有限范围；性能尚未收敛 |
| FP8 GEMM 离线权重 | `leetcuda_rocm.fp8_gemm_prepacked` | 复用 `quantize_fp8_weight` 输出的 B8T/scales，仅在线量化 A | 同上；B 的离线量化时间须单独统计；无反向 |
| NVFP4 GEMM 完整链路 | `leetcuda_rocm.fp4_gemm` | BF16 → E2M1 数据／E4M3 分组 scale → 精确 FP16 展开 → FP16 WMMA/FP32 累加 → BF16；四种 scale 模式、两种 tile／调度 | 连续 BF16；M>0、N%8=0、K%64=0；level-1 scale/inverse 正常有限 FP32，缩放累加不溢出；性能仍落后库基线 |
| NVFP4 GEMM 离线权重 | `leetcuda_rocm.fp4_gemm_prepacked` | 复用 `quantize_fp4_weight` 输出的 B4T、canonical SFB、可选列 scale；仅在线量化 A | 同上；默认 BM128/BN128/stages4，宽配置 BM32/BN256/stages6；无反向 |

前两个为独立 Triton GPU 实现。测试检查生成汇编中的
`v_wmma_f32_16x16x16_f16` / `v_wmma_f32_16x16x16_bf16` 指令，确认实际使用
AMD 矩阵单元。PyTorch matmul 和 SDPA 用作测试参考及性能基线。

FP8 为原生 HIP 实现，Python 和独立 C++ 共用同一头文件，确认
`v_wmma_f32_16x16x16_fp8_fp8`。量化字节与独立 CPU E4M3 RNE 参考逐位一致，
矩阵结果与量化后的 FP64 参考比较；相对未量化输入的量化损失另列。

FP4 同样是 Python／C++ 共用的原生 HIP 实现，确认 FP16 WMMA。
独立检查量化码点、scale 布局、补零行与 FP64 GEMM；65,024 个 GPU 中间展开值
逐位匹配参考，包含正负零。宽配置为适应 64 KiB LDS 将 BM 从原 128 降为 32，
BN/stages 保持原约定；此差异及量化损失在报告中明确列出。

新增实现通过正确性验证不意味着所有性能形状都优于基线：长序列因果 Attention
仍有性能差距，详细数据见 [GEMM 与 Attention 验证报告](matrix_attention.md)。

## 4. 尚未支持或尚未完成验证的部分

| 算子／目录／功能 | 状态 | 尚缺内容 |
| --- | --- | --- |
| 原版 `kernels/hgemm` | 36/36原接口功能通过 | 显式FP32状态升级及原失败证据保留；性能及WGMMA等未绑定实现继续推进 |
| 原版 `kernels/ws-hgemm` | 原接口已支持 | 独立生产者/消费者和三槽同步通过；性能待优化 |
| 原版 `kernels/flash-attn` | 29个原接口全部功能通过 | 保留三项转置V布局；19项QK精度升级；24特化有spill，性能待优化 |
| SGEMM | 17/17 原接口功能支持 | TF32采用显式残差补偿的BF16矩阵/hipBLAS替代，性能仍待优化 |
| HGEMV 的 CuTe 变体 | 3/3已支持 | 均明确提升FP32累加；原FP16失败保留，性能按形状记录 |
| Matrix Transpose 的 CuTe 变体 | 已支持 | 11 个接口已用显式 HIP 布局实现；CUDA CuTe 源码保留 |
| Reduce Sum 的 BF16 自身累加变体 | 3/3已支持，精度升级 | 原BF16失败保留；使用补偿FP32、确定性两级归约 |
| FP8/FP4 GEMM、FFPA 扩展 | 三者功能通过，整体性能未完成 | E4M3/E5M2、NVFP4完整链路及FFPA三路径已接入；FFPA双卡各2755、跨平台1700份比较通过；性能仍有差距，FP4完整链两张AMD和5090各1560项、跨平台全部逐位一致 |
| 新增 Attention 的扩展功能 | 计划范围功能完成 | 独立`attention`支持GQA、bool/加性mask、显式Philox dropout、Q/K/V一阶梯度及D≤256；NQ≠NK左上因果。双卡、5090、性能和空间通过；二阶导数/mask梯度明确不支持 |
| `kernels/cutlass`、CuTe DSL 示例 | 未移植 | NVIDIA 库、指令及对应编译入口 |
| `kernels/swizzle` 示例 | 未移植 | 独立布局、矩阵乘和转置示例 |
| `kernels/openai-triton` 原有示例 | 四个已验证，性能待优化 | vector-add、fused-softmax、layer-norm（含梯度）、merge-attn-states 两张 AMD 卡与 5090 各 195 项通过；不计入 229 个扩展接口 |
| `kernels/interview` 的 GPU 算子与 C++ benchmark | FP8/FP4/FFPA与base/SGEMV已接入 | FFPA新增独立C++729项；base/SGEMV 22 kernel已在两张AMD与5090各370项通过；其余SGEMM/HGEMM/FlashAttention及统一入口继续推进 |
| 原目录 Python 性能脚本及 NVIDIA 工具示例 | 未整体适配 | CUDA 专用构建参数、依赖和 Nsight 等工具流程 |
| 其他 AMD GPU 架构 | 尚未验证 | 当前实现只对 gfx1201 提供资格声明，不能直接外推至 MI 系列／wave64 GPU |

BF16 原自身累加的3个入口，U2已通过显式精度升级验收：

- `block_all_reduce_sum_bf16_bf16`
- `block_all_reduce_sum_bf16x2_bf16`
- `block_all_reduce_sum_bf16x8_pack_bf16`

这3个入口计入当前229个通过接口；原BF16逐次舍入失败保留，新实现采用补偿FP32。
名称以 `_f32` 结尾的既有版本保持原实现。

不在 229 个 Python 扩展入口分母内的内容仍需要独立迁移。其中主要源码范围如下，
这里按**实现文件**计数，与接口数不相加：

| 补充范围 | 实现文件数 | 当前原因与状态 |
| --- | ---: | --- |
| `interview` 的 GPU kernel 头文件 | 8 | FP8/FP4/FFPA三个头文件已接入；base/SGEMV两个头文件的22 kernel已通过双卡及5090各370项，370项跨平台对照也全部通过，其中250项逐位一致；SGEMM/HGEMM/FlashAttention三个头文件及统一构建继续推进 |
| `openai-triton` 的 Triton 实现 | 4 | vector-add、fused-softmax、layer-norm（含反向）、merge-attn-states 已接入：AMD 两卡与 5090 各 195 项检查通过，108 组输出对照通过；性能仍待优化。CUDA 版 merge-attn-states 的 1 个绑定已通过验证并计入 229，不重复计入本行 |
| `cutlass`／CuTe DSL GPU 示例 | 4 | 1 个 CuTe vector-add 源文件、3 个 CuTe DSL SGEMM 示例；需替换 NVIDIA 后端与构建方式 |
| `swizzle` GPU 示例 | 3 | 独立矩阵乘／转置／布局实现，尚未迁移和验证 |
| `nvidia-nsight` GPU 示例 | 2 | 示例算子代码尚未迁移，Nsight 分析流程也需要换成 AMD 工具 |

其他未绑定的CUDA实现也不在229的分母：2个HGEMM WGMMA、1个HGEMM TN-swizzle、1个tiling-QK acc_f32_rr Attention均已明确加入后续清单。4个历史SGEMM stage2/3名称只有声明，不误计为已存在实现。
上述补充文件数用于标明范围，不声称是全仓独立 GPU kernel 函数的总数。
没有实现内容的 `kernels/transformer` 占位目录不计为算子。

FP8 硬件能力先导实验：E4M3/E5M2 两种格式 × 3 个形状，**6/6 项通过**，
确认实际 FP8/BF8 WMMA。仅覆盖预量化输入，不增加公开接口通过数；
见 [FP8 能力报告](fp8_capability_20261103.md)。
后续 E4M3 完整链路的验收见 [FP8 迁移报告](fp8_pipeline_20261103.md)，
E5M2随后已完成独立的完整链路验收，见[2026-11-06报告](fp8_e5m2_20261106.md)，先导结果与正式验收分开保留。

## 5. 验证依据与复现入口

| 验证项 | 结果 | 证据 |
| --- | --- | --- |
| NVFP4 完整链路，Python／C++ | GPU1 1,540 项、GPU0 1,541 项、大尺寸 40 项、C++ 263 项通过；Python/C++ 各 96 组性能比较，仍落后 BF16 库基线 | [FP4 迁移报告](fp4_pipeline_20261103.md) |
| FP8 E4M3/E5M2完整链路，Python／C++ | 每种格式两卡各756项、C++143项；每种Python/C++各48性能；480输出跨平台通过，性能仍落后库基线 | [FP8完整格式报告](fp8_e5m2_20261106.md) |
| 本轮基础算子，PCI 0000:83:00.0，2026-11-06 | 25模块、229接口、1262项；全接口同精度CUDA参考/AMD对照462/462，原FP16树460/462 | [当前基础报告](ffpa_20261106/qualification229.json) |
| 本轮 swizzle 专项，PCI 03 / 43 | 两卡各 1329 项；42 个 WMMA 特化；372/372 CUDA/AMD 对照通过；性能 1/3 达标 | [本轮报告](flash_swizzle_20261105.md) |
| 上轮基础算子，PCI 0000:c3:00.0，2026-11-05 | 24 模块、170 接口、931 项检查通过 | [历史基础报告](qualification_20261105_flash_tiled.json) |
| 上轮 tiling FlashAttention 专项 | 854 项、28 个 WMMA 特化；与 CUDA 248/248 对照通过 | [本轮专项报告](flash_tiled_20261105.md) |
| 上轮基础算子，PCI 0000:e3:00.0，2026-11-05 | 24 模块、168 接口、927 项检查通过 | [历史基础报告](qualification_20261105_flash_shared.json) |
| 上轮 shared FlashAttention 专项 | 534 项、18 个 WMMA 特化；与 CUDA 146/146 对照通过 | [专项报告](flash_shared_20261105.md) |
| 上轮 WMMA 批次，HIP mask 6，未记录 PCI | 23 模块、166 接口、923 项检查通过 | [历史基础报告](qualification_20261105_wmma.json) |
| 本轮 WMMA 专项及全接口跨平台对照 | 专项 1465 项、60 组 AMD 性能数值检查；全接口 336/336 对照通过 | [WMMA 迁移报告](wmma_hgemm_20261105.md) |
| 上轮基础算子，物理 GPU0，2026-11-05 | 23 模块、157 接口、890 项检查通过 | [历史基础报告](qualification_20261105.json) |
| 历史异步批次，2026-11-03 | 23 模块、156 接口、888 项检查通过 | [历史基础报告](qualification_20261103_async.json) |
| 异步 SGEMM，物理 GPU0／GPU1 | 两卡各 213 项检查、18 组性能比较通过；实际发布汇编确认预取与计算重叠 | [异步 SGEMM 报告](async_sgemm_20261103.md) |
| 上批 BLAS 接入后的全套回归，物理 GPU0 | 23 模块、153 接口、837 项检查通过 | [历史基础报告](qualification_20261103_blas.json) |
| BLAS 集成，物理 GPU1／逻辑 GPU1 | SGEMM/HGEMM 共 13 接口、217 项通过 | [BLAS 第二张卡](blas_device1_20261103.json) |
| BLAS 正确性与性能 | 170 项扩展检查、9 组性能比较 | [BLAS 迁移报告](blas_20261103.md) |
| 布局转置，物理 GPU1／逻辑 GPU1，2026-11-03 | 25 接口、324 项检查通过 | [转置第二张卡](layouts_device1_20261103.json) |
| 布局转置性能，物理 GPU0 | 252 项扩展检查、33 组性能对比 | [布局报告](layouts_20261103.md) |
| 上一批新增模块，物理 GPU2，2026-11-03 | Merge Attention States 与 SGEMM，87 项检查通过 | [本轮第二张卡报告](continuation_device1_20261103.json) |
| HGEMV CuTe 替代研究，2026-11-03 | 81 项研究检查、3 个变体精度未验收，确认 FP16 WMMA 指令 | [HGEMV 研究](hgemv_research_20261103.json) |
| 同步 HGEMM 研究，2026-11-03 | 33 项 FP16 算术模拟一致、55 项边界通过，11 个变体精度未验收 | [研究结果](hgemm_research_20261103.json) |
| 历史基础算子，物理 GPU0，2026-11-02 | 21 模块、135 变体、316 项检查通过 | [基础数值报告](qualification.json) |
| 原有算子的非默认设备补充检查，物理 GPU2 | 48 项通过，覆盖 histogram、reduce、dot product | [非默认设备报告](device1.json) |
| 新增 GEMM 与 Attention，物理 GPU0 | 68 项检查、24 组性能对比；28 个采样编译配置包含 WMMA，均无 spill | [新增实现报告](matrix_attention.json) |
| 新增 GEMM 与 Attention，物理 GPU2／逻辑 GPU1 | 68 项检查通过 | [第二张卡报告](matrix_attention_device1.json) |

以上数字分属不同时间与测试集；第二张卡上的重复验证不计为新增算子。
2026-11-02 的 21 模块报告保留为历史，当前基础资格以2026-11-06第十四轮的25模块、229接口、1262项报告为准。

在项目根目录使用锁定的 ROCm 10 启动器运行：

```bash
cd /home/zihaomu/zihaomu_workspace/leetcuda_workspace/LeetCUDA

# 先核对实际 PCI、占用和预约；2 仅是本轮主卡实际使用的 HIP 序号
export HIP_VISIBLE_DEVICES=2

# 原有基础算子
./scripts/rocm/run.sh python -m leetcuda_rocm.smoke

# 新增 GEMM / FlashAttention：正确性、WMMA 指令和性能
./scripts/rocm/run.sh python -m leetcuda_rocm.matrix_attention_check

# FP8 完整链路及专用 C++ benchmark
./scripts/rocm/run.sh python -m leetcuda_rocm.fp8_check --performance \
  --output artifacts/rocm/fp8_pipeline.json
./scripts/rocm/build_bench.sh fp8 --run --performance

# NVFP4 完整链路、专用 C++ benchmark、大尺寸自动宽 tile
./scripts/rocm/run.sh python -m leetcuda_rocm.fp4_check --performance \
  --output artifacts/rocm/fp4_pipeline.json
./scripts/rocm/build_bench.sh fp4 --run --performance
./scripts/rocm/run.sh python -m leetcuda_rocm.fp4_large_check \
  --output artifacts/rocm/fp4_large.json
```

不得将默认 HIP 序号视为物理 GPU0；运行前核对 PCI、占用和预约，物理 GPU3 禁止计算。
环境身份、源码 hash、数值容差和逐项结果以各报告为准。
更多接口约束见 [ROCm 使用说明](README.md)，
运行环境见 [环境锁](../../scripts/rocm/environment.lock.json)。

FFPA补充范围已功能验收：三个Split-D路径，双卡各2755项、5090修复参考1220项、跨平台1700份输出/LSE、C++729项通过；27特化零scratch。原CUDA18项stage3尾部故障保留并提供显式修复。两端各135性能配置完成，性能差距仍开放。见[FFPA报告](ffpa_20261106.md)。

独立补充范围以[逐项清单](leetcuda_rocm10_supplemental_scope.md)为准；[base/SGEMV正式报告](interview_base_20261106.md)已完成。

独立SGEMM四路径已验收：[234项双卡与原CUDA专项、234项共同输出比较](interview_sgemm_20261106.md)，四kernel零scratch。interview功能40/53、完整跨平台34/53；其余范围继续执行。

NVFP4完整链跨平台证据已闭合：[1560项全部逐位一致](fp4_cross_20261106.md)。两端各48性能配置已测，AMD相对本机BF16库仍有明显差距，继续优化。interview功能和完整跨平台当前均40/53。

独立HGEMM补充：[五路径验收](interview_hgemm_20261106.md)。双卡各477项、明确FP32参考及跨平台各336项通过；interview功能45/53，44条CUDA源码参考加1条WGMMA数学参考。剩8条独立FlashAttention及其他补充范围继续执行。

未绑定计算入口补充3/4：[WGMMA两种原精度及TN swizzle验收](unbound_gemm_20261106.md)。新增load_unbound(hgemm)，不计入原229；剩余Attention与其他补充范围继续执行。
