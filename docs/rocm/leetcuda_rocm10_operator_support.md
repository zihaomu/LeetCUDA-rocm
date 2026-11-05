# LeetCUDA ROCm 10 算子支持清单

更新日期：2026-11-05。当前基础资格报告已在 ROCm 10.1.0 / R9700 / gfx1201 上重新验证；
此前 AMD WMMA GEMM / FlashAttention 的独立报告保留其 2026-11-02 身份。

设备标签更正：本轮发现 HIP 序号与物理 GPU 序号不一致。历史报告未记录 PCI 时，
其“物理 GPUx”标签视为未核实推定；数值结果保留，但不证明物理卡身份或独占性。
最新回归使用 PCI `0000:e3:00.0`（物理 GPU7 / HIP mask 4），并增加禁用物理 GPU3 的 PCI 校验。

## 1. 总量、适配比例与可行性

### 1.1 当前比例

**按原有 PyTorch 扩展公开计算接口统计：共 229 个，已验证支持 168 个，
适配比例为 73.36%；目前未验收 61 个，占 26.64%。**

新增接口 `hgemv_tensor_core_cute` 明确把累加状态从 FP16 提升为 FP32，仍使用
AMD WMMA，保持 FP16 输入/输出。它属于数值精度改善后的迁移，不代表逐位复现
NVIDIA FP16 MMA。上一批另增加 9 个明确采用 FP32 累加的原 WMMA HGEMM 接口；168 个中共有 10 个记录了累加精度变化。本轮两个 FlashAttention 原入口本身就采用 FP32 累加。

| 状态 | 数量 | 占 229 个原始接口的比例 | 含义 |
| --- | ---: | ---: | --- |
| 已适配并通过验证 | **168** | **73.36%** | 当前 24 个模块单卡 927 项检查通过；本轮 FlashAttention 专项 534 项通过 |
| 尚未适配／尚未验证 | **45** | **19.65%** | 需要架构相关重写或库后端迁移 |
| 已运行，但精度门限未通过 | **16** | **6.99%** | 11 个 HGEMM、2 个 HGEMV、3 个 BF16 Reduce |
| **合计** | **229** | **100.00%** | **168 + 45 + 16 = 229** |

按扩展模块统计，共 **25 个**：原始公开接口全部通过当前测试的 **19 个（76%）**，
部分通过的 **5 个（20%）**，尚无原始接口验收通过的 **1 个（4%）**。
部分通过的是 HGEMV、SGEMM、HGEMM、Reduce、FlashAttention；尚无验收通过的是
WS-HGEMM。至少有一个接口通过的模块覆盖率为
**24/25 = 96%**，不能代替 **168/229 = 73.36%** 的接口适配比例。

另有 **6 个独立 AMD 计算接口**：`gemm`、`flash_attention`、`fp8_gemm`、
`fp8_gemm_prepacked`、`fp4_gemm`、`fp4_gemm_prepacked`，以及 2 个权重量化辅助接口
`quantize_fp8_weight`、`quantize_fp4_weight`。
这些接口单独统计：前两者没有逐一实现原版 HGEMM／FlashAttention 的同名变体约定；
FP8/FP4 原实现属于独立 C++ 范围，原本没有计入 229 个 Python 接口。

新增 [FP8 完整链路与独立 C++ 验证](fp8_pipeline_20261103.md)：
保留四种量化粒度、producer/consumer 执行方式和离线 B；实际使用 FP8 WMMA。
功能通过，性能仍需优化；不因此增加原始接口的通过数量。

新增 [FP4 完整链路与独立 C++ 验证](fp4_pipeline_20261103.md)：
保留 NVFP4 打包、两级 scale 和四种模式，在寄存器中精确展开为 FP16，再使用
FP16 WMMA/FP32 累加；支持普通／producer-consumer 调度、默认／宽 tile 与离线 B。
这是 NVFP4 数据格式的 AMD 实现，未声称使用原生 FP4 矩阵指令；性能仍待优化。

### 1.2 第三轮：两个原始 FlashAttention 接口

新增 `flash_attn_mma_stages_split_q_shared_kv_acc_f32` 和
`flash_attn_mma_stages_split_q_shared_qkv_acc_f32`；原模块达到 2/29。
两个入口分别实现 KV 与 QKV 的 LDS 复用，stage 1/2 有不同的实际缓冲和预取行为。

- 专项 146 个共享 CPU 输入用例、388 项边界／契约／并发检查通过。
- 原 CUDA 与 AMD 各 146/146 通过独立 FP64 参考，146/146 跨平台输出在容差内一致。
- 全部 168 接口的 340 组基础共享输入再次通过两端独立参考及跨平台容差检查。
- 18 个 GPU 特化使用 WMMA；两个较大 shared-QKV 配置出现 scratch spill，性能未达标。
- 两端各完成 22 组性能与 graph replay 检查；AMD 使用有模型驻留的共享卡，计时不作为独占性能验收。
- 下一批优先 tiling-QK / tiling-QKV，减少完整 Q 常驻寄存器的压力，再推进 swizzle 与 split-KV。

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
其中保留第一轮剩余 72 个接口的原因；后续完成 9 个 WMMA HGEMM、2 个 FlashAttention，现剩 61 个。

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

| 原因分类 | 数量 | 分布 | 需要做什么 |
| --- | ---: | --- | --- |
| NVIDIA 矩阵指令或 CuTe 实现需改写 | **41** | HGEMM 11、FlashAttention 27、SGEMM 2、WS-HGEMM 1 | 更换为 AMD 矩阵指令、HIP/Triton 或相应后端，重新设计布局与调度 |
| `cp.async` 异步搬运流水线需改写 | **3** | HGEMM 3 | 重做搬运、同步和流水线；SGEMM 的 3 个接口已有通过验证的 AMD VGPR 预取实现 |
| cuBLAS TF32 模式尚未替换 | **1** | SGEMM TF32 1 | 当前 hipBLAS/rocBLAS 返回不支持；需单独实现并验证数学模式 |
| FP16 自身累加精度未验收 | **11** | 同步 HGEMM 11 | 算术语义已验证；需要单独评估累加精度契约，或使用已支持的 FP32 累加接口 |
| FP16 HGEMV 精度未验收 | **2** | 标量和 packed CuTe GEMV | 与 CUDA 逐位一致，但长 K 精度不足；FP32 累加矩阵版已独立验收 |
| BF16 自身累加精度未达标 | **3** | Reduce 3 | 重新评估算法或精度契约；改成 FP32 累加不等于原变体通过 |
| **合计** | **61** | **41 + 3 + 1 + 11 + 2 + 3** | **45 个待迁移、16 个精度未验收** |

当前实测证明通过当前资格门限的是 168 个原始接口，另加 6 个独立 AMD 计算接口。
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
  CuTe DSL 示例及第三方依赖。68.12% 不能写成整个仓库所有 GPU 实现的完成率。**
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
| Reduce Sum（`reduce`） | 20 | 17 | 0 | 3 | 85.00% | FP32、FP16、BF16 输入 FP32 累加、FP8、INT8；排除下文列出的 3 个变体 |
| Dot Product（`dot_product`） | 5 | 5 | 0 | 0 | 100.00% | FP32/FP16 点积 |
| RoPE（`rope`） | 3 | 3 | 0 | 0 | 100.00% | 旋转位置编码 |
| Embedding（`embedding`） | 6 | 6 | 0 | 0 | 100.00% | 查表及向量化实现 |
| Histogram（`histogram`） | 2 | 2 | 0 | 0 | 100.00% | 标量与向量化直方图 |
| SGEMV（`sgemv`） | 3 | 3 | 0 | 0 | 100.00% | FP32 矩阵向量乘，k16/k32/k128 变体 |
| HGEMV（`hgemv`） | 6 | 4 | 0 | 2 | 66.67% | k16/k32/k128 与 FP32 累加 WMMA 通过；标量/packed 自身 FP16 累加未验收 |
| SGEMM（`sgemm`） | 17 | 14 | 3 | 0 | 82.35% | 10 个同步 FP32 变体、3 个 AMD VGPR 预取变体与 1 个 BLAS；WMMA、TF32 待迁移 |
| Matrix Transpose（`mat_transpose`） | 25 | 25 | 0 | 0 | 100.00% | 原 14 个实现及 11 个显式 HIP 布局变体 |
| NMS（`nms`） | 1 | 1 | 0 | 0 | 100.00% | IoU ballot 与单 block 顺序消除 |
| 原版 HGEMM（`hgemm`） | 36 | 11 | 14 | 11 | 30.56% | 2 个 BLAS、9 个 WMMA 已通过；11 个同步变体精度未验收；14 个其他变体待迁移 |
| WS-HGEMM（`ws-hgemm`） | 1 | 0 | 1 | 0 | 0.00% | 原 CuTe 实现尚未迁移 |
| 原版 FlashAttention（`flash-attn`） | 29 | 2 | 27 | 0 | 6.90% | shared-KV / shared-QKV 的 FP32 累加原入口通过；独立 AMD 前向 API 另列 |
| Merge Attention States | 1 | 1 | 0 | 0 | 100.00% | FP32/FP16/BF16，含空状态、可选 LSE 和设备／stream 检查 |
| **合计：25 个扩展模块** | **229** | **168** | **45** | **16** | **73.36%** | 其中 10 个明确提高累加精度；6 个独立 AMD 计算接口另列 |

“全部接口通过”也仅表示通过现有测试集，仍须遵守原实现的 dtype、形状、连续性、
对齐和 tile/pack 限制。已支持的基础模块调用入口为 `leetcuda_rocm.load("模块名")`。

## 3. 新增的 AMD 原生实现

| 算子 | 公共接口 | 已支持功能 | 当前限制 |
| --- | --- | --- | --- |
| FP16/BF16 GEMM | `leetcuda_rocm.gemm` | FP32 累加、AMD WMMA、转置及 strided 输入、M/N/K 尾部、可选预分配输出 | 二维矩阵乘、正尺寸、前向计算；不提供 backward |
| FlashAttention 前向 | `leetcuda_rocm.flash_attention` | FP16/BF16、FP32 累加、AMD WMMA、在线 softmax、因果/非因果、strided 输入；非因果模式支持 NQ≠NK | D 仅支持 16/32/64/128；因果模式要求 NQ=NK；无 GQA、dropout、任意 mask、backward |
| FP8 GEMM 完整链路 | `leetcuda_rocm.fp8_gemm` | BF16 → E4M3 动态量化 → FP8 WMMA/FP32 累加 → BF16；四种 scale 模式；普通／producer-consumer 调度 | 连续 BF16；M>0、N%8=0、K%16=0；FP32 scale/inverse 必须处于正常有限范围；性能尚未收敛 |
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
| 原版 `kernels/hgemm` | 2 个 BLAS、9 个 WMMA 原接口通过 | 11 个同步 FP16 累加变体精度未验收；14 个其他绑定待迁移，WGMMA 等未绑定实现也未覆盖；独立 AMD GEMM 已可用 |
| 原版 `kernels/ws-hgemm` | 未移植 | Warp specialization HGEMM 实现 |
| 原版 `kernels/flash-attn` | 2 个 FP32 shared-KV/QKV 原接口通过 | 其余 27 个变体待迁移，包括 tiling、FP16 累加、split-KV、swizzle 和 CuTe；已有独立 AMD FlashAttention 前向 API |
| SGEMM 其他变体 | 14/17 原接口支持 | 3 个异步接口已用 AMD VGPR 预取流水线实现；TF32 WMMA 与 cuBLAS TF32 数学模式仍待迁移 |
| HGEMV 的 CuTe 变体 | 矩阵版已支持，另外 2 个精度未验收 | 矩阵版明确使用 FP32 WMMA 状态；标量/packed 与 CUDA 逐位一致但长 K 精度不足 |
| Matrix Transpose 的 CuTe 变体 | 已支持 | 11 个接口已用显式 HIP 布局实现；CUDA CuTe 源码保留 |
| Reduce Sum 的 BF16 自身累加变体 | 验证未通过 | 下文列出的 3 个变体因数值误差超标被排除 |
| FP8/FP4 GEMM、FFPA 扩展 | FP8/FP4 功能通过，整体未完成 | E4M3 与 NVFP4 完整链路、离线 B、Python 和专用 C++ benchmark 已接入；性能仍落后库基线；E5M2 完整链路、FFPA 待迁移 |
| 新增 Attention 的扩展功能 | 部分支持 | backward、GQA、dropout、任意 attention mask、其他 head dimension、NQ≠NK 的因果模式 |
| `kernels/cutlass`、CuTe DSL 示例 | 未移植 | NVIDIA 库、指令及对应编译入口 |
| `kernels/swizzle` 示例 | 未移植 | 独立布局、矩阵乘和转置示例 |
| `kernels/openai-triton` 原有示例 | 四个已验证，性能待优化 | vector-add、fused-softmax、layer-norm（含梯度）、merge-attn-states 两张 AMD 卡与 5090 各 195 项通过；不计入 229 个扩展接口 |
| `kernels/interview` 的 GPU 算子与 C++ benchmark | FP8/FP4 局部接入 | 两个头文件增加 AMD 分支，有专用 C++ 构建入口；原 `build.sh` 多模块 benchmark 体系仍未整体接入 |
| 原目录 Python 性能脚本及 NVIDIA 工具示例 | 未整体适配 | CUDA 专用构建参数、依赖和 Nsight 等工具流程 |
| 其他 AMD GPU 架构 | 尚未验证 | 当前实现只对 gfx1201 提供资格声明，不能直接外推至 MI 系列／wave64 GPU |

BF16 自身累加的 3 个排除项：

- `block_all_reduce_sum_bf16_bf16`
- `block_all_reduce_sum_bf16x2_bf16`
- `block_all_reduce_sum_bf16x8_pack_bf16`

它们未通过当前数值误差门限，未计入已通过的 168 个原始接口。BF16 输入可使用
名称以 `_f32` 结尾、以 FP32 累加的对应版本。

不在 229 个 Python 扩展入口分母内的内容仍需要独立迁移。其中主要源码范围如下，
这里按**实现文件**计数，与接口数不相加：

| 补充范围 | 实现文件数 | 当前原因与状态 |
| --- | ---: | --- |
| `interview` 的 GPU kernel 头文件 | 8 | FP8/FP4 两个头文件核心路径已有 AMD 分支及专用 C++ 检查／性能报告；其余基础算子、SGEMV、SGEMM、HGEMM、FlashAttention、FFPA 头文件与整套原构建仍待接入 |
| `openai-triton` 的 Triton 实现 | 4 | vector-add、fused-softmax、layer-norm（含反向）、merge-attn-states 已接入：AMD 两卡与 5090 各 195 项检查通过，108 组输出对照通过；性能仍待优化。CUDA 版 merge-attn-states 的 1 个绑定已通过验证并计入 229，不重复计入本行 |
| `cutlass`／CuTe DSL GPU 示例 | 4 | 1 个 CuTe vector-add 源文件、3 个 CuTe DSL SGEMM 示例；需替换 NVIDIA 后端与构建方式 |
| `swizzle` GPU 示例 | 3 | 独立矩阵乘／转置／布局实现，尚未迁移和验证 |
| `nvidia-nsight` GPU 示例 | 2 | 示例算子代码尚未迁移，Nsight 分析流程也需要换成 AMD 工具 |

其他未绑定的 CUDA 实现（例如 HGEMM 的 WGMMA 源文件）也没有被 229 的分母覆盖。
上述补充文件数用于标明范围，不声称是全仓独立 GPU kernel 函数的总数。
没有实现内容的 `kernels/transformer` 占位目录不计为算子。

FP8 硬件能力先导实验：E4M3/E5M2 两种格式 × 3 个形状，**6/6 项通过**，
确认实际 FP8/BF8 WMMA。仅覆盖预量化输入，不增加公开接口通过数；
见 [FP8 能力报告](fp8_capability_20261103.md)。
后续 E4M3 完整链路的验收见 [FP8 迁移报告](fp8_pipeline_20261103.md)，
不能将 E5M2 的先导探测写成完整 E5M2 量化 API 已适配。

## 5. 验证依据与复现入口

| 验证项 | 结果 | 证据 |
| --- | --- | --- |
| NVFP4 完整链路，Python／C++ | GPU1 1,540 项、GPU0 1,541 项、大尺寸 40 项、C++ 263 项通过；Python/C++ 各 96 组性能比较，仍落后 BF16 库基线 | [FP4 迁移报告](fp4_pipeline_20261103.md) |
| FP8 完整链路，Python／C++ | GPU0/GPU1 各 753 项通过；独立 C++ 143 项通过；Python/C++ 各 48 组性能比较，性能仍落后库基线 | [FP8 迁移报告](fp8_pipeline_20261103.md) |
| 本轮基础算子，PCI 0000:e3:00.0，2026-11-05 | 24 模块、168 接口、927 项检查通过 | [当前基础报告](qualification_20261105_flash_shared.json) |
| 本轮 FlashAttention 专项 | 534 项、18 个 WMMA 特化；与 CUDA 146/146 对照通过 | [专项报告](flash_shared_20261105.md) |
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
2026-11-02 的 21 模块报告保留为历史，当前基础资格以 2026-11-05 第三轮的 24 模块、168 接口、927 项报告为准。

在项目根目录使用锁定的 ROCm 10 启动器运行：

```bash
cd /home/zihaomu/zihaomu_workspace/leetcuda_workspace/LeetCUDA

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

默认选择物理 GPU0，运行前确认其空闲；物理 GPU3 禁止用于计算。
环境身份、源码 hash、数值容差和逐项结果以各报告为准。
更多接口约束见 [ROCm 使用说明](README.md)，
运行环境见 [环境锁](../../scripts/rocm/environment.lock.json)。
