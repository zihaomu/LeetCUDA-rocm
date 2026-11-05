# CuTe、CuTeDSL、swizzle及profiling：24个GPU定义验收

2026-11-06。补充的24个GPU定义均已在ROCm10.1.0/gfx1201完成功能迁移。它们与原229个绑定接口、interview的53个定义分别统计，不能相加称作去重算子数。

| 范围 | 定义数 | 双AMD每卡 | 5090检查 | 共同输出 |
| --- | ---: | ---: | ---: | ---: |
| CuTe向量 + profiling + transpose | 1+12+3 | 254 | 224 | 168/168逐位一致 |
| CuTeDSL SGEMM | 3 | 158 | 72 | 66/66逐位一致 |
| swizzle HGEMM | 5 | 403 | FP32参考198 | 198/198，其中44逐位一致 |

最终格式整理后三组再次分别254/158/403通过。统一独立HIP可执行文件为这24定义提供100项直接调用、Graph和canary检查，不链接Torch。全部9个独立C++组共2907项通过。

## 实现对应关系

- **CuTe向量**：每线程8/16个相邻half，128位读写；AMD有界尾部/未对齐回退。原表达式两端实际融合为两次FP16 FMA，按FP64计算每次FMA再舍入half的独立参考逐位验收。初轮逐操作half舍入参考错误已归档，没有放宽门限。
- **profiling 12定义**：保留原elementwise_add和ReLU的scalar、f32x4、half、half2、manual-half8及packed-half8计算体。HIP缺少`__hmax2`，显式采用两个half max；packed版本保留128位访存。无guard原half8使用精确线程数，对不满足向量长度/对齐的输入明确拒绝。
- **transpose 3定义**：32×32 plain、padding33和XOR LDS布局，AMD以32×8线程每线程4行加载，统一barrier与有界tail。原CUDA的条件barrier不适合尾块，CUDA只测完整tile，AMD尾块单独对CPU验证。
- **CuTeDSL 3定义**：naive保留FP32累加后FP16舍入再存FP32；mma_atom保留128×128×8 SIMT、每线程64个FP32输出和padding4；pipelining保留三级无paddingLDS、提前2块载入及寄存器预取。这是对应算法的独立HIP后端，原算法本身是SIMT，不宣称使用矩阵单元或移植NVIDIA DSL编译器。原三个Python示例在HIP上已有实际运行入口。
- **swizzle 5定义**：普通/异或A、16×8与128×128分块、LDS或寄存器输出分别实现，B保持NN `[K,N]`；大块AP/BP=0/8形成11个实际特化。AMD WMMA16×16将小块多余8列置零，只存8列，是明确的矩阵几何调整。5条均显式升级FP32累加。

原swizzle CUDA198项有66项精度失败，包括长K和状态控制；同源码仅升级累加器的FP32参考198项全部通过，原失败保留。11个精度控制要求结果1.03125，明确区分原FP16运行状态1.0。
HGEMM门限针对FP64参考舍入到FP16的结果，atol=0.01、rtol=0；原始FP64误差另外记录。不能将输出量化误差隐藏在“FP64误差≤0.01”的笼统说法中。

## 验证及资源

三组CPU冻结夹具分别覆盖向量长度/对齐/尾部、转置非整tile、SGEMM/HGEMM两种种子与K4096、抵消/舍入/零值、非法dtype/alias/device、非默认stream和NaN投毒Graph。
254=198数值/Graph+56边界；158=105数值/Graph/状态+41边界+12流水线；403=330数值/Graph+71边界+2布局，额外11条状态诊断复用这些输出，不重复计入独立检查数。

17个profile/vector/transpose、3个DSL、11个swizzle编译特化均**零scratch/spill**。DSL的VGPR11/222/225，LDS0/8448/24576B；swizzle VGPR30或107，LDS768–10496B。ISA分别确认FP16 packed FMA、FP32 dual-FMAC、128位读写以及实际WMMA；SIMT不误记为WMMA。

CUDA CuTeDSL计时首轮闭包固定了旧stream，目标计算不在测量Graph中，43–109倍结果无效并已冻结。改为每次取得current stream，新增每种variant一条跨capture stream NaN投毒检查后，最终72项与12组性能重新通过。最终数字只使用修正后报告。

| 性能范围 | 配置数/端 | AMD同机基线/本实现 | 5090同机基线/本实现 |
| --- | ---: | ---: | ---: |
| profile/vector/transpose | 17 | 0.313–7.935 | 0.564–3.403 |
| CuTeDSL | 12 | 0.360–1.023 | 0.0127–1.769 |
| swizzle FP32参考 | 20 | 0.125–1.194 | 0.033–1.375 |

基线依次是对应Torch表达式、禁用TF32的torch.mm及FP16 torch.mm。区间仅表示已测workload，不能据此宣称整体超过成熟库。

## AMD profiling流程

```bash
HIP_VISIBLE_DEVICES=6 scripts/rocm/profile_examples.sh artifacts/rocm/profile_trace
HIP_VISIBLE_DEVICES=6 scripts/rocm/profile_examples.sh artifacts/rocm/profile_waves counters
```

rocprofv3实测12核各20次，240个目标dispatch，记录时间、LDS、scratch和寄存器。SQ_WAVES两次采样均精确匹配每种向量宽度的理论wave数量。
SQ_INSTS_VALU、SQ_INSTS_WAVE32_VALU、SQ_INSTS_SALU、SQ_INSTS_SMEM虽被环境列出却全返回零，**该四类计数器未验收**；保留原始失败，不能解释为没有执行指令。已验证的counter入口仅采集SQ_WAVES，指令类型使用静态ISA佐证。Nsight工具本身没有移植。

5090 CuTeDSL4.4.2放在项目私有缓存，依赖与wheel哈希归档；[官方安装说明](https://docs.nvidia.com/cutlass/4.4.2/media/docs/pythonDSL/quick_start.html)。未修改共享Torch环境。

## 使用入口与证据

```python
from leetcuda_rocm.supplemental import build
lib, metadata = build("profile")  # 另有vector、transpose
from leetcuda_rocm.cute_dsl import matmul
from leetcuda_rocm.swizzle_examples import native
swizzle = native()
# FP16 A[M,K], B[K,N], out[M,N]; kind 0..4; padding for kind 1/3
swizzle.forward(a, b, out, 3, 8, 8)
```

具体签名/合法布局以对应模块为准。swizzle原CUDA main、Nsight CLI、CuTeDSL编译器及原book CUDA构建命令不声明为HIP兼容；提供已验收的原生入口和统一构建映射。

```bash
HIP_VISIBLE_DEVICES=6,2 scripts/rocm/run.sh scripts/rocm/qualify_supplemental.sh
HIP_VISIBLE_DEVICES=6 scripts/rocm/build_all_bench.sh --output artifacts/rocm/all_cpp
```

[全部来源审计102项](supplemental_examples_20261106/source_audit.json)、[CUDA源码/二进制审计223项](supplemental_examples_20261106/cuda_archive_audit.json)、[profiling结论](supplemental_examples_20261106/supplemental/profiling_validation.json)、[31章映射](supplemental_examples_20261106/book_mapping.json)。各组完整JSON、失败报告、源代码快照位于同名子目录；大夹具、输出、二进制及ISA在两端专用artifacts保留。
