# 独立 interview HGEMM 五路径验收

2026-11-06。**五种AMD调度实现功能验收完成：两卡各477项通过；336项跨平台参考比较通过。**
其中252项对应四种CUDA源码调度的显式FP32参考，84项对应无法在5090执行的WGMMA路径的cuBLAS数学参考。
不将这84项称为原WGMMA运行，也不加入原229个Python接口分母。

## 路径与硬件映射

| Kind | 原入口 | AMD实现 |
| --- | --- | --- |
| 0 | hgemm_mma_stages_tn | 128×128×16 CTA、8 waves、plain TN LDS、多stage VGPR预取 |
| 1 | hgemm_mma_stages_tn_swizzle | BK64、每K16片段原XOR地址置换、真实双份寄存器ping-pong；stage2为128×128，stage3/4缩成64×64以保留stage深度 |
| 2 | hgemm_mma_stages_tn_cute | CuTe Swizzle<3,3,3>坐标变换、128×128×32、8 waves、16行C条带复用LDS epilogue；采用原CuTe FP32配置的BN128，K循环保留以控制寄存器 |
| 3 | hgemm_wgmma_stages_tn | 软件wavegroup：1个producer wave+4个consumer waves，64×128×32，多槽acquire/release序号；实际FP16 WMMA |
| 4 | hgemm_tma_mma_ws_tn | 4个producer+4个consumer waves，各producer独立发布、consumer等待全部producer；stage2为64×128×64，stage3/4为64×128×32 |

B以连续`[N,K]`的BT存储，C=A×BTᵀ。每种配置使用真实128-bit全局读取和FP16 WMMA，不使用空操作TMA/WGMMA替身。
AMD最大LDS为64KiB，因此部分tile缩小；保留stage2/3/4的实际槽数及GEMM计算分工，不把缩小tile解释为原CUDA launch ABI兼容。
CuTe的CUDA4 warps变为AMD8 waves；原CuTe FP16 BN256配置不作为当前FP32资格路径。

所有正式AMD计算采用**FP16输入/输出、FP32累加**。对原FP16累加路径属于明确精度升级；CuTe使用其已有的FP32累加选项作CUDA参考。
另有FP16状态诊断：每K16完成后舍入FP16，验证状态语义与升级后的差异，不声称它是AMD原生FP16累加指令或已完整复现NVIDIA内部加法顺序。

## 正确性与原CUDA结果

| 检查 | 结果 |
| --- | --- |
| AMD物理GPU4/PCI83 | 477/477 |
| AMD物理GPU0/PCI03 | 477/477 |
| 5090原精度源码调度+WGMMA数学参考 | 228/336；108项原低精度误差失败保留 |
| 5090显式FP32参考+WGMMA数学参考 | 336/336 |
| AMD与FP32 CUDA参考比较 | 336/336在容差内；整份输出逐位相同0项 |
| 原WGMMA编译探测 | ptxas明确拒绝sm120上的wgmma.mma_async/commit_group/wait_group |

每张AMD的477项由336项共同形状输出/Graph、60项尾部/非对齐输出/Graph、60项FP16/FP32状态输出/Graph、15项布局双射、6项流水线计数组成。
流水线用延迟consumer验证stage2/3/4各自实际领先深度；全部producer与consumer完成12个tile，timeout=0，输出同时对独立参考验证。
FP16和FP32状态诊断均逐位符合各自预期；不是依靠较宽误差门限区分。

共同shape覆盖M128/256、N256/2304、K64/128/192/512/1024/2048及block swizzle。
使用同一冻结FP32文件、两个固定偏移，转换为实际FP16输入后做CPU FP64参考；较长矩阵按文件长度循环取值，B逆序取值。
原swizzle的stage3预加载要求至少两个BK64 tile，因此其共同短K测试为128；AMD另外覆盖K17/K80和M/N尾部。
所有矩阵沿用**绝对误差0.01、rtol=0**，未为108项原CUDA失败放宽门限。
原CUDA失败来自四种FP16累加路径，最大误差随K增长；AMD与明确FP32参考均通过。

CUDA FP32参考只升级三种手写MMA的累加器及最后FP16打包，CuTe选择原FP32选项；保留原cp.async/TMA、swizzle、register ping-pong和epilogue。
CuTe宿主包装增加测试stream参数适配；这些参考改动及完整staged头文件有SHA和源码审计，不静默改写上游快照。
WGMMA原指令在5090硬件不可用，数学参考是明确标记的cuBLAS FP32累加，不用于证明NVIDIA WGMMA调度复现。

## 资源与计时

36个特化（15个FP32、15个状态control、6个流水线audit）均有实际WMMA和128-bit读取，**scratch/spill全部为0**。
初版CuTe有16～564B scratch；原候选、失败编译日志和报告保留。调整BN和K循环后复验全部477项与跨平台比较。

| FP32路径 | VGPR范围 | LDS范围 | scratch/spill |
| --- | --- | --- | --- |
| plain TN | 123～123 | 16384～32768B | 0 |
| TN swizzle/ping-pong | 181～226 | 49152～65536B | 0 |
| CuTe layout/epilogue | 201～202 | 32768～65536B | 0 |
| software wavegroup | 209～209 | 24616～49232B | 0 |
| four-producer WS | 148～187 | 36960～49280B | 0 |

主卡记录228组普通HIP event计时，CUDA每个参考版本记录168组；5次预热、100次调用，包含提交间隙。
这些是当前C++路径的耗时记录，没有宣称超过本机成熟库，也没有把跨GPU耗时比当成优化倍数；原精度失败项的计时仅供诊断。
M7仍需统一的同机基线与固定性能门槛验收。

## 构建与边界

原`interview/hgemm.cuh`增加HIP分支，C++调用入口为`leetcuda_interview_hgemm::launch<Kind,Stages>`，显式传stream；与原CUDA kernel launch参数不作ABI兼容声明。
当前资格为gfx1201，FP16连续A/BT/C、合法正尺寸及32位索引范围、同一设备、输出不重叠输入；caller负责选择设备与分配存储。
支持所测尾部及FP16标量对齐，向量读取在不满足16字节对齐时走有界标量加载。其他架构未验收。

```bash
HIP_VISIBLE_DEVICES=6,2 scripts/rocm/run.sh python -m leetcuda_rocm.interview_hgemm_check run \
  --backend hip --performance \
  --fixtures artifacts/rocm/continuation_all/interview/fixtures.bin \
  --output artifacts/rocm/continuation_all/interview_hgemm/recheck.json
```

非默认stream、capture后NaN重置、输出前后canary；执行文件不链接Torch/c10/Python。
执行前核对占用、PCI和预约，物理GPU3禁止计算。

- [AMD主卡](interview_hgemm_20261106/hip_final.json)、[第二卡](interview_hgemm_20261106/hip_final_second.json)
- [原精度CUDA](interview_hgemm_20261106/cuda_original.json)、[明确FP32参考](interview_hgemm_20261106/cuda_fp32.json)、[336项比较](interview_hgemm_20261106/comparison.json)
- [资源](interview_hgemm_20261106/resources_final.json)、[源码审计](interview_hgemm_20261106/source_audit.json)、[WGMMA实际编译失败](interview_hgemm_20261106/wgmma_probe.log)

interview功能45/53（84.91%）；44条路径有CUDA源码参考，另1条WGMMA有CPU及5090数学参考。剩8个独立FlashAttention kernel、其他补充源码及M7继续执行，整体迁移未完成。
