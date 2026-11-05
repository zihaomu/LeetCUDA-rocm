# U2：三个 BF16 Reduce 精度迁移

日期：2026-11-06。新增三个原 BF16 累加入口后，原接口 **218/229（95.20%）**，
Reduce **20/20**。剩余11个同步SIMT HGEMM；整体补充范围和性能工作继续。

## 实现与精度约定

`block_all_reduce_sum_bf16_bf16`、`block_all_reduce_sum_bf16x2_bf16`、
`block_all_reduce_sum_bf16x8_pack_bf16` 保留1/2/8元素读取pack和原合法thread-count派发。
输入BF16、输出FP32[1]；明确将原BF16部分和升级为**补偿FP32**主值及残差。
wave32和block均合并两个状态，第二级kernel按固定次序合并block结果，替代无序全局atomicAdd。
未对齐和尾部有界读取，空输入返回0；非有限值分支保留单个无穷及NaN传播。
使用当前stream、设备guard，检查连续BF16、设备、grad及32位索引上限。

原BF16 kernel留作私有control；精度升级不声称逐位复现原舍入。
FP32累加也有有限动态范围，未声称对任意可溢出的输入实现精确数学求和。

## 验收

| 项目 | 结果 |
| --- | --- |
| 两张AMD卡 | 各 **252/252**：36共享输入、180边界/运行时检查、36原BF16 control |
| 升级路径精度 | FP64独立sum参考，atol=0.0002、rtol=0.00002 |
| 原版门限 | 保留atol=0.15、rtol=0.01，与升级路径分开报告 |
| 原BF16 control | 36/36匹配CPU逐次舍入模型，其中6组未过FP64精度门限 |
| 5090原版 | 30/36过参考；6组失败完整保留 |
| 专项跨平台 | 两端参考均通过的30组全部在原门限内一致 |
| 全库AMD | **25模块、218接口、1185/1185** |
| 全接口共同输入 | **440/440**两端参考与跨平台比较通过 |
| ISA | 18个pack/thread特化，VGPR14–15、LDS8–256B，零spill/scratch；x8真实128-bit global load |

覆盖两个seed、百万元素、标量/空/非对齐/尾部、6种thread派发、输入不变、
20次确定性复验、8条并发stream各4次、第二设备及guard。
特别测试2^24与1、2^30与2^-10的抵消残差，并验证BF16最小subnormal求和结果精确一致。
原x2/x8在抵消输入中丢失残差，最大误差8192/6144；新实现通过严格FP32门限。
另覆盖单个正负无穷、混合无穷、NaN，不能把非有限预期值当作普通有限精度通过。

AMD主卡物理4/PCI83/HIP6，副卡物理5/PCIa3/HIP7；锁定ROCm10.1/HIP7.16环境。
物理GPU3未参与计算。5090固定镜像，专项使用original快照，全库对照使用独立ported工作树。
所有fixture、源码、二进制和输出hash在报告中记录；ISA二进制另存冻结副本。

## 性能

graph每次8调用、20ms×5组交替采样，与同卡Torch.sum(dtype=FP32)比较。
9个配置均先验证精度和填NaN后的graph replay。

| pack | 形状 | AMD ms | Torch/AMD |
| --- | --- | ---: | ---: |
| 1 | 8×1024 | 0.006832 | 0.6122× |
| 2 | 8×1024 | 0.006746 | 0.6212× |
| 8 | 8×1024 | 0.007004 | 0.5987× |
| 1 | 128×1024 | 0.021938 | 0.3708× |
| 2 | 128×1024 | 0.019190 | 0.4229× |
| 8 | 128×1024 | 0.007166 | 1.1303× |
| 1 | 1024×1024 | 0.029450 | 0.3053× |
| 2 | 1024×1024 | 0.017092 | 0.5244× |
| 8 | 1024×1024 | 0.011357 | 0.7876× |

x8中等规模约1.13倍，其余被测配置仍慢于Torch；补偿计算及两次launch有开销，功能通过不表示性能已收敛。
初始原型的HIPify同行花括号编译失败及修复前源码保留在实验目录，未计为成功实验。

## 证据与复现

- [主卡](reduce_precision_20261106/hip.json)、[第二卡](reduce_precision_20261106/hip_device1.json)
- [原CUDA](reduce_precision_20261106/cuda.json)、[专项比较](reduce_precision_20261106/comparison.json)
- [全库](reduce_precision_20261106/qualification218.json)、[440组比较](reduce_precision_20261106/qualified218_comparison.json)、[源码审计](reduce_precision_20261106/source_audit.json)

```bash
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.reduce_precision_check fixtures --fixtures artifacts/rocm/continuation_all/u2/fixtures.pt
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.reduce_precision_check run --fixtures artifacts/rocm/continuation_all/u2/fixtures.pt --output artifacts/rocm/continuation_all/u2/hip.json --performance
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.reduce_precision_check run --device 1 --fixtures artifacts/rocm/continuation_all/u2/fixtures.pt --output artifacts/rocm/continuation_all/u2/hip_device1.json
```

复验前核对占用、预约及实际PCI；大fixture、输出与冻结二进制保存在两端专用artifacts目录。
