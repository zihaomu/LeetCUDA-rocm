# U1：两个标量/packed CuTe HGEMV 精度迁移

日期：2026-11-06。新增 `hgemv_f16_cute`、`hgemv_f16x8_cute` 后，
原接口 **215/229（93.89%）**，HGEMV **6/6**；剩11个HGEMM及3个BF16 Reduce精度工作包。

## 算术、布局与契约

两个接口保留FP16输入/输出，**显式将乘加、lane状态与wave归约提升到FP32**。
这不是逐位复现原FP16累加。历史FP16源码、独立舍入模型及失败结论保留。
每CTA128线程、四个wave，每wave负责一行输出：

- scalar：每lane处理 `lane+32*i` 的K元素，运行FP32 FMA，末尾XOR树归约。
- packed：每lane处理连续8元素，128-bit读取A/B；每tile保留even/odd两个局部累加器，
  合并到运行状态后做wave归约。未对齐地址和尾部有界读取；没有别名到scalar入口。
- A[M,K]、B[K,1]、C[M,1]，连续FP16同设备；当前stream和device guard。
- 空M/K、任意尾部、未对齐连续A/B支持；dtype/rank/shape/stride、别名、grad、混设备和索引上限检查。

## 验收

| 检查 | 结果 |
| --- | --- |
| 两张AMD专项 | 各 **150/150**：30共享输入、90边界/精度状态/stream/契约、30原FP16舍入control |
| AMD门限 | 原门限atol=rtol=0.003，FP64参考舍入FP16，未放宽 |
| 原FP16 control | 30/30精确匹配独立CPU逐次舍入模型；其中11/30未过FP64精度门限 |
| 5090原版 | 19/30过参考；scalar 9/15、packed 10/15 |
| 5090仅初始化修复 | 仍19/30过参考；本fixture的30个输出与原版逐位相同 |
| 专项跨平台 | 原版、修复版中两端参考均通过的19组，各自 **19/19** 比较通过 |
| 全库AMD | **25模块、215接口、1179/1179** |
| 全接口共同输入 | **434/434**两端参考及跨平台比较通过 |
| ISA | scalar VGPR9、1个静态FP32 FMA；packed VGPR29、8个FP32 FMA及2个128-bit global load；均零LDS/spill/scratch |

覆盖两种seed、K128到65536、AMD尾部65537、M4097、零维、zero/抵消/互逆尺度、
未对齐A/B、输入不变、输出canary、8条stream各4次排队、第二设备/guard和FP32状态区分构造。
主卡物理GPU4/PCI83/HIP6，第二卡物理GPU5/PCIa3/HIP7，锁定ROCm10.1/HIP7.16与项目Torch/Triton。
物理GPU3未参与计算。5090使用固定镜像、original快照和单独ported工作树。

原CUDA长K误差没有被修改报告：scalar最大约0.32985，packed约0.10688；
scalar在一个K1024输入就超门限，两个变体的互逆尺度输入也有失败。
AMD原FP16 control精确匹配舍入模型且同样有11个精度失败，新FP32路径全部通过，明确区分原误差与迁移缺陷。

另修复原CUDA packed kernel每tile的 `sum_half2` 未初始化：只增加 `clear(sum_half2)`，
保留原FP16计算与布局。当前fixture原版恰好得到相同输出，不声称该初始化修复提高了这些输入的精度。
CUDA kernel发射仅适配current stream以便graph验证；完整源码/hash分别记录。

## 性能

每graph8调用，20ms×5组交替采样。AMD6项均先过数值门限，并通过填NaN后graph replay验证。
同一卡当场测Torch.mm；不把5090/AMD时间比当作算法加速。

| 接口 | M×K | AMD耗时ms | Torch/AMD |
| --- | --- | ---: | ---: |
| `hgemv_f16_cute` | 64×128 | 0.003091 | 1.4181× |
| `hgemv_f16x8_cute` | 64×128 | 0.002949 | 1.5002× |
| `hgemv_f16_cute` | 1024×1024 | 0.006549 | 1.2973× |
| `hgemv_f16x8_cute` | 1024×1024 | 0.004211 | 2.0188× |
| `hgemv_f16_cute` | 4096×8192 | 0.086271 | 0.9064× |
| `hgemv_f16x8_cute` | 4096×8192 | 0.055178 | 1.4191× |

packed在三个被测规模均快于Torch；scalar的大规模约0.906×，仍待优化。
结论限于这三个工作负载、该设备与采样方法，不代表所有形状都更快。
CUDA性能中未过精度的输入单列，原始时间保留，不能作为合格性能结果宣传。

## 证据与复现

- [主卡](hgemv_vector_20261106/hip.json)、[第二卡](hgemv_vector_20261106/hip_device1.json)
- [原CUDA](hgemv_vector_20261106/cuda_original.json)、[仅初始化修复](hgemv_vector_20261106/cuda_fixed.json)、[输出对比](hgemv_vector_20261106/initialization_comparison.json)
- [专项比较](hgemv_vector_20261106/comparison.json)、[全库](hgemv_vector_20261106/qualification215.json)、[434组比较](hgemv_vector_20261106/qualified215_comparison.json)、[源码审计](hgemv_vector_20261106/source_audit.json)

```bash
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.hgemv_vector_check fixtures --fixtures artifacts/rocm/continuation_all/u1/fixtures.pt
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.hgemv_vector_check run --fixtures artifacts/rocm/continuation_all/u1/fixtures.pt --output artifacts/rocm/continuation_all/u1/hip.json --performance
HIP_VISIBLE_DEVICES=6,7 scripts/rocm/run.sh python -m leetcuda_rocm.hgemv_vector_check run --device 1 --fixtures artifacts/rocm/continuation_all/u1/fixtures.pt --output artifacts/rocm/continuation_all/u1/hip_device1.json
```

复验前核对设备占用与预约；所有大fixture、输出及冻结二进制在两端专用artifacts目录。
