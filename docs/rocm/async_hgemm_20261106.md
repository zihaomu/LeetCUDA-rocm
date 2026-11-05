# B5：三个异步 HGEMM 验收

日期：2026-11-06。范围是原始 229 个计算接口中的三个异步 SIMT HGEMM。
状态：**B5功能验收完成，209/229（91.27%）**；双卡专项、全库及422/422组跨平台对照通过。性能待优化。
整个迁移仍继续，B6/B7、16 个精度工作包与整体补充范围尚未关闭。

## 实现和契约

三种入口保留 128×128 CTA、BK16/BK32、8×8/16×8 线程微分块。
在 AMD 上使用 128-bit global load 到 VGPR，当前 tile 的 FP32 FMA 执行完后再
wait/publish 下一个 tile 到另一 LDS 槽；两个 LDS 槽真实交替。采用 stride136
的转置 A 与行主 B，非对齐输入和尾部由有界读取处理。
原 CUDA 的 `cp.async` 搬到共享内存改为 AMD 寄存器预取和 LDS 发布。

**FP16 输入输出、FP32 累加是明确的精度升级**。原 CUDA 每步 FP16 FMA
的误差不会被隐藏；没有把本实现当作原 FP16 舍入的逐位模拟。原 CUDA 每线程
连续小块在 AMD 上改为分散到四个象限，微分块元素数及每个输出 K 次序保留。
本批是 SIMT 路径，不宣称使用 WMMA；ISA 使用 `v_fma_mix_f32`。

- 输入 A[M,K]、B[K,N]、输出 C[M,N]，同设备、连续 FP16；前向接口。
- M/N/K 尾部、未对齐连续视图、空 M/N 与 K=0 支持；K=0 写零。
- 拒绝 dtype/rank/shape/stride 不匹配、输出与输入共享存储、requires_grad、混用设备及索引/网格超限。
- 当前 device guard 与 PyTorch current stream；非默认及并发 stream、graph replay 均检查。
- 私有同步调度 control 保持相同算术，仅把预取改为阻塞加载；不加入原接口分母。

## 环境与正确性

锁定 ROCm10.1.0/HIP7.16、Torch2.14.0+rocm10.1.0、gfx1201；镜像和完整运行身份在 JSON。
HIP mask 6,5 对应物理 GPU4 PCI83 / GPU6 PCIc3；物理GPU3禁止计算，物理GPU5当时被预约。
5090 使用项目专用目录、锁定 CUDA 镜像和原始快照 `3dbd618e703943e6761db05f29dd66cb7de94d13`。

| 验收 | 结果 |
| --- | --- |
| AMD 主卡与第二张卡 | 各 **216/216**：45共享输入、126边界/设备/stream/精度状态、45同步control位一致 |
| 精度 | 保持 `atol=rtol=0.01`；独立 FP64 参考，有限输出、canary |
| 共享输入 | 15组CPU冻结输入×3入口；两种种子，K到4096，额外K32/64/96/544/1024 |
| AMD 边界 | 两种种子，奇数M/N/K、K8193、空维度、抵消、互逆尺度、未对齐、非法契约、8条并发stream |
| 累加状态 | 构造输入精确验证FP32状态16.125；原FP16累计停留16.0 |
| 全接口跨平台 | 209接口、422/422组两端参考及比较通过 |
| 全库AMD | 24模块、209接口、**1109/1109** |
| 性能与graph | 主卡9配置；输出填NaN后replay恢复，精度与同步control位一致 |

原版 CUDA 45组：16通过、29失败。两个K32异步入口末尾固定读取slot1，
在K32/96/160/544（奇数tile数）额外产生8组明显错误；只修复末尾槽后，
修复版24通过、21数值失败。FP16每步FMA在K256及更长K已经会超门限，
不是只在K4096失败。原始数据保持不变，修复版以 `ported` 来源单独报告。

原版/AMD双方参考通过的16组、修复版/AMD双方参考通过的24组跨平台比较全部通过。
其他组保留跨平台输出差异，不宣称全45组原版精度通过。

## 指令和资源

| 微分块/BK | VGPR | LDS bytes | scratch bytes | 预取指令次序 |
| --- | ---: | ---: | ---: | --- |
| 8×8/K16 | 98 | 17408 | 0 | global_load → FP32 FMA → wait |
| 8×8/K32 | 114 | 34816 | 0 | 同上 |
| 16×8/K32 | 212 | 34816 | 0 | 同上 |

三种特化分别64/64/128条静态FMA。静态ISA确认编译后的加载/计算/wait次序；
这不是按周期测得访存完全被隐藏。完整元数据与HSACO/hash见主卡报告。

## 性能

以下每格为 **同步control时间/预取时间；Torch.mm时间/预取时间**。
同卡同次、每graph8调用、每组20ms、5组样本，完整原始样本保留。

| 接口 | 256³ | 1024³ | 2048³ |
| --- | --- | --- | --- |
| `hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async` | 1.0946× / 0.1970× | 1.0273× / 0.1882× | 1.0010× / 0.1624× |
| `hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async` | 1.0575× / 0.1938× | 1.0426× / 0.1782× | 1.0516× / 0.1574× |
| `hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async` | 1.0121× / 0.1447× | 1.0011× / 0.1705× | 1.0089× / 0.1540× |

预取测得约1.001–1.095倍，1%以内的差异不视为稳定收益。
所有配置仍落后同机Torch库；本批关闭功能迁移和性能测量，**不宣称性能已收敛**。
5090原版另有9条计时；精度失败的配置不作为可用算法的加速结论。

## 证据与复现

- [主卡专项](async_hgemm_20261106/hip.json)、[第二张卡](async_hgemm_20261106/hip_device1.json)
- [原CUDA](async_hgemm_20261106/cuda.json)、[CUDA槽修复版](async_hgemm_20261106/cuda_fixed.json)、[专项比较](async_hgemm_20261106/comparison.json)
- [全库1109项](async_hgemm_20261106/qualification209.json)
- [209接口AMD](async_hgemm_20261106/qualified209_hip.json)、[5090](async_hgemm_20261106/qualified209_cuda.json)、[跨平台结果](async_hgemm_20261106/qualified209_comparison.json)
- [源码和报告hash审计](async_hgemm_20261106/source_audit.json)

```bash
HIP_VISIBLE_DEVICES=6,5 scripts/rocm/run.sh python -m leetcuda_rocm.async_hgemm_check fixtures --fixtures artifacts/rocm/continuation_all/b5/fixtures.pt
HIP_VISIBLE_DEVICES=6,5 scripts/rocm/run.sh python -m leetcuda_rocm.async_hgemm_check run --fixtures artifacts/rocm/continuation_all/b5/fixtures.pt --output artifacts/rocm/continuation_all/b5/hip.json --performance
HIP_VISIBLE_DEVICES=6,5 scripts/rocm/run.sh python -m leetcuda_rocm.async_hgemm_check run --device 1 --fixtures artifacts/rocm/continuation_all/b5/fixtures.pt --output artifacts/rocm/continuation_all/b5/hip_device1.json
```

复验前重新检查GPU预约与PCI，以上mask不代表永久可用。
大fixture、tensor输出、二进制在两端 `artifacts/rocm/continuation_all/b5`，JSON和脚本进Git。
