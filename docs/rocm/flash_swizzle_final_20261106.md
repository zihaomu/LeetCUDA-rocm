# B1 FlashAttention swizzle 最终性能验收

2026-11-06。Q、QK、QKV 三个原入口的功能与固定性能目标全部通过。没有调整原计划的容差、形状、三轮稳定性或资源门槛。

| 入口 | 三轮几何平均加速（冻结父实现/本实现） | 最坏本实现/父实现 | 最大三轮波动 | 判定 |
| --- | --- | ---: | ---: | --- |
| Q | 1.178779 / 1.177236 / 1.176710 | 0.9577 | 4.7435% | 通过 |
| QK | 1.219128 / 1.216831 / 1.216176 | 0.9480 | 4.5794% | 通过 |
| QKV | 1.112933 / 1.114702 / 1.114971 | 0.9922 | 2.8949% | 通过 |

每入口18个固定配置、3轮，每轮几何均值≥1.05、任意配置退化≤10%、波动≤5%。42个特化全部零scratch和VGPR spill，LDS≤48KiB。QKV在D64/128的8配置冲刺目标也通过：1.110675 / 1.111541 / 1.111891，均≥1.10。
这些是同一AMD卡与冻结父实现的比值，不能解释成与5090或成熟库的整体性能比。

## 实现变化

三个入口继续使用不同的Q、QK、QKV XOR布局。Q/QK的V仍是线性布局，行跨度增加8个half的padding，改变LDS bank分布。QK计算完成并同步后，让P复用已不再读取的Q/K存储；V双槽保持独立。Q/QK最大动态LDS为41472B，QKV为49152B。

保留query/key tile、K深度16、大D输出分片与每个key block后的FP16状态舍入频率，未用改变精度获取性能。非swizzle父路径与QKV布局保持原有方案。新增两条真实GPU探针检查padding8、行列值、双槽边界和canary，布局检查合计10项。

## 功能、双卡与旧入口回归

主卡物理GPU4/PCI `0000:83:00.0`，第二卡物理GPU2/PCI `0000:43:00.0`；device guard还使用允许的物理GPU0。物理GPU3始终排除。ROCm10.1.0、HIP7.16、Torch2.14/Triton3.8固定镜像与设备身份见每份JSON。

- 两张AMD卡各1419项：372共享输入、879边界、42状态精度、116设备/stream、10布局；相比早期1329项，设备检查已覆盖全部29个Attention导出。
- 5090原CUDA的372数值、42状态以及162性能记录已完成；共同372输出全部通过容差，其中36逐位一致。
- 既有shared/tiling四入口394项与冻结二进制逐位一致，174条三轮性能记录全部未超过10%退化。
- 全库15组15163项及原229接口462组同精度跨平台对照通过，见[统一验收](unified_acceptance_20261106.md)。

## 稳定性复测与失败候选

最终固定二进制首测中，QK的N1024/D64/stage1三轮波动为6.0006%，因此该入口首测判失败。随后完整重测QK的18配置×3轮，全部通过。最终报告保留Q/QKV首测全部三轮，并用QK复测的完整三轮替换；没有按配置或最好样本挑选结果。二者使用同一源码、夹具、测试脚本、GPU和同一不可变二进制。

更早一次全模块测量在N256/D64/stage1出现5.54%波动，也完整保留。plain-V向量化、BR32、K深度32三类候选没有达到性能目标，其中BR32还产生spill。研究报告、实际编译源码和失败记录在[research目录](flash_swizzle_final_20261106/research/)；候选probe仅编译tiled源码，不能将其报告中列出的完整来源误认为实际参与编译的全部文件。

## 实际加载文件的可追溯性

审计发现多进程Torch JIT会重新链接同一个缓存`.so`，导致测试前后文件SHA变化；提取的5个HSACO内容集合相同，但原路径不足以证明实际加载文件。
首次“加载后复制并重新导入”的修复仍被pybind模块缓存复用，验证失败并保留记录。最终改为独立子进程编译，在父进程首次导入前按二进制SHA冻结文件；父进程只加载该固定路径，并校验receipt和文件SHA。

最终三份AMD报告加载与ISA提取的SHA均为`59a46e75d33501f322a619cd7e22c14e75ff54a392dbffbe46aee8fe46acdb90`。实际C函数地址对应的`/proc/self/maps`也指向冻结文件，见[加载审计](unified_acceptance_20261106/full/immutable_loader_audit.json)。四个CUDA二进制及原源码SHA也逐一复核并冻结。

## 复现与证据

```bash
# mask仅为本机此次映射；先检查PCI、占用及预约。
HIP_VISIBLE_DEVICES=6,2 scripts/rocm/run.sh python -m leetcuda_rocm.flash_swizzle_check run \
  --fixtures artifacts/rocm/flash_tiled/fixtures.pt \
  --frozen-baseline artifacts/rocm/flash_swizzle/baseline/tiling.json \
  --rounds 3 --performance --output artifacts/rocm/b1_final.json
python3 -m leetcuda_rocm.flash_swizzle_acceptance \
  --hip artifacts/rocm/b1_final.json --output artifacts/rocm/b1_acceptance.json
```

[验收判定](flash_swizzle_final_20261106/acceptance.json)、[最终数据集](flash_swizzle_final_20261106/hip.json)、[首测及不稳定判定](flash_swizzle_final_20261106/frozen_acceptance.json)、[完整QK复测](flash_swizzle_final_20261106/frozen_qk_retest.json)、[第二卡](flash_swizzle_final_20261106/hip_second.json)、[5090对照](flash_swizzle_final_20261106/cross.json)、[已有四入口回归](flash_swizzle_final_20261106/existing_regression.json)、[源码/二进制审计](flash_swizzle_final_20261106/source_audit.json)。
完整夹具、输出、DSO和ISA保留在本机及5090专用目录的`artifacts/rocm/continuation_all/`，Git保存JSON、来源快照和复现程序；没有把大二进制加入源码仓库。
