# NVFP4 存储与量化链路的 AMD 实现

目标环境：锁定 ROCm 10.1.0 / Radeon AI PRO R9700 / gfx1201。
实现保留原 NVFP4 两级量化和打包格式，以 **FP16 WMMA + FP32 累加** 完成矩阵计算。
它不使用原生 FP4 矩阵指令，也不应按原生 FP4 峰值吞吐量宣传。

## 计算和存储契约

输入为 BF16 A[M,K]、B[K,N]，输出为 BF16 C[M,N]。每 16 个 K 元素先产生一个
非负 E4M3 scale，再将数据舍入为 E2M1；每个字节低四位存偶数元素，高四位存奇数元素。
SF 按原 `sf_byte_offset` 的 128 行 × 4 组布局存储，末尾行补零。B 离线格式为
打包的 B4T[N,K/2]、canonical SFB、可选 FP32 列 scale 及其启用标记。

四种模式保留：

| Python mode | 原 C++ enum | A 的 level-1 | B 的 level-1 |
| --- | --- | --- | --- |
| `single` | `kSingleLevel` | 无 | 无 |
| `a` | `kLevel1A` | 每行 | 无 |
| `b` | `kLevel1B` | 无 | 每列 |
| `two_level` | `kTwoLevel` | 每行 | 每列 |

level-1 为行／列 amax/2688；level-2 为每 16 个归一化值的 amax/6 经 E4M3 RNE
舍入。保留 SF 下溢导致组内归零、SF 饱和和 E2M1 RNE/satfinite 的有限输入语义。
未启用的 level-1 scale 不读取，检查会故意在这些缓冲区放 NaN。

加载矩阵 tile 时，将 E2M1 数据乘以对应 E4M3 SF，在寄存器内精确展开为 FP16。
先导 CPU 穷举证明全部 2,032 种有限码点／scale 组合都可精确展开。GPU 检查遍历
256 个打包字节 × 127 个 SF，共 32,512 对、65,024 个展开值，以 FP32 位模式逐位
检查 FP16 展开结果，包含正负零；另通过实际 WMMA 路径验证全部 2,032 个组合。
前一检查在 BF16 输出舍入前执行，防止中间值误差被舍入掩盖。
详见 [先导证据](fp4_feasibility_20261103.md)。
FP16 WMMA 使用 FP32 accumulator，level-1 scale 在 epilogue 乘回，最后转 BF16。
这不声称与 NVIDIA block-scaled MMA 的内部舍入顺序逐位相同。

## AMD 调度和原接口关系

共享实现为 [`fp4_gemm_hip.cuh`](../../kernels/rocm/fp4_gemm_hip.cuh)。
`interview/fp4_gemm.cuh` 在定义 `LEETCUDA_ROCM` 时包含它；原 CUDA 分支保留。
主要 C++ 名称 `fp4_gemm_bf16`、`fp4_gemm_quantize_b`、`fp4_gemm_bf16_b_offline`、
workspace/activation 大小接口和宽 tile 判定保持对应。AMD 的数据类型是 HIP BF16
和打包 uint8 字节；没有引入 CUTLASS 类型或 CuTe tensor layout 依赖。
`fp4_gemm_tma_fwd` 是兼容入口名称，实际没有使用 NVIDIA TMA。

量化保留 A 的行 amax 与逐组量化，以及 B 的分段列 amax、独立转置量化。
矩阵 tile 的环形 LDS 缓冲保持 FP4 打包数据和 SF，展开只发生在寄存器中。
`use_ws=False` 由 consumer 协作搬运；`True` 增加独立的 128 个 producer 线程。
两种路径的同步与缓冲区循环均实际执行，没有忽略 `use_ws`。

| 配置 | BM | BN | BK | stages | consumer 线程 | LDS 字节 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认 `Fp4GemmTraits<>` | 128 | 128 | 64 | 4 | 256 | 36,864 |
| 宽 `Fp4GemmTraits<256,6>` | 32 | 256 | 64 | 6 | 128 | 62,208 |

原宽配置 BM=128 的六级缓冲超过本机 64 KiB LDS，因此 AMD 选择能放入 LDS 的最大
32 倍数 BM，最多为 128。BN 和 stages 不变；这是公开的架构适配差异。
还检查了 BN/stages 为 128/2、128/3、256/4 的配置。模板只接受 2～6 stages，
超出 LDS 或线程资源的配置在编译期拒绝。`tile="auto"` 保留原 M/N/K 均 ≥4096
时选择宽配置的规则；可以显式指定 `default` 或 `wide`，两者均实际验证。

## 验证与性能口径

独立 CPU 参考用数值表、中点与偶数舍入实现 E2M1/E4M3；SF 布局通过五个维度的
reshape/transpose 构造，不调用生产 `sf_byte_offset`。打包字节、SF 字节以及全部
补零行要求逐位一致；启用的 level-1 scale 要求 `rtol=2e-7, atol=0`。
矩阵参考由独立量化结果展开为 FP64，执行 GEMM、level-1 缩放和 BF16 输出，
门限为 `atol=2^-10, rtol=2^-7`。未改变旧 HGEMM/HGEMV 失败项目的门限。

覆盖 3 个种子、7 个普通形状、8 类特殊输入，四种模式、两种 tile、两种调度、
三条调用路径。特殊输入包括零值、零分组、动态范围、SF 下溢、饱和、舍入中点、
正负零及互为 2^100／2^-100 的输入缩放；另有输入契约、alias 和非默认 stream 检查。
全部有限 E2M1/E4M3 组合的 GPU 检查要求与参考数值精确相等。

物理 GPU1／逻辑 GPU0（`HIP_VISIBLE_DEVICES=1`）的当前版本通过 **1,540 项检查**，
并通过 **96 个性能配置**的完整矩阵正确性验证，见
[完整 JSON](fp4_pipeline_20261103.json)。使用锁定的 ROCm 10.1.0 foundation，
报告保留镜像、Torch/Triton/HIP 身份、源码 SHA256、数值门限及实际生成的 WMMA 汇编。

| 验证范围 | 结果 | 证据 |
| --- | --- | --- |
| 物理 GPU1，Python 完整链路 | 1,540 项检查、96 个性能配置通过 | [主报告](fp4_pipeline_20261103.json) |
| 物理 GPU0／逻辑 GPU1，`HIP_VISIBLE_DEVICES=1,0` | 1,541 项通过，含 active device 为 0 时输入在 device 1 的 guard 复核 | [第二张卡](fp4_device1_20261103.json) |
| 物理 GPU0，自动宽 tile 与大尺寸尾块 | 4096³、4097×4104×4160，40 项完整矩阵检查通过；四种模式、两种调度、full/offline B | [大尺寸报告](fp4_large_20261103.json) |
| 物理 GPU1，独立 C++ 入口 | 263 项检查、96 个性能配置通过；包括额外 BN/stage 配置 | [C++ 报告](fp4_cpp_20261103.json) |

主报告的 1,540 项包括 1,508 项普通／特殊输入检查、27 项接口契约、1 项 packed
展开穷举和 4 项 WMMA 展开穷举。第二张卡额外增加 1 项设备 guard。C++ 的 263 项
由 165 项小形状／stage／基线检查及 98 项性能形状检查构成，不与 Python 项目直接相加。
这些数字是检查数，不是新增算子数。

性能分别测 `quantized`（仅矩阵计算）、`full`（A/B 量化 + GEMM）、`offline_b`
（仅 A 在线量化 + GEMM）。预分配输出和 workspace；离线 B 的准备时间不计入
`offline_b`。BF16 库基线精度与本 FP4 链路不同，不能省略量化损失。
Python 验证性能形状的完整输出矩阵；独立 C++ 小形状逐元素检查，大形状每个配置
抽查 64 个输出，且单独校验 hipBLAS 基线。

### 初始版本至当前版本的优化

先固定数值门限，再依次强制内联 tile 计算并限制外层展开、直接构造 FP16 位模式、
用 packed FP16 乘法展开 E2M1×SF。三轮使用同一物理 GPU1、相同的图捕获方法：
每图 32 次调用，自适应 replay 以约 40 ms 为目标，交替采集 5 组样本。
这是共享主机上的顺序测量，没有独占预约；最终测量结束后发现其他任务占用 GPU0/1/2，
后续运行待释放后才启动。原始 candidate／baseline 样本均保留，不能据此承诺独占环境性能。

| 版本 | 正确性检查 | 性能配置 | scratch 指令 | 设备函数调用指令 | packed FP16 乘法指令 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始实现 | 1,539 | 96 | 1,432 | 16 | 0 |
| 内联及限制循环展开 | 1,539 | 96 | 0 | 0 | 0 |
| FP16 位模式与 packed 展开 | 1,540 | 96 | 0 | 0 | 385 |

指令数是整个编译产物的静态计数，不是一次 GEMM 的动态执行次数。三轮的源码与汇编
哈希对应，历史源码和 code object 已归档；见 [逐配置优化比较](fp4_optimization_20261103.json)、
[初始结果](fp4_pipeline_initial_20261103.json)、[内联结果](fp4_pipeline_inline_20261103.json)。

当前 Python 路径，2048³、`two_level`、默认 tile，单位 μs（5 组中位数）：

| WS | 路径 | FP4 当前 | 未量化 BF16 库基线 | 相对初始 FP4 加速 |
| --- | --- | ---: | ---: | ---: |
| 关 | 纯 GEMM | 662.34 | 168.09 | 5.41× |
| 关 | 完整量化链路 | 718.42 | 167.96 | 5.03× |
| 关 | B 离线 | 689.33 | 167.02 | 5.20× |
| 开 | 纯 GEMM | 634.43 | 168.54 | 4.90× |
| 开 | 完整量化链路 | 692.93 | 168.18 | 4.56× |
| 开 | B 离线 | 663.76 | 168.19 | 4.72× |

优化后仍明显慢于 BF16 库基线；宽 tile 在本批 512³／2048³ 上也更慢。
不能把降低存储位宽等同于提高吞吐量。上述随机输入下，FP4 输出相对未量化 FP64 GEMM
的 relative L2 约 **13.4%**，这是量化损失；相对独立 FP4 量化参考的数值检查另行通过。

C++ 使用同样的每图 32 次、自适应约 40 ms、5 组交替样本方法。2048³、`two_level`、
默认 tile、WS 开启时，纯 GEMM／完整链路／B 离线分别为 **639.78 / 692.84 / 665.21 μs**，
对应未量化 BF16 hipBLAS 基线 **166.59 / 166.89 / 167.00 μs**。
报告记录该可执行文件实际链接的 HIP、hipBLAS、rocBLAS、HSA、COMGR 路径与 hash；
不同 Python／C++ 运行库的测量分开保留，不拼接为单一性能结果。

新增 FP4 后，共用 Python 导出及 C++ 构建脚本的 FP8 回归也通过：
[Python 753 项](fp8_entrypoint_20261103.json)、[C++ 93 项](fp8_cpp_entrypoint_20261103.json)。
该回归没有重测 FP8 性能，不改写原有性能证据。

## 调用与限制

```python
import torch
from leetcuda_rocm import fp4_gemm, fp4_gemm_prepacked, quantize_fp4_weight

a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
c = fp4_gemm(a, b, mode="two_level", use_ws=False, tile="default")
weight = quantize_fp4_weight(b, level1=True)
d = fp4_gemm_prepacked(a, weight, level1_a=True, use_ws=True, tile="wide")
```

需要同设备、连续、无梯度、有限的 BF16 输入；正尺寸，K%64=0，N%8=0；元素数量
和 padding 后的 SF 索引受 32 位限制。数据 16 字节对齐，workspace 256 字节对齐。
非零 level-1 scale 和其倒数必须为正常有限 FP32，缩放累加不得溢出；接口不扫描值域。
支持预分配 `out=`、`workspace=`。不支持 NaN/Inf、反向或未验证的其他 AMD 架构。

```bash
# 本轮使用物理 GPU1；每次运行前重新确认占用，GPU3 禁止计算。
HIP_VISIBLE_DEVICES=1 ./scripts/rocm/run.sh python -m leetcuda_rocm.fp4_check \
  --performance --output artifacts/rocm/fp4/pipeline.json
HIP_VISIBLE_DEVICES=1 ./scripts/rocm/build_bench.sh fp4 --run --performance
HIP_VISIBLE_DEVICES=0 ./scripts/rocm/run.sh python -m leetcuda_rocm.fp4_large_check \
  --output artifacts/rocm/fp4/large.json
```

原 `interview/build.sh` 的整套 benchmark 仍待迁移；此处是 FP4 专用 C++ 入口。
FP4 原实现没有计入 229 个 Python 原始计算接口，因此本批不提高 153/229 的比例。

后续共用 ISA 提取器已修正为不改写输入 ELF，见 [只读审计](isa_readonly_20261103.json)。
本报告的数值与性能数据保持原样；FP4 Torch／C++ 产物的 GPU 代码经复核逐位不变。
历史 JSON 中 `leetcuda_rocm/isa.py` 的源码 hash 对应审计报告中的 `legacy_isa_source`。
