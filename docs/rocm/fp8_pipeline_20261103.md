# FP8 GEMM 完整链路迁移

2026-11-03，ROCm 10.1.0 / Radeon AI PRO R9700 / gfx1201。
**E4M3 完整链路和离线权重路径的功能验证通过，性能尚未收敛。**
这是 `kernels/interview/fp8_gemm.cuh` 的 AMD 分支，不改变原始 Python 接口
153/229 的覆盖率。本 FP8 批次使独立 AMD 计算接口由 2 个增加为 4 个，量化辅助接口单列；
后续 [FP4 批次](fp4_pipeline_20261103.md) 将总数增至 6 个。

加入 FP4 后，共用导出／构建脚本的回归为
[Python 753 项通过](fp8_entrypoint_20261103.json)、[C++ 93 项通过](fp8_cpp_entrypoint_20261103.json)。
以下原始性能测量及其环境身份保留；新回归仅验证正确性，未重测 FP8 性能。

后续 ISA 提取器修正为不改写输入 ELF，见 [只读审计](isa_readonly_20261103.json)。
FP8 Torch／C++ 产物的 GPU 代码经复核逐位不变；历史测量未改写，旧提取器源码
保存在审计报告 `legacy_isa_source` 所指位置。

## 实现与语义

共用实现为 [`fp8_gemm_hip.cuh`](../../kernels/rocm/fp8_gemm_hip.cuh)：
BF16 输入 → 动态 E4M3 量化 → 原生 FP8 WMMA、FP32 累加 → scale 反量化 → BF16。
完整链路、离线 B 与只计算 GEMM 均调用同一套内核。

- 保留 A 按行／128 行块、B 按列／128 列块的四种组合。
- `use_ws=False` 由计算线程协作搬运；`True` 有独立的 128 个 producer 线程，
  与 consumer 在线程块内通过 LDS 环形缓冲协作。没有忽略该参数，也没有称为原 NVIDIA TMA。
- AMD 默认 tile 为 64×64×64、2 stages。另测试 32×64×128/s2、64×32×64/s3、
  64×64×64/s4；tile/stage 参数确实参与加载、矩阵计算和缓冲区大小。
- 保留 C++ `fp8_gemm_bf16`、`fp8_gemm_quantize_b`、
  `fp8_gemm_bf16_b_offline` 及 workspace/activation 大小接口。
  原 `fp8_gemm_tma_fwd` 名称仅作兼容入口，AMD 实现用 LDS。
- 原 `interview/fp8_gemm.cuh` 通过 `LEETCUDA_ROCM` 选择此分支；CUDA 分支保留，
  本 AMD 主机没有执行 CUDA 回归。原多模块 `interview/build.sh` 尚未整体迁移。

实际 Torch 扩展与独立 C++ 可执行文件的代码对象均包含
`v_wmma_f32_16x16x16_fp8_fp8`。编译产物、汇编、源码及环境身份见 JSON 报告。
此指令的参数形式与 AMD 的 [RDNA4 builtin 文档](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html)
一致；它使用 E4M3 输入和 FP32 累加。

## 正确性与边界

Python 主报告 **753 项检查通过**，另有 **48 组性能比较**。
物理 GPU1／逻辑 GPU1 的同一套 **753 项检查也全部通过**，可见设备为 `0,1`；
两次都包含输入设备与当前设备不同的 guard。第二张卡重复验证不增加接口数。
检查包括 3 个种子、7 个形状、5 类特殊输入，四种模式、两种调度和三条调用路径。
普通样本从 1×8×16 到 256×256×256，并覆盖 K=576/1024、M/N/K 尾块、
128 行／列块边界。特殊输入包含零分组、动态范围、精确量化中点与正负零、
互为 2^100 和 2^-100 的输入缩放。另验证输入契约、存储对齐、alias、公开入口、
非默认 stream，以及输入设备与当前设备不同的 guard。

量化参考在 CPU 根据 E4M3 指数／尾数定义和 RNE 中点规则独立实现，不调用 GPU FP8
转换。A/B 字节逐位一致；scale 的 `rtol=2e-7, atol=0`。GEMM 参考先解码这些
独立量化字节，再用 FP64 矩阵乘、scale 和 BF16 输出，与实现按
`atol=2^-10, rtol=2^-7` 比较，零分组额外要求精确零。
该门限针对新的 BF16 输出 FP8 链路，没有更改旧 HGEMM/HGEMV 的精度门限。

相对**未量化** BF16 输入的 FP64 矩阵结果，512³／2048³ 随机样本的 relative L2
约为 3.7%。这是单独记录的量化损失，不作为移植错误，也不承诺任意分布都有这一误差。

Python 在性能形状上验证完整输出矩阵。C++ 独立程序在 3 个小形状上逐元素验证，
在 512³／2048³ 上每种配置验证 64 个输出位置；C++ 的大矩阵抽样不能写成逐元素覆盖。
其 CPU 量化编码器与 FP64 点积参考独立于生产 HIP 代码，并额外核对 hipBLAS 基线。

## 使用

```python
import torch
from leetcuda_rocm import fp8_gemm, fp8_gemm_prepacked, quantize_fp8_weight

a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
c = fp8_gemm(a, b, mode="row_col", use_ws=False)
weight = quantize_fp8_weight(b, per_col=True)
d = fp8_gemm_prepacked(a, weight, per_row=True, use_ws=True)
```

输入必须是同设备、连续、无梯度的 BF16 矩阵；正尺寸、K%16=0、N%8=0，
元素数不超过 32 位索引范围。数据指针需要 16 字节对齐，传入 workspace 需要
256 字节对齐和足够容量。可传 `out=`、`workspace=` 复用缓冲区。
只支持已验证的 gfx1201，不外推至 wave64/Instinct。

有限输入还需满足非零分组的 `amax/448` 与 `448/amax` 为正常有限 FP32 值、
缩放累加不溢出。这与原 FP32 scale/inverse 算法有关；不扫描值域来避免额外同步。
NaN/Inf、scale 的下溢／上溢、反向、E5M2 完整量化链不在本次支持声明内。

```bash
# Python 完整检查和性能，默认物理 GPU0
./scripts/rocm/run.sh python -m leetcuda_rocm.fp8_check --performance \
  --output artifacts/rocm/fp8_pipeline.json

# 独立 HIP/C++，运行前确认 GPU0 空闲
./scripts/rocm/build_bench.sh fp8 --run --performance
```

## 性能口径与剩余工作

三种时间分别为：`quantized` 仅 GEMM、`full` 量化 A/B + GEMM、`offline_b`
仅在线量化 A + GEMM。预分配输出／workspace，排除分配；离线 B 的准备不计入
`offline_b`，不能据此宣称模型完整加载／运行时间。
基线为相同 BF16 输入输出的库矩阵乘，精度格式与 FP8 不同。

Python 使用 5 轮交替的 Triton HIP graph 计时，每轮按 40ms 估计构图并取重复测量中位数。
C++ 每图 32 次调用、每样本 32 次 replay、5 轮交替取中位数。
主机其他卡有负载；小矩阵和不同测量组的绝对时间存在较大波动，原始样本完整保留。
两套时间分开报告，不把不同计时方法的绝对时间直接互比。

C++ 最终报告 **143 项检查通过、48 组性能比较完成**。其中 90 项为小形状和 tile/stage 检查，48 项为性能形状的抽样检查，5 项核对 hipBLAS 基线。

以下为 C++ `row_col` 模式的全部两种规模、两种调度和三条路径；单位 μs，倍率为“基线时间 / 本实现时间”，小于 1 表示本实现较慢。

| M=N=K | WS | 路径 | FP8 μs | BF16 hipBLAS μs | 倍率 |
| ---: | --- | --- | ---: | ---: | ---: |
| 512 | False | quantized | 13.15 | 6.97 | 0.530× |
| 512 | False | full | 49.79 | 6.87 | 0.138× |
| 512 | False | offline_b | 16.25 | 6.88 | 0.424× |
| 512 | True | quantized | 12.07 | 7.01 | 0.581× |
| 512 | True | full | 48.31 | 6.83 | 0.141× |
| 512 | True | offline_b | 15.08 | 6.86 | 0.454× |
| 2048 | False | quantized | 230.14 | 149.72 | 0.651× |
| 2048 | False | full | 373.84 | 149.99 | 0.401× |
| 2048 | False | offline_b | 245.47 | 151.27 | 0.616× |
| 2048 | True | quantized | 225.26 | 153.80 | 0.683× |
| 2048 | True | full | 365.43 | 155.43 | 0.425× |
| 2048 | True | offline_b | 241.36 | 155.99 | 0.646× |

四种模式的 C++ 共 48 组测量均慢于 BF16 hipBLAS。2048³ 的 `row_col`、WS 路径中，纯计算约 225 μs，完整链路约 365 μs；对应基线约 154～156 μs。

Python 报告使用另一种构图和采样方式，且 Torch 与独立 C++ 分别加载其锁定环境中的运行库。其 `row_col`、2048³、WS 测得约 505 / 1090 / 459 μs（纯计算／完整／离线 B），配对基线约 256～271 μs。两套记录都有明显差距；不挑选不同测量组中的最小值拼成加速比。后续优化需要固定测量方式并复测。

当前测量中 FP8 仍慢于 BF16 库基线。源码显示 B 量化沿整个 K 做两遍扫描，
每 128 列只有一个线程块；例如 N=2048 时仅 16 个线程块。
后续将单独量化此开销，研究分段 amax、独立转置量化、LDS 访问与矩阵 tile，
并保持量化字节和矩阵参考门限不变。功能通过不代表整体性能验收完成。

证据：

- [Python GPU0 正确性、ISA 与 48 组性能](fp8_pipeline_20261103.json)
- [Python GPU1 正确性与 ISA](fp8_pipeline_device1_20261103.json)
- [独立 C++ 正确性、ISA、运行库身份与 48 组性能](fp8_cpp_20261103.json)
- [FP4 下一阶段可行性研究](fp4_feasibility_20261103.md)
