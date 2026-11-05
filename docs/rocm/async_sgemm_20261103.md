# 三个原始异步 SGEMM 接口的 AMD 迁移

2026-11-03，锁定 ROCm 10.1.0 / R9700 / gfx1201。
三个原始公开接口已接入 `leetcuda_rocm.load("sgemm")`，原接口通过数由
153/229 增至 **156/229（68.12%）**；当前待迁移／验证 56 个、精度未验收 17 个。
全套回归 **23 模块、156 接口、888 项检查通过**，见
[完整资格报告](qualification_20261103_async.json)。原 CUDA 分支保留，本机未执行 CUDA 回归。

## 接口与调度

| 原始接口 | BM×BN×BK | 每线程输出 | block 线程 | 两级 LDS 字节 |
| --- | --- | --- | ---: | ---: |
| `sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async` | 64×64×16 | 连续 8×4 | 128 | 16,384 |
| `sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async` | 128×128×16 | 分成四个象限的 8×8 | 256 | 32,768 |
| `sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async` | 128×256×16 | 连续 8×16 | 256 | 49,152 |

实现位于 [`sgemm_async.cu`](../../kernels/sgemm/sgemm_async.cu) 的 AMD 分支。
保留原来的 A 在线转置、B 行布局、两个 LDS 缓冲、每线程输出映射与逐 K 的 FP32 FMA 顺序。
三个接口分别实例化上述配置，没有将它们映射到同一个库函数。

本机编译器拒绝 `global_load_lds`、`global_load_async_lds` 和
`global_load_async_to_lds_b128` 三种 builtin：缺少对应目标特性，完整诊断见
[编译探测](async_copy_probe_20261103.json)。AMD 的
[direct-to-LDS 文档](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/cross-arch/direct-to-lds-builtins.html)
列明了架构限制；LLVM 的
[AMDGPU builtin 文档](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html)
给出了这些目标特性要求。上述结论针对锁定编译器和 gfx1201，不外推至其他 AMD 架构。

因此，这里的 `_async` 对应 **AMD VGPR 预取软件流水线**：

1. 用 `global_load_b128` 提前读取下一个 tile 的 A/B 到寄存器。
2. 从当前 LDS 缓冲取数，执行 FP32 FMA。
3. 用 `s_wait_loadcnt 0` 等待预取完成，再写下一个 LDS 缓冲。
4. block 同步后交换缓冲；最后计算剩余 tile。

它没有复现 NVIDIA `cp.async` 的直接入共享内存、commit group 或缓存提示机制。
等待指令通过读写寄存器约束连接所有预取值，不能被替换为空操作。K=16 的单 tile
只经过首批加载和收尾；奇偶 tile 数均单独检查。

完全展开 K 循环曾使 8×16 变体产生 scratch spill，自动插入的 scratch 等待提前结束了
全局加载的重叠。保留 K 循环后，三个变体均无 scratch 读写和设备函数调用。
实际发布扩展包含两个 GPU code object；验证从其 `.hip_fatbin` 中逐一提取并反汇编，
不是只查看第一个对象或独立探测程序。

| 变体 | 静态 vector load 指令数 | 最后一次预取与等待之间的静态 FMA 操作数 | scratch／设备函数调用 |
| --- | ---: | ---: | --- |
| 8×4 | 8 | 32 | 0／0 |
| 8×8 | 8 | 64 | 0／0 |
| 8×16 | 12 | 128 | 0／0 |

FMA 计数包含 `v_dual_fmac_f32` 的两个操作，数值是汇编区间的静态计数，
不包含 K 循环运行时的重复执行次数。各 code object、汇编和扩展二进制的 hash 均记录在报告中。
JIT 缓存可在后续构建中重新链接，因此实测二进制另外按 SHA256 归档，报告中的
`binary` 指向独立归档，`build_cache_binary` 仅记录原构建缓存位置。

同时修正了共用提取器：旧 `llvm-objcopy --dump-section` 调用未指定输出文件，会原位
重写输入 ELF；现在将 ELF 输出到临时文件，并检查输入 hash 不变。
[只读审计](isa_readonly_20261103.json) 对 FP4／FP8 的 Torch 与 C++ 四个产物，以及
包含两个 code object 的 SGEMM 扩展共五个产物验证通过，提取出的 GPU 代码与既有
报告逐位一致。历史 FP4／FP8 测量保持原样，其提取器源码保存在审计报告的
`legacy_isa_source` 字段所指位置。

## 正确性、边界与性能

- [物理 GPU0／逻辑 GPU0](async_sgemm_20261103.json)：**213 项检查、18 组性能比较通过**。
- [物理 GPU1／逻辑 GPU1](async_sgemm_device1_20261103.json)：**213 项检查通过**。
- 两次可见设备均为 `0,1`，都验证 active device 与输入设备不同时的 guard 及设备恢复。
- 每次包括 81 项数值检查、54 项输入拒绝检查、6 项设备检查和 72 项双 stream 并发检查。
  数值检查覆盖 3 个随机种子、1/2/3/5/8/17/65/256 个 K tile、不同 M/N block 数、
  零值、消去与不同量级输入；性能形状还核对完整输出矩阵及每条基线。

独立参考为 FP64 GEMM 后转 FP32，门限保持 `atol=rtol=2e-4`。
81 项数值检查还要求输出与对应原同步 HIP 版本**逐位一致**；没有改成 FP16/BF16 矩阵计算。
新增接口在全套基础回归中各增加 7 项；888 项总数还包含多设备可见时增加的既有模块检查，
因此不能将检查数量的增量直接当作新增算子数量。

性能使用复用缓冲、每图 32 次调用、自适应约 40 ms replay 和 5 组交替样本。
每种配置分别与原同步 HIP 版本、`torch.mm` FP32 成对测量，关闭 TF32 许可。
下面为与 PyTorch 成对测量的中位数，单位 μs，比例为基线／当前实现：

| M=N=K | 变体 | AMD 预取实现 | PyTorch FP32 | 比例 |
| ---: | --- | ---: | ---: | ---: |
| 256 | 8×4 | 11.28 | 16.17 | 1.434× |
| 256 | 8×8 | 15.24 | 15.83 | 1.038× |
| 256 | 8×16 | 49.85 | 15.50 | 0.311× |
| 1024 | 8×4 | 153.91 | 135.85 | 0.883× |
| 1024 | 8×8 | 108.32 | 136.47 | 1.260× |
| 1024 | 8×16 | 201.63 | 130.66 | 0.648× |
| 2048 | 8×4 | 1217.97 | 1165.00 | 0.957× |
| 2048 | 8×8 | 853.98 | 1169.33 | 1.369× |
| 2048 | 8×16 | 1444.33 | 1176.07 | 0.814× |

相对原同步版本，8×8 在三种规模上分别快 1.316×、1.082×、1.020×；
8×4 有退化，8×16 在 2048³ 上也略慢。加载与计算重叠并不保证每个形状都更快。
本次是共享主机上的采样，保留全部原始样本，不声称所有 workload 或其他硬件都有收益。

输入为同设备、连续、无梯度、16 字节对齐的 FP32 矩阵；M/N 必须分别是表中 BM/BN
的整数倍，K%16=0，三者均为正；矩阵元素数不超过 INT_MAX，输出不能与输入重叠。
支持非默认 stream 和设备 guard。没有尾块回退、autograd 或其他架构的资格声明。

```python
from leetcuda_rocm import load

lib = load("sgemm")
lib.sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async(a, b, out)
```

```bash
# 先确认 GPU0/1 可用；GPU3 禁止计算。
HIP_VISIBLE_DEVICES=0,1 ./scripts/rocm/run.sh python -m leetcuda_rocm.async_gemm_check \
  --guard-device 1 --performance --output artifacts/rocm/async_sgemm.json
HIP_VISIBLE_DEVICES=0,1 ./scripts/rocm/run.sh python -m leetcuda_rocm.async_gemm_check \
  --device 1 --guard-device 0 --output artifacts/rocm/async_sgemm_device1.json
HIP_VISIBLE_DEVICES=0,1 ./scripts/rocm/run.sh python -m leetcuda_rocm.smoke
```

剩余的 3 个 HGEMM 异步变体、52 个矩阵指令／CuTe 变体和 1 个 TF32 库接口继续迁移；
17 个既有精度未验收项保持原结论。完整 C++ benchmark 和 FFPA 等补充范围仍未完成。
