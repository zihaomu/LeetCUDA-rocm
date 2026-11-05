# NVFP4 后续迁移路线探测

2026-11-03，锁定 ROCm 10.1.0 / gfx1201。本文保留早期能力与数值可行性研究；
**该先导实验本身没有发布 FP4 GEMM，也不增加已通过接口数**。
后续实现已完成 GPU 功能验证，当前状态见 [FP4 完整链路报告](fp4_pipeline_20261103.md)。

原 `interview/fp4_gemm.cuh` 使用 E2M1 数据、每 16 个 K 元素一个非负 E4M3
scale，以及可选的 FP32 行／列 level-1 scale。必须保留这两级量化及原始
`sf_byte_offset` 布局，不能用普通 FP8 量化替换后称为 NVFP4。

当前编译器在 gfx1201 上编译 `__builtin_amdgcn_cvt_scalef32_pk_f32_fp4`
失败，诊断明确要求 `fp4-cvt-scale-insts`。这只证明该转换指令在本目标不可用。
AMD 的 [RDNA4 WMMA 指令列表](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html)
列出 FP16/BF16、INT8/INT4、FP8/BF8；不能把 INT4 当作浮点 E2M1。
Clang 的 [builtin 文档](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html)
也对该 FP4 转换列出独立的目标特性要求。

可实施的替代路线：保留 FP4 打包数据及 E4M3 scale，在加载 tile 时展开
`E2M1 × E4M3`，送入 AMD FP16 WMMA，并用 FP32 累加。独立 CPU 穷举了
16 个带符号 E2M1 码点 × 127 个非负有限 E4M3 scale，共 **2,032/2,032**
组合，展开为 FP16 再转回 FP64 全部精确一致，正负零也保留；非零绝对值范围
为 2^-10～2688。这说明这一展开步骤不会引入新的舍入误差。

该路线属于 **FP4 存储／量化 + FP16 矩阵计算**，不是原生 FP4 矩阵指令，
也没有证明与 NVIDIA block-scaled MMA 的内部累加顺序逐位相同。
在该先导阶段，两级量化、SF 布局、尾块、WS/stage 参数、离线 B、
完整矩阵参考与性能尚未验证；本先导报告不提供 GPU GEMM 正确性或吞吐量结论。

证据：[环境、完整编译诊断与穷举结果](fp4_feasibility_20261103.json)。复现：

```bash
./scripts/rocm/run.sh python -m leetcuda_rocm.research_fp4
```
