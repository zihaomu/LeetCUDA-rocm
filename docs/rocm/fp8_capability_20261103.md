# FP8/BF8 矩阵单元能力探测：2026-11-03

ROCm 10.1.0 / R9700 gfx1201 上，E4M3 (`torch.float8_e4m3fn`) 与 E5M2
(`torch.float8_e5m2`) 各 3 个矩阵形状均通过，共 **6/6**。
输入已量化为 FP8/BF8，累加与输出为 FP32，参考为同一量化输入的 FP64 矩阵乘。
测试覆盖 `[M,N,K]=[17,33,19]、[128,256,256]、[512,640,320]`，均在非默认 stream 运行。
门限 `atol=2e-4, rtol=2e-4`，最大绝对误差约 `1.02e-6`。

生成代码中分别确认 `v_wmma_f32_16x16x16_fp8_fp8` 和
`v_wmma_f32_16x16x16_bf8_bf8`，不是先转成 FP16 后做 GEMM。六个配置均未 spill。
[原始结果与 ISA hash](fp8_capability_20261103.json) 保存了环境与源码身份。

首次直接复用 FP16/BF16 GEMM 的整数零 padding 时，Triton 拒绝将整数转换为 FP8。
探测 kernel 改为读取 FP8 编码字节，用零字节处理边界后 bitcast 回 FP8，再进入 `tl.dot`。
这个调整解决了编译接口问题，未修改已有 FP16/BF16 GEMM 的实现或验证结果。

这只是硬件与编译路径验证，**完整 FP8 GEMM 迁移尚未完成**：原 C++ 算例还包括
BF16 动态量化、A per-row/per-block 与 B per-col/per-block 四种 scale 组合、
离线 B、端到端量化开销、多级流水和 benchmark。当前不增加 229 个原始 Python
接口中的通过数量，也不增加公开 AMD API 数量。

AMD 的 [RDNA4 WMMA 文档](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html)
列出 FP8/BF8 支持；这里的可行性判断另外有真实 GPU 数值与 ISA 证据。

```bash
./scripts/rocm/run.sh python -m leetcuda_rocm.research_fp8
```

测试结果写入 `artifacts/rocm/fp8/research.json`，每个配置的汇编同时写入该目录。
