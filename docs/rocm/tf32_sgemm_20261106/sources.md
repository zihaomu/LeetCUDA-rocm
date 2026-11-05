# B6 原型依据

- NVIDIA本地锁定CUDA头文件 `/usr/local/cuda/include/crt/mma.h:96` 中 `__float_to_tf32` 为 `cvt.rna.tf32.f32`；原WMMA wrapper原地修改A/B。
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 定义RNA为最近值，ties away from zero。
- [AMD RDNA4 WMMA builtin](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html) 列出BF16/FP16/FP8等输入；原型使用BF16矩阵指令分解TF32。

原型首轮编译失败为向量元素不能绑定引用，改为临时标量；数值第一轮K8193残差项反复加入运行总和产生误差，改为BK8局部积归约后通过。原始失败日志/JSON保留。正式实现采用显式TF32x3精度升级；没有声称硬件原生TF32或CUDA逐位一致。
