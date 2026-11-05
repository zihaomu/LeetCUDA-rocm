# Swizzle 验收证据

最终主卡报告 `hip.json` 是 artifacts 中 `vector_hip.json` 的逐字节副本；`comparison.json` 内的 `vector_hip.json` hash 指向这份数据。`hip_device1.json` 为第二张 AMD 卡。`cuda.json` 使用上游 original/；qualified173_cuda.json 使用包含历史修复的 ported/。

`baseline/` 是修改前在同机复验的冻结参考；`initial/` 保存 Q→QK→QKV 顺序推进时各入口的正确性、对照、性能与初版源码；`scalar_layout/` 是保留连续向量但仍使用逐元素表达式的失败优化候选。前两阶段的源代码仅用于解释失败实验，不参与当前扩展构建。

初期报告共用的 `isa_device0` 反汇编路径曾被后续特化覆盖，其旧反汇编路径不再视为可直接重放的证据；报告中的原二进制 SHA256 和 artifacts/binaries 下不可变快照仍保留。最终报告改用独立的 `vector_hip_isa_device0` / `hip_device1_isa_device1` 路径，hash 已单独审计。

大 fixture、输出 .pt、二进制 .so 和完整 ISA 位于本机 `artifacts/rocm/flash_swizzle/`、共享输入位于 `artifacts/rocm/flash_tiled/`；远端位于 `/home/zihaomu/zihaomu_workspace/test_lap/leetcuda_rocm10/artifacts/flash_swizzle/`。JSON 中保留其原始路径和 hash，没有重写历史结果。
