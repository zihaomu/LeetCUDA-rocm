# 独立FlashAttention八路径：设计研究记录

2026-11-06。此文记录源码复核与当前实现设计；尚未通过测试的路径不增加支持数。本文开工时interview为45/53；现八路径均完成功能验收，interview为53/53。最终结果见[验收报告](interview_attention_20261106.md)。

| Kind | 原定义 | 必须保留的机制 |
| ---: | --- | --- |
| 0 | flash_attn_mma_stages_split_q | Q一次加载，K多级预取，V单槽；pad8及每16列XOR布局；QK/PV/运行状态精度分别标注 |
| 1 | flash_attn_tma_mma_ws_stages_split_q | 独立producer/consumer，独立K/V深度，释放K与释放V的时点不同 |
| 2 | flash_attn_3_tma_ws_stages_split_q | 两组消费者分别处理奇数/偶数KV块，独立K/V bank及状态，最后稳定合并 |
| 3 | flash_attn_mma_stages_split_q_cute | CuTe GMMA swizzle坐标、寄存器fragment预取及P转换、协作预取 |
| 4 | flash_attn_tma_mma_ws_split_q_cute | CuTe布局与P寄存器转换，真实独立K/V生产者/消费者 |
| 5 | flash_attn_3_tma_mma_ws_split_q_cute | CuTe布局及fragment转换，双消费者奇偶KV和状态合并 |
| 6 | flash_attn_3_cute_tma_copy_smoke | 单独Gmem→LDS→Gmem复制路径；布局映射必须与冻结Cutlass独立对照 |
| 7 | flash_attn_cute_persist_d_sm120 | 全D的Q常驻寄存器、FP16预乘scale·log2e、阈值8的条件重缩放、持久CTA循环、GQA、右下对齐因果mask及独立NQ/NKV |

重要复核：FA3源码明确是奇偶KV的split-KV双消费者，并非可以直接替换为单消费者FA2。basic的book默认Bc=64（不可依赖历史摘要中Bc=16的说法）。persist-D的因果约定为k_pos≤q_pos+NKV−NQ，区别于已完成训练Attention的左上对齐约定。

AMD原生设计使用gfx1201 FP16 WMMA/FP32累加。NVIDIA TMA替换为真实128位global load、VGPR等待、LDS发布及acquire/release序列计数。保留各种调度机制，明确记录适配64KiB LDS时改变的CTA大小；不将CuTe/WS/FA3/persistent统一别名到现有FFPA。

初始几何计划：合作路径Br64；单WS与FA3路径Br32；D128缩小Bc，FA3每组独立存储。K容量4、V容量1或2，运行时stage决定实际活动槽数及预取距离；固定容量与性能影响单独标注。persist-D使用K/V容量2。FA3合并复用完成计算后已失效的QKV LDS，避免额外全局中间矩阵。

验收将覆盖：两张AMD卡、5090原源码和显式精度参考、CPU FP64、Graph与非默认stream、有限值/尾部/多头/GQA/因果空行、独立stage推进及延迟角色审计、pad/XOR/CuTe坐标、copy逐位一致、P寄存器转换控制、persistent CTA多任务及Q仅加载一次、ISA和同机SDPA性能。原CUDA的精度/同步/资源失败保留原证据，修复参考单独记录。

状态：八路径功能、双卡、原CUDA/显式修复、1280份共同输出和独立C++646项均通过；14/16生产特化spill及SDPA性能差距保持开放。本文保留初始设计，正式实现与限制以验收报告为准。
