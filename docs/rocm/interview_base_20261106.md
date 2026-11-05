# 独立 interview base / SGEMV 验收

2026-11-06。**base.cuh的19个GPU kernel与sgemv.cuh的3个GPU kernel，合计22个，已完成功能与跨平台验收。**
它们属于独立C++补充范围，不加入原229个Python扩展接口，也没有新增Python算子API。

| 原头文件 | 完成的kernel |
| --- | --- |
| base.cuh | block_reduce_all；dot/dot_vec4；relu/relu_vec4；elementwise_add/elementwise_add_vec4；histogram；merge_attn_states；softmax_per_token/safe_softmax_per_token/online_safe_softmax_per_token；rms_norm/rms_norm_vec4；layer_norm/layer_norm_vec4；rope；mat_transpose/mat_transpose_padded |
| sgemv.cuh | sgemv_k32、sgemv_k128、sgemv_k16 |

## 实现与契约

原SIMT、标量/float4读取、分段shuffle、原子计数/归约、online softmax、RoPE和padded转置算法保留。
`common.cuh`增加明确的HIP SIMT分支，base的FP16头文件与shuffle掩码类型适配HIP。
未提供任何无操作PTX替身；矩阵与异步指令由后续独立AMD实现负责。
当前编译明确限制gfx1201/wave32，其他架构没有资格声明。

这是原始C++ kernel层的适配，调用者必须提供合法launch geometry与数据域：

- dot_vec4沿用N按256对齐；relu_vec4要求N按4对齐。标量和add_vec4另测N=1/17/255/257尾部。
- 三种softmax逐CTA处理一行，行宽与block线程数一致；norm标量线程数为K、vec4线程数为K/4。本次K128/256/1024。
- SGEMV K32/K128路径的K分别按32/128对齐，覆盖M尾部；K16为K=16、每warp两行，测试M按8对齐。
- padded转置按64行×16列tile、RoPE按完整launch范围；没有声称增加原例未实现的任意尾部支持。
- merge_attn_states沿用原LSE的正/负无穷空分区标记；覆盖单侧空分区。双侧均空的定义未扩展。
- softmax普通版本仍有原exp输入域限制；不将数值稳定版本与普通版本混成同一别名。

## 结果

| 检查 | 结果 |
| --- | --- |
| AMD主卡，物理GPU4/PCI83 | 370/370 |
| AMD第二卡，物理GPU0/PCI03 | 370/370 |
| RTX5090原头文件 | 370/370 |
| 共同输入逐输出比较 | 370/370，其中250项逐位一致 |
| 覆盖组成 | 192项CPU FP64/精确整数参考检查 + 178项浮点kernel Graph replay |
| C++运行方式 | 独立可执行文件，使用非默认stream；不链接Torch、c10或Python |

使用同一冻结little-endian FP32二进制输入；两个固定片段与多种shape，记录文件/源码/可执行文件/输出SHA256。
Graph在capture之后把输出重置为NaN，再replay并对独立参考复验；校验非空图及输出前后canary。
Histogram的整数原子路径单独精确校验，不计入178个浮点Graph检查。
逐元素与布局操作要求精确相等；norm/softmax/merge/RoPE为atol=rtol=2e-5；reduce/dot/GEMV为atol=2e-4、rtol=2e-5。
CPU oracle使用读入的真实FP32输入、双精度算术或精确整数计数。未放宽失败样本门限。

初版每端132项已经通过；扩展尾部与Graph后达到370项，旧报告保留。
首轮编译中的平台宏断言与ISA相对路径错误已修复并保留日志，不计为通过记录。
两端各192个原调用配置另记录5次预热后100次事件平均耗时，包含提交间隙；不是与Torch库的加速比较，也不是跨GPU优化倍数。

## 构建与证据

```bash
# 生成/核对输入，不使用GPU。
python3 scripts/rocm/make_standalone_fixtures.py --output artifacts/rocm/continuation_all/interview/fixtures.bin
# 先检查占用/PCI/预约并选择允许的卡；物理GPU3禁止计算。
HIP_VISIBLE_DEVICES=6,2 scripts/rocm/run.sh python -m leetcuda_rocm.standalone_check run \
  --backend hip --performance \
  --fixtures artifacts/rocm/continuation_all/interview/fixtures.bin \
  --output artifacts/rocm/continuation_all/interview/recheck.json
```

冻结输入生成方式与SHA见[manifest](interview_base_20261106/fixtures.json)。本机原始输入/输出位于`artifacts/rocm/continuation_all/interview`。
CUDA使用5090专用目录、锁定镜像及上游`3dbd618`的原头文件。AMD和CUDA使用相同C++独立参考与检查驱动，编译不同后端。

- [主卡](interview_base_20261106/base_extended_hip.json)、[第二卡](interview_base_20261106/base_extended_second.json)
- [5090原版](interview_base_20261106/base_extended_cuda.json)、[逐输出比较](interview_base_20261106/base_extended_comparison.json)
- [源码审计](interview_base_20261106/source_audit.json)

本项关闭两组独立头文件的当前验收，不代表整个interview或整体迁移完成。
继续SGEMM/HGEMM/独立FlashAttention、CuTeDSL/swizzle/Nsight示例、4个未绑定计算入口和性能收敛。
