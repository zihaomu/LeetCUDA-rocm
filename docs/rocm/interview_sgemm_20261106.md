# 独立 interview SGEMM 验收

2026-11-06。`sgemm.cuh`四个GPU kernel已完成功能与跨平台验收，不加入原229个Python接口分母。

| 路径 | AMD实现与契约 |
| --- | --- |
| sgemm | 原SIMT算法；32×32×32 tile、1024线程；M/N/K按32对齐 |
| sgemm_vec4 | 原SIMT算法；128×128×32 tile、1024线程、每线程4×4输出及128-bit访存；M/N按128、K按32对齐 |
| f32x4_tf32x4_kernel | RNA TF32舍入，包含正负ties、指数/次正规边界、原地转换；N按4对齐及16字节对齐 |
| sgemm_tf32 | 128×128×8 CTA、4×2个wave、每wave 2×4输出fragment、原默认stage2及padding4；M/N按128、K按8对齐 |

原CUDA头文件分支及两个SIMT函数保持原实现。TF32以两个BF16分量精确表示一个TF32输入，四次BF16 WMMA求一个矩阵乘积，FP32累加；明确保留单TF32算术模型，与B6公开接口的TF32x3精度升级分别标记。
输入按原bench先转换为TF32；参考使用这些实际舍入输入的FP64矩阵乘。未把TF32对未舍入FP32输入的量化误差称为移植误差，也未放宽既有公开接口门限。

双缓冲使用两组含padding的LDS、全局128-bit加载到VGPR、当前tile计算及下一slot发布；等待真实load完成并执行CTA同步。
不使用空操作cp.async替身。TF32矩阵范围外的指数通过真实FP32 FMA回退，保持TF32输入语义；测试成对2^116/2^-116尺度。
只资格默认tile/stage2配置；其他stage编译时拒绝。原始unsafe尾部、额外tile模板配置不在本次原bench契约中。

## 验收结果

| 检查 | 结果 |
| --- | --- |
| AMD物理GPU4/PCI83 | 234/234 |
| AMD物理GPU0/PCI03 | 234/234 |
| RTX5090原CUDA | 234/234 |
| 共同输入逐输出比较 | 234/234；168项逐位一致 |
| Graph replay | 每端79项，capture后NaN重置及canary复验 |
| 统一CLI对base回归 | 370/370 |
| C++执行文件 | 无Torch/c10/Python链接；非默认stream |

三组固定输入片段；SIMT K32～1024、TF32 K8～1024、多CTA矩形及短流水线，另覆盖指数极值回退。
转换使用独立CPU指数网格算术参考并做原地bit比较；矩阵使用CPU FP64。转换精确相等，矩阵atol=2e-4、rtol=2e-5。
每端记录79组5次预热、100次普通event平均耗时；包含提交开销，只提供实测记录，不声称达到Torch库性能或跨GPU加速。

| 内核 | VGPR | LDS | scratch/spill |
| --- | ---: | ---: | --- |
| sgemm | 26 | 8192B | 0/0 |
| sgemm_vec4 | 106 | 32768B | 0/0 |
| TF32转换 | 14 | 0 | 0/0 |
| TF32矩阵 | 166 | 20736B动态+64B标志 | 0/0 |

TF32内核包含32条静态BF16 WMMA指令及128-bit全局读取；其余三个没有矩阵指令，算法未混为别名。
首轮210项记录及两次编译修复日志保留；未将编译失败记为通过。

## 复现与证据

```bash
HIP_VISIBLE_DEVICES=6,2 scripts/rocm/run.sh python -m leetcuda_rocm.standalone_check run \
  --suite sgemm --backend hip --performance \
  --fixtures artifacts/rocm/continuation_all/interview/fixtures.bin \
  --output artifacts/rocm/continuation_all/interview/sgemm_recheck.json
```

执行前仍核对占用、PCI和预约；物理GPU3禁止计算。AMD使用锁定ROCm10.1镜像/gfx1201，5090使用锁定CUDA镜像和上游`3dbd618`。

- [主卡](interview_sgemm_20261106/sgemm_extended_hip.json)、[第二卡](interview_sgemm_20261106/sgemm_extended_second.json)
- [原CUDA](interview_sgemm_20261106/sgemm_extended_cuda.json)、[跨平台比较](interview_sgemm_20261106/sgemm_extended_comparison.json)
- [ISA资源](interview_sgemm_20261106/sgemm_resources.json)、[源码审计](interview_sgemm_20261106/source_audit.json)

当前interview补充范围功能40/53（75.47%），完整跨平台34/53（64.15%）。继续HGEMM、FlashAttention、FP4完整链对照及其他补充范围，整体迁移未完成。
