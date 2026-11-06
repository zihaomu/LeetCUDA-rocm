# ROCm 10 第二阶段性能优化验收

2026-11-06。独立FA、FP8量化搬运、FFPA WS、D256反向四项均完成本轮优化与验收。基线为`87edbab`，只修改四个计算实现文件；接口、数值门限及已支持算子数量不变。FP4已[独立立项](fp4_research_goal.md)，本轮没有宣称FP4获得新加速。

## 最终实测

以下为**同一AMD卡上，冻结旧实现耗时／新实现耗时的几何平均**，不是相对5090或Torch库的加速。每项完整执行三轮，所有被测配置均满足回退≤10%、轮间波动≤5%；优化目标配置几何平均≥1.05×。未修改的路径只要求稳定性与无回退，不要求额外加速。

| 工作包 | 固定优化目标 | 最终加速 | 额外回归范围 |
| --- | --- | ---: | --- |
| 独立FA | 16种Kind/D × N512/2048，B1/H2 | **1.202×** | 7种调度全部32配置 |
| FP8完整在线GEMM | BF16输入，M/N=256，K256/1024；E4M3/E5M2、4种scale、2种调度，共32配置 | **1.577×** | 纯计算、离线B、小形状B量化，共108配置 |
| FP8 B量化准备 | 同上两种矩阵及N136/K80小形状，共12配置 | **1.811×** | 宽短矩阵保留旧路径；另做80配置扫描 |
| FFPA WS | B1/H2/N384，9种D × 5种stage，共45配置 | **1.073×** | async/non_ws合计90配置，三轮全部无回退 |
| D256训练Attention | N1024、H4/HK4、causal，FP16/BF16前向+反向 | **1.087×** | D64/D128及所有前向，共12配置 |

三轮全部配置中的最大轮间波动分别为FA **4.25%**、FP8 **4.86%**、FFPA **4.08%**、训练Attention **2.28%**。波动按`(最大轮耗时−最小轮耗时)/轮耗时中位数`计算，对新旧两端分别检查。最差回退分别为0.013%、0.303%、0.293%、0.598%。每轮每配置保留5次交替采样，每次目标40ms；计时前双方配对预热200ms。

正式性能卡：FA/FFPA为物理GPU5（PCIa3），FP8为物理GPU6（PCIc3），训练Attention为物理GPU4（PCI83）。使用锁定ROCm10.1.0/HIP7.16/Torch2.14/gfx1201；物理GPU3未参与计算，已预约的物理GPU0也未使用。

## 改动与剩余问题

**独立FA**将Q/K连续、对齐的8个FP16 LDS读取表达为128位向量读取，减少编译器拼装临时值。实际加载二进制中，生产特化spill由14/16降为13/16，最大scratch由1408B降为752B；LDS容量不变。仍有13个生产特化存在spill，不能称为已经完全解决寄存器压力。

**FP8**将较大B矩阵的scale归约与分块量化/转置拆成两个kernel，向量读取BF16、配对转换FP8并合并写入Bᵀ。scale仍使用原FP32 `amax/max_finite`；通过BF16幅值恢复原amax后求inverse，保持原舍入结果。对全部满足原FP32正常scale/inverse契约的正有限BF16幅值穷举验证了恢复恒等式，GPU码点与scale继续接受独立CPU逐位检查。

新路径仅在`K>=256 && N>=64 && (N<=512 || K>=1024)`使用。初版在N2048/K256的4个格式/scale组合慢16%～26%，已保留失败证据并恢复这类宽短矩阵的旧路径。最终补扫K256/512/1024/4096、N64/136/512/2048/8192、两格式与两种B scale，共80配置，码点与scale对冻结旧实现逐位一致，未见>10%回退。该扫描为单轮3次20ms探索采样，不能替代主表的三轮正式验收，也不代表所有未测形状都保证提速。

**FFPA WS**仅在WS消费者中向量读取Q/K及P；保留producer wave、八个consumer waves、QK/V独立环及stage语义。async/non_ws沿用原读取方式。27个编译特化仍全部零spill/scratch；async/WS静态LDS仍为61888B，尚未解决占用率和小矩阵效率问题。

**D256反向**仅将delta重计算改为8个wave，dQ/dK/dV保持原配置。在固定D256性能输入上，delta本身约加速2.6～2.8×，spill由173降至0；整体前向加反向快约8.7%。保留基于未舍入概率重算delta的精度策略；dQ/dKV及前向的spill仍存在。

## 正确性与跨平台回归

| 范围 | 最终源码验收 |
| --- | --- |
| 独立FA | 双AMD各2578项；C++646项；公开API/Graph14项；对冻结5090匹配参考1280项通过 |
| FP8 | 双AMD上E4M3/E5M2每格式各756项；C++每格式93项；共同输入800项；对冻结5090输出480项通过 |
| FFPA | 双AMD各2755项；C++729项；对冻结5090修复参考1700项通过 |
| 训练Attention | 双AMD各377项；D256单有效key、全mask、空序列、大logits、dropout/GQA等额外64项；对冻结CUDA算法输出336项通过 |

保持原误差门限，覆盖非默认stream、Graph重放、边界、stage与延迟注入、设备契约、线性显存检查。上表中不同验证层存在重叠，不把这些检查相加称为独立算子数。

本轮重新计算AMD输出，**复用此前已冻结并校验SHA的5090输出，没有重新执行NVIDIA实验**。FA仍使用原输出与显式slot/warmup、空mask修复组成的匹配参考；FFPA使用已记录的stage3尾部修复参考。最初误用FFPA原CUDA输出时重现了已知18项错误，失败比较保留，随后对正确标记的修复参考1700项全部通过。训练Attention比较明确保留新旧不同源码SHA，仅AMD的delta启动配置发生优化。

此前15163项统一验收及原229接口/462组对照属于历史完整运行。本轮针对四个受影响实现重跑上述专项，未把历史结果改写成一次新的全库执行。统一审计会同时校验原报告、未改源码和四项新证据。

## 未采用的候选

| 候选 | 结果与处理 |
| --- | --- |
| FA producer逐向量立即提交 | 2578项正确，性能0.908×；破坏访存重叠，拒绝 |
| FA初次三轮测量 | 平均有加速，但部分配置波动>5%；保留为不稳定结果，换空闲卡并增加预热后完整重测 |
| FFPA分层消费者屏障 | 2755项正确，WS约1.001×；未达到收益目标 |
| FFPA缩短轮询等待 | 单卡2752项正确，WS约1.000×；未采用 |
| D256全部反向改8/16 waves | 分别约0.993×/0.569×，未采用 |
| D256缩小tile、缩减K tile并改变KV tile | 分别约0.493×/0.627×，未采用 |
| FP8不区分宽短矩阵的两kernel分派 | 4项出现>10%回退；最终加入保留旧路径的分派 |

另保留编译探针候选于本机`artifacts/rocm/performance_phase2/`；未做完整执行的候选不宣称通过正确性或性能。

## 复现与证据

无需GPU即可核对已提交证据、源码SHA、每个配置的三轮门槛：

```bash
python3 scripts/rocm/audit_performance_phase2.py
python3 scripts/rocm/audit_final_reports.py
```

测量前再次核对GPU预约和占用。以下恢复冻结基线，使用当前四个实现与旧实现同卡配对；FFPA性能依赖原冻结fixture，可由现有`ffpa_check fixtures`生成。

```bash
python3 docs/rocm/performance_phase2/harness/prepare_baseline.py
HIP_VISIBLE_DEVICES=7 scripts/rocm/run.sh python docs/rocm/performance_phase2/harness/fa/evaluate.py performance \
  --header kernels/rocm/interview_flash_attention_hip.cuh --rounds 3 --warmup-ms 200 \
  --output artifacts/rocm/performance_phase2/recheck_fa.json
HIP_VISIBLE_DEVICES=7 scripts/rocm/run.sh python docs/rocm/performance_phase2/harness/fp8/performance.py \
  --header kernels/rocm/fp8_gemm_hip.cuh --rounds 3 \
  --output artifacts/rocm/performance_phase2/recheck_fp8.json
HIP_VISIBLE_DEVICES=7 scripts/rocm/run.sh python docs/rocm/performance_phase2/harness/ffpa/evaluate.py performance \
  --header kernels/rocm/ffpa_attention_hip.cuh --rounds 3 \
  --output artifacts/rocm/performance_phase2/recheck_ffpa.json
HIP_VISIBLE_DEVICES=7 scripts/rocm/run.sh python docs/rocm/performance_phase2/harness/training/evaluate.py performance \
  --source leetcuda_rocm/attention_training.py --rounds 3 \
  --output artifacts/rocm/performance_phase2/recheck_training.json
```

[证据与源码清单](performance_phase2/manifest.json)记录每份原始报告及冻结基线hash；[FA](performance_phase2/fa/performance.json)、[FP8](performance_phase2/fp8/performance.json)、[FFPA](performance_phase2/ffpa/performance.json)、[D256](performance_phase2/training/performance.json)包含所有样本与二进制身份。完整二进制、ISA、tensor夹具与输出保留在本机实验目录；提交的报告不是自包含二进制分发包。实时过程见[进度文档](performance_phase2_live.md)。

本轮达成的是固定工作负载相对旧AMD实现的性能提升，仍未达到所有形状的成熟库性能、独立FA零spill或多AMD架构资格。
