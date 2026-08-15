# 在 8 张海光 K100-AI 上运行 DeepSeek-V4-Flash：从零到生产

在 8×海光 K100-AI 64GB（gfx928，DTK 26.04）上，将 **DeepSeek-V4-Flash**（284B MoE，激活 13B）部署为 OpenAI 兼容服务的完整实录：**四条线路**全部跑通（含自研 int4 线），39 轮调优记录、十余个自研补丁、完整测试与量化工具链，以及提交给海光的两份详细 Bug 报告。

## 两条 0731 线路：单流线 + 高并发线

**主线（w8a8 + DSpark）**——单流最快：

| 指标 | 旧主线（4 月版 + MTP） | **w8a8 + DSpark** | 变化 |
|---|---|---|---|
| 单流解码（编程任务） | 18.7 tok/s | **33.2 tok/s** | **+78%** |
| DSpark accept len / rate | — | **4.38 / 0.68** | 一次猜 5 个中 4 个多 |
| 23K prompt prefill | 343 tok/s | **437 tok/s** | +27% |
| KV 池容量 | 600,832 | 1,026,816 | +71% |

**int4 线（轮次 35–36 自研）**——高并发/长上下文：

| 指标 | w8a8 线 | **int4 线** |
|---|---|---|
| 每卡权重 | ~36 GB | **20.8 GB** |
| **KV 池** | 1,026,816 | **2,971,136（2.9×）** |
| **64 并发** | 未测（KV 受限） | **64/64 全通，聚合 56–65 tok/s** |
| 单流解码 | 33.2 | 18–20 |
| 数学/Think/ToolCall/needle | 全通 | **全通** |

int4 线 = 全量 284B 专家 AWQ int4（量化 35,328 个矩阵仅 35 分钟，脚本见 `scripts/int4/`），
2.9× KV 池支撑约 30 路 98K 上下文同时驻留（w8a8 约 10 路）。
**混合部署（交互式走 w8a8、长上下文/批量走 int4）与轮次 34 的网关分流结论天然契合。**

两线共用补丁体系（`patches/sglang-0811/`）：同步加载、triton 路由（自研）、dflash renorm、
**moe_align 根修复（自研，见下）**；int4 线另需 3 个 wna16 配套补丁（自研）。

### ⚠️ 护栏一：DSpark 只能跑贪心（已定位到具体分支）

| 场景 | 可用性 |
|---|---|
| 贪心（temperature=0） | ✅ 全并发稳定 |
| **temperature>0 + `top_k=1`** | ✅ 8 并发 8/8（聚合 48.9 tok/s） |
| temperature>0 + `top_k≥2` | ❌ `HSA_STATUS_ERROR_EXCEPTION 0x1016` 或 detokenizer 卡死 |

`top_k=1` 数学等价贪心但走完整采样代码路径——它通过说明诱因是**「拒绝后重采样」分支**被密集执行
（accept rate 掉到 0.25 时故障出现）。实用缓解：网关对 `temperature>0` 强制注入 `top_k=1`，
或路由到无投机备用线。详见 `docs/调优记录-轮次29/30.md`。

### ⚠️ 护栏二：Think 需请求显式携带 `{"chat_template_kwargs": {"thinking": true}}`

不带时答案正确但 `reasoning_content` 为空。依赖该字段的下游需改造。

### ⚠️ 护栏三：`--kv-cache-dtype bfloat16` 静默乱码

0811 线解码内核按 fp8 布局读 KV；bf16 会让**指标全线"变好"**（accept rate 冲到 1.00）而输出全是乱码。
fp8 KV 是本线必需项（与 0728 线结论相反，不可跨线套用）。详见 `docs/调优记录-轮次25.md`。

## 一个 bug 统一了三个"独立"故障：moe_align 根修复（轮次 35）

`fused_moe_triton` 家族此前在本卡上三处"各自"崩溃：EP 朴素 dispatcher 段错误（轮次 28）、
未量化 bf16 小模型 VM fault、int4 内核 VM fault（轮次 33）。经验二分证明**内核本身数值正确**，
真凶是 `moe_align_block_size`：

- 默认 sgl_kernel C++ 版在本卡输出纯垃圾（`num_post = -1093386606`）——刷屏几十轮的
  `Launch params (1024,1,1) > launch bounds (256)` 警告在本卡是**真未定义行为**
- lightop 版排序正确但 **pad 槽位不填充**（-777 哨兵抓现行），残留负数骗过内核
  `token_mask` → 按负偏移读显存 → VM fault
- 生产 w8a8 从不崩，只因 lightop groupgemm 路径不经过它

**修复两行**：换 lightop 算子 + 缓冲区预填充（`patches/sglang-0811/patch_moe_align.py`）。
上游 sglang 已在 #32395 重构该路径，0811 镜像早于该修复。
修复直接解锁了 int4 线。全过程见 `docs/调优记录-轮次35.md`。

> **轮次 38 补记**：生产 w8a8 启动器 `run_0811_probe.sh` 一直**没打**这个补丁——它 4 小时不崩只是 UB 侥幸；DSpark verify 阶段其实一直在碰这个 C++ 内核。重启后同一处刷 `launch bounds (1024>256)` 崩溃。已派生 `run_0811_prodfix.sh`（`patches/sglang-0811/mk_prodfix_launcher.py`）在生产线补上该补丁。补上后 launch-bounds 消失，但 w8a8 int8 验证图捕获还剩一个**独立的** `hipErrorInvalidValue`（坑 B，待配置绕开）。int4 线用同一套捕获约 29 秒跑完并正确服务，据此判定 **GPU 健康、非 wedge**，避免了一次会误伤宿主 10+ 台 VM 的整机重启。详见 `docs/调优记录-轮次38.md`。

## 单流为什么停在 33：天花板的三层定界

1. **应用层参数已穷尽**（轮次 25–31 否决 9 项）：瓶颈是每步从 HBM 搬运的专家权重字节数
   （模型原生 4-bit 156GB，w8a8 跑在 292GB ≈ 1.9× 原生流量）。
2. **构建/打包缺口**（轮次 28/32，报障中）：`aiter` 仅 gfx938、`deep_ep` 仅 gfx936/938、
   低比特预编译算子族未编入、`humming`（原生 mxfp4）集成代码已发但包未随镜像发布。
   `tests/probe_arch.sh` 几十秒复现。
3. **自研 int4 后的新定界**（轮次 36）：权重字节已砍半，但单流反而 18–20——
   (a) 无校准 4-bit 量化使目标分布偏移，DSpark accept rate 0.68→0.30 左右，α-clip 无法挽回；
   (b) `fused_moe_kernel_gptq_awq`（Triton 在核解包）步频与 w8a8 持平（~130ms/步），
   相比 lightop 手工汇编 groupgemm 有结构性差距——**带宽红利没有转化为步速，转化成了 KV 容量**。

要突破 33：校准量化（AutoAWQ 激活感知）拉回接受率 + wna16 内核工程，或厂商侧
`humming` 包 / DeepEP gfx928 构建（每步权重读取量降至约 1/5）。

## int4 与 w8a8 能否「结合」：精度混布被证伪，w4a8 才是正解（轮次 37）

把「结合」落成实验：让 DSpark 取材的最后三层（40/41/42）专家保 **bf16**、其余 40 层 int4，
看接受率是否回升。构建了 226.9GB 混布检查点（`moe_wna16` 对这三层的 `FusedMoE` 返回
`UnquantizedFusedMoEMethod`，走轮次 35 修复的 moe_align 路径），实测 **接受率仍 0.12–0.38
（均值 ~0.24），单流反而降到 15.4**——**证伪**：漂移是全主干累积、不是局部，
真正的解法是**重校准 MTP draft 头以适配 int4 目标**，而非保几层精度。

因此 int4 与 w8a8 的正确结合是：**① 按负载切换**（交互走 w8a8、长上下文/高并发走 int4，
同镜像同补丁）；**② 补丁复用**（moe_align 根修复本为 int4 而生，却反哺了 w8a8 的 bf16 回退路）；
**③ w4a8 合成**（int4 权重 + int8 激活）——int4 服务日志里 `slimquant_w4a8_marlin` 已
`resolved_backend=lightop`，**lightop 侧疑似已有 w4a8 汇编路径**，是把两者合到单条推理路径的
下一步方向。详见 `docs/调优记录-轮次37.md`。

## cuda graph 捕获的硬限：`cuda-graph-max-bs≤16`（轮次 39）

精度混布线并发曲线在 16 并发压平（56.4 tok/s），**24 并发因掉进 eager 反而跌到 55.3**，
于是尝试把 `cuda-graph-max-bs` 提到 32——三种配置**全部在图捕获阶段 `hipErrorInvalidValue` 崩溃**：

| 配置 | 捕获起始空闲显存 | 结果 |
|---|---|---|
| bs=32, mem=0.85, DSpark | 7.39 GB | ❌ |
| bs=32, **mem=0.78**, DSpark | **11.83 GB** | ❌ 排除"显存不足" |
| bs=32, mem=0.85, **无投机**（仅 decode 图 8 档） | 7.18 GB | ❌ 排除"verify 图特有" |

两个对照分别否掉最自然的两种解释 → **bs>16 的批档在本卡上捕获必崩，是硬限而非可调参数**，
并发聚合被锁死在 ~56 tok/s。**这同时解释了轮次 38 的 w8a8 坑 B**——不是 int8 特有 bug，
而是同一个捕获限制（w8a8 verify 图更大，bs16 就已越界），两个"独立故障"再次归一。

⚠️ **运维护栏**：一小时内约 5 次密集重启后，**连原本稳定的 bs16 也开始捕获失败**；
停手静置后恢复完全正常（单流/16 并发/32 并发均回到基线，KV 池不变）。
**本机密集起停会诱发 GPU 暂时性不稳定（非永久损伤）——连续捕获失败时先怀疑重启 churn，
而不是立刻归因于刚改的参数。** 详见 `docs/调优记录-轮次39.md`。

## 成果一览

| 路线 | 状态 | 单流解码 | 聚合吞吐 | 定位 |
|---|---|---|---|---|
| **sglang 线 · w8a8 + DSpark** | ✅ **主线**（限贪心） | **33.2 tok/s** | 8/10 并发全通 | 交互式，1M 上下文 |
| **sglang 线 · int4 + DSpark（自研）** | ✅ **高并发线** | 18–20 tok/s | **64 并发 64/64，56–65 tok/s** | 长上下文/批量，KV 2.9× |
| sglang 线 · 0811 无投机 | ✅ 采样备用 | ~12.3 tok/s | 8 并发 52.2（任意温度） | temperature>0 高并发 |
| sglang 线 · 0728 + MTP | ✅ 回退线 | 18.7 tok/s | 8 并发 ~59 | 默认开 Think |
| vLLM 线（6 补丁） | ✅ 回退线 | ~9.3 tok/s | 128 并发 ~116 tok/s | 高并发吞吐 |
| FlagOS 线（19 补丁） | ✅ 精度参考 | 性能不足 | — | 数值对齐基准 |

## 相关仓库

| 仓库 | 在本项目中的作用 |
|---|---|
| [deepseek-ai/DeepSeek-V4-Flash](https://github.com/deepseek-ai/DeepSeek-V4-Flash) | 模型主页；DSpark 结构与 config 字段依据 |
| [HuggingFace · DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 权重来源；`generation_config.json`（`do_sample=true`——官方默认就是采样） |
| [HYGON-AI/sglang-das](https://github.com/HYGON-AI/sglang-das) | 主线上游，**报障目标仓库**；dflash renorm 补丁移植自其提交 `18e167b7`；镜像内 gptq_awq 内核与其最新分支逐字节一致（轮次35核对） |
| [HYGON-AI/vllm-plugin-das](https://github.com/HYGON-AI/vllm-plugin-das) | vLLM 回退线上游（6 个 gfx928 补丁针对它） |
| [HYGON-AI/inference-cookbook-das](https://github.com/HYGON-AI/inference-cookbook-das) | 官方示例；其启动器面向 BW 卡，K100-AI 照抄必崩 |
| [flagos-ai/DeepSeek-V4-FlagOS](https://github.com/flagos-ai/DeepSeek-V4-FlagOS) | **反量化脚本 `convert_weight.py` 来源**——int4 线的 bf16 源即由它产出 |
| [flagos-ai/vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) | FlagOS 线（19 补丁），数值对齐基准 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 框架上游；triton 路由缺口至今未修；moe_align 已在 **#32395** 重构（0811 镜像早于此） |

## 仓库结构

```
docs/
  在8张K100-AI上运行DeepSeek-V4-Flash-从零到成果.md   # 主文档
  0731升级与DSpark实战.md                              # 0731 + DSpark 攻坚全过程
  2026-08-14-当日总结.md                               # 轮次 26-32 总账：5 成果 / 9 否决 / 方法论
  Bug报告-提交海光.md                                  # 六类 20+ 项
  Bug报告-DSpark与triton路由.md                        # 代码层三问题
  Bug报告-gfx928构建与打包缺口.md                      # 构建/打包四问题
  启动手册-sglang线.md / 启动手册-vLLM线.md
  调优记录-全程.md                                     # 轮次 1–23
  调优记录-轮次24..34.md                               # 单轮记录（各含完整证据链）
  调优记录-轮次35.md                                   # ★ moe_align 根因与修复（一个 bug 统一三个故障）
  调优记录-轮次36.md                                   # ★ 全量 int4 上线与定位
  调优记录-轮次37.md                                   # ★ 精度混布证伪 + int4↔w8a8 结合正解（w4a8 线索）
  调优记录-轮次38.md                                   # ★ w8a8 重启 capture 排障：生产线补 moe_align、int4 无损诊断、避免误伤 VM
  调优记录-轮次39.md                                   # ★ 参数调优三条否决：cuda-graph-max-bs 是硬限；密集重启护栏
patches/
  sglang/         # 0728 镜像补丁与启动器
  sglang-0811/    # triton 路由 / dflash renorm / ★moe_align 根修复 / ★wna16 三补丁
  vllm/           # 6 个 gfx928 正确性补丁
scripts/
  start_prod_dspark.sh       # 主线启动器（w8a8 + DSpark）
  int4/                      # ★ int4 全链路：量化器(α-clip) / 启动器生成 / MoE 配置生成
  moe_tuning/                # MoE 内核调优工具链
  quant_w8a8_0731.py 等      # w8a8 量化管线
tests/
  probe_arch.sh              # 构建目标探针（先跑这个）
  probe_sampling_crash.py    # DSpark 采样崩溃最小复现（top_k 判别）
  probe_int4_triton.py       # int4 内核探针（轮次33，历史）
  awq_int4_probe/            # AWQ 打包 + CPU 往返校验工装（打包正确性证明）
  conc.py                    # 并发聚合压测
  bench_hol.py               # 队头阻塞测试（轮次34：86×，架构性）
  run_ab_experiments.sh      # A/B 编排（基线组/完整热身/换前缀）
  test_dv4.sh / longctx_fresh.py / ...   # 验收与长上下文套件
```

## 关键技术点

1. **多线程 H2D 死锁**：`should_async_load → False`，加载 8 小时假死 → 3 分钟。
2. **vLLM 线 6 处 gfx928 正确性 bug**：逐一根因定位并给出补丁。
3. **KV dtype 与内核强绑定**：0728 线必须 bf16，0811 线必须 fp8——结论不可跨线套用。
4. **投机三代**：MTP 1.55× → DSpark 2.75×（accept len 4.38）。
5. **把"不支持"的镜像救活**：env 选路 + 一处未接线的 triton 分支。
6. **测量陷阱**：JIT 污染 / 前缀缓存命中——压测前必须完整长度热身两次。
7. **性能变好要怀疑，变差也要怀疑**：bf16 KV（指标变好，乱码）与强开共享专家融合（指标变差，乱码）——任何改动必须过数学/Think/ToolCall 再采信。
8. **对照实验定位看不到栈的故障**：无投机对照（轮次24）、top_k=1/2 判别（轮次29）。
9. **先取证再调优**：sub-optimal 警告实测 +1%；cuda graph 假设被一行日志证伪。
10. **先查构建目标再做实验**：`strings *.so | grep gfx` + 直接取 `torch.ops` 算子，几十秒否掉两条半天级路线。
11. **框架默认值是保护性的**：三处强行覆盖全部失败。
12. **零停机切换**：探针容器验收后 `docker rename`。
13. **★ 厂商内核是可修的代码**（轮次35）：Triton 内核就是镜像里的 Python 文件。经验二分（逐字拷贝→变体开关）+ 哨兵值预填充，一个下午找到并修复了被三次归档为"厂商缺陷"的根因——被标注为无害的警告（launch bounds）要定期重审。
14. **量化的代价要在系统层面量**（轮次36）：int4 砍半的权重字节没有变成步速（内核效率差距），却变成了 2.9× KV 池；无校准量化的分布偏移直接反映在投机接受率上——**accept rate 是量化质量的免费在线探针**。
15. **别急着重启，先做无损诊断**（轮次38）：w8a8 停起几轮后起不来，像 GPU wedged。先勘察环境（宿主跑着 10+ 台 VM、桥接封 sudo、无 host rocm-smi），再用**一条已知能起的旁路**（int4，同套 DSpark verify 捕获）区分"GPU 坏"还是"某条路径坏"——int4 正常起来即证 GPU 健康，避免了一次会误伤所有 VM 的整机重启。**先分清是硬件还是软件，再决定要不要动机器。**
16. **否决也是成果**（轮次39）：想提 `cuda-graph-max-bs` 时，用两个对照（多给 4.5GB 显存 / 换成无投机只留 decode 图）分别排除"显存不足"与"verify 图特有"，把 bs>16 捕获崩溃干净定性为**硬限**——并顺带归一了轮次 38 的 w8a8 故障。**把"参数可调"这个假设否掉，等于把后续精力从参数层解放出来。**

## 说明

- 模型权重不在本仓库（0731 原始 156GB / w8a8 292GB / int4 198GB，需自行量化）。
- `<NODE_A_IP>` 等为脱敏占位符；基础镜像来自海光 sourcefind 仓库。

*文档基线日期：2026-08-15。*

---

Copyright © 2026 DaoTech Team. Licensed under the MIT License.
