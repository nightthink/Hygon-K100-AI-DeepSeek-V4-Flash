# 在 8 张海光 K100-AI 上运行 DeepSeek-V4-Flash：从零到生产

在 8×海光 K100-AI 64GB（gfx928，DTK 26.04）上，将 **DeepSeek-V4-Flash**（284B MoE，激活 13B，自行量化 W8A8）部署为 OpenAI 兼容服务的完整实录：三条路线全部跑通，含全部补丁、启动器、测试脚本、32 轮调优记录，以及提交给海光的两份详细 Bug 报告。

## 定稿配置：0731 + DSpark

主线已从 4 月版 + MTP 切换到 **0731 正式版 + DSpark 投机解码**，单流解码提升 **+78%**：

| 指标 | 旧主线（4 月版 + MTP） | **现主线（0731 + DSpark）** | 变化 |
|---|---|---|---|
| 单流解码（编程任务） | 18.7 tok/s | **33.2 tok/s** | **+78%** |
| DSpark accept len / rate | — | **4.38 / 0.68** | 一次猜 5 个中 4 个多 |
| 23K prompt prefill | 343 tok/s | **437 tok/s** | **+27%** |
| KV 池容量 | 600,832 | **1,026,816** | **+71%** |
| 8 / 10 并发（贪心） | 59 | 8/8、10/10 全通 | 稳定 |
| 模型能力（DeepSWE） | 7.3 | **54.4** | 官方基准 |

切换手法：新配置以探针容器启动并跑完全套验收后，直接 `docker rename` 成正式容器名，
**零停机、不重跑 14 分钟的 JIT 与图捕获**。

需要三个补丁（见 `patches/sglang-0811/`）：同步加载、**triton 路由（自研，上游至今缺失）**、dflash renorm 兜底（移植自海光 2026-08-13 提交）。

### ⚠️ 护栏一：只能跑贪心（已定位到具体分支）

| 场景 | 可用性 |
|---|---|
| 贪心（temperature=0），1/2/4/8/10 并发 | ✅ 全部稳定 |
| **temperature>0 + `top_k=1`，8 并发** | ✅ **8/8 通过**（聚合 48.9 tok/s） |
| temperature>0 + `top_k≥2`，8 并发 | ❌ `HSA_STATUS_ERROR_EXCEPTION 0x1016` 或 detokenizer 卡死 |

`top_k=1` 数学上等价贪心但**走完整采样代码路径**，它通过说明：采样路径本身与随机性本身都不是诱因，
**触发条件是「拒绝后重采样」分支被密集执行**（accept rate 从 0.6+ 崩到 0.25 时故障出现）。

这个定位解释了此前说不通的疑点：`SGLANG_DSPARK_FORCE_TORCH_ACCEPT` 无效，是因为它换掉的是
**接受判定**内核，不是**拒绝重采样**分支。放宽接受阈值同样无效——那两个阈值参数根本不作用于采样路径。

**实用缓解**：网关层对 `temperature>0` 的请求强制注入 `top_k=1`（API 兼容性保住、服务不崩，
代价是输出实质确定性），或路由到无投机备用线（`scripts/start_0811_nospec.sh`，8 并发 52.2 tok/s，任意温度稳定）。

详见 `docs/调优记录-轮次29.md`、`docs/调优记录-轮次30.md`。

### ⚠️ 护栏二：Think 改为按请求显式开启

0731 线上 `reasoning_content` 不再默认产出，需要请求携带 `{"chat_template_kwargs": {"thinking": true}}`。
不带该参数时答案本身正确，但 `reasoning_content` 为空。Tool Call 无此要求。
**依赖 reasoning_content 的下游客户端需同步改造**。

### ⚠️ 护栏三：`--kv-cache-dtype bfloat16` 静默产出乱码

本线的解码内核 `triton_fp8_attention_fwd` 按 **fp8 布局**读 KV。改成 bf16 后服务照常启动、
不报任何错误，**性能指标全线"变好"**（单流 33.8→44.1、accept rate 0.68→**1.00**），
但输出是彻底的乱码。`accept rate 恒为 1.00` 是这类失效的关键告警信号。

**fp8 KV 是本线必需项**，与 0728 线结论相反：两条线走不同内核，结论不可互相套用。详见 `docs/调优记录-轮次25.md`。

## 为什么停在 33 而不是 50：我们在用 2 倍的权重跑这个模型

**应用层参数调优已穷尽**（轮次 25–28、30、31 共否决 9 项），瓶颈是**权重访存带宽**。
但轮次 31 查权重规格时发现了更根本的一层：

| | 目录大小 | 每参数（284B 计） |
|---|---|---|
| 原始 `DeepSeek-V4-Flash-0731` | **156 GB** | ≈ **4.4 bit** |
| 我方 W8A8 | **292 GB** | ≈ **8.2 bit** |

模型的路由专家**原生就是 4-bit**（专家张量占 90.3% 字节）。解码每步只有约 6 个 token
（DSpark 验证窗口）× topk 6 = 36 行，摊到 256 个专家上平均每专家不足 1 行——
耗时几乎全是"把 256 个专家的权重从 HBM 读一遍"。**我们正在以模型原生设计约 1.9 倍的权重流量运行。**

进一步查明 sglang 其实认得这个布局（`Auto-detected DSV4 routed-expert layout: is_fp4_experts=True`），
并有完整的后端映射，但在 gfx928 上全部落空：

| `--moe-runner-backend` | fp4 专家路径 | gfx928 |
|---|---|---|
| `auto`（默认） | **反量化 fp4 → fp8** | 能跑，但等于 8-bit |
| `marlin` / `flashinfer_mxfp4` | Mxfp4Marlin / trtllm | ❌ CUDA 专属 |
| **`humming`** | **Mxfp4Humming** | ❌ **集成代码已发布，`humming` 包未随镜像发布** |

根源是各内核库的 offload-arch 覆盖不一致，以及打包遗漏：

| 库 | 编进去的架构 | 本卡（gfx928） |
|---|---|---|
| `sgl_kernel`（sglang 主内核库） | gfx906 gfx926 **gfx928** gfx936 gfx938 | ✅ |
| `aiter`（AMD 内核库） | **仅 gfx938** | ❌ |
| `deep_ep`（专家并行 a2a） | **仅 gfx936 gfx938** | ❌ |
| `humming`（原生 mxfp4） | **模块不存在** | ❌ |

这张表也解释了启动器里为什么要把一长串 aiter 开关全部关掉——**不是调优偏好，是整个库没有本卡的设备代码**。
用 `tests/probe_arch.sh` 可在几十秒内复现。

**所以 33 tok/s 的天花板目前由厂商构建与打包决定，不是硬件决定。**
详见 `docs/调优记录-轮次28.md`、`docs/调优记录-轮次31.md`、`docs/调优记录-轮次32.md`
与 `docs/Bug报告-gfx928构建与打包缺口.md`。

全过程（含权重管线、四关突破、测量陷阱）见 **`docs/0731升级与DSpark实战.md`**。

## 成果一览

| 路线 | 状态 | 单流解码 | 聚合吞吐 | 定位 |
|---|---|---|---|---|
| **sglang 线 · 0811 镜像 + DSpark** | ✅ **主线**（限贪心） | **33.2 tok/s** | 8 并发 8/8、10 并发 10/10 | 0731 模型 + DSpark，1M 上下文 |
| **sglang 线 · 0811 镜像 + 无投机** | ✅ 采样场景备用 | ~12.3 tok/s | 8 并发 52.2（任意温度稳定） | 需要 temperature>0 且高并发时使用 |
| **sglang 线 · 0728 镜像 + MTP** | ✅ 回退线 | 18.7–18.8 tok/s | 8 并发 ~59、10 并发 ~66 | 默认开 Think，1M 上下文 |
| **vLLM 线**（hy3-0706 镜像 + 6 补丁） | ✅ 回退线 | ~9.3 tok/s | 128 并发 ~116 tok/s | 高并发吞吐强项 |
| **FlagOS 线**（vllm-plugin-fl + 19 补丁） | ✅ 精度参考 | 性能不足 | — | 数值对齐基准 |

长上下文实测（0728 线）：23K prompt prefill ~343 tok/s（TTFT ~45s），98K ~395 tok/s（~4.1min），前缀缓存命中 TTFT 0.8s。

## 相关仓库

本项目不重复造轮子，三条路线分别建立在下列上游之上。列出它们在本项目中的**实际用途**，
便于复现者直接定位来源。

### 模型

| 仓库 | 在本项目中的作用 |
|---|---|
| [deepseek-ai/DeepSeek-V4-Flash](https://github.com/deepseek-ai/DeepSeek-V4-Flash) | 模型主页与技术说明；0731 正式版的 DSpark 结构、`dspark_block_size` 等 config 字段以此为准 |
| [HuggingFace · deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 权重来源（本仓库不含权重，需自行下载后按 `scripts/` 量化）；`generation_config.json` 的本地部署推荐参数出处（`do_sample=true`、`temperature=1.0`——即官方默认就是采样） |

### 海光（HYGON-AI）

| 仓库 | 在本项目中的作用 |
|---|---|
| [HYGON-AI/sglang-das](https://github.com/HYGON-AI/sglang-das) | 海光 sglang 分支，主线的上游。**报障目标仓库**（分支 `v0.5.15.post1_dev`）；`dflash renorm` 兜底补丁移植自其 2026-08-13 提交 `18e167b7`；MoE 调优器取自其 `benchmark/kernels/fused_moe_triton/`；本仓库两份 `docs/Bug报告-*.md` 即针对该仓库 |
| [HYGON-AI/vllm-plugin-das](https://github.com/HYGON-AI/vllm-plugin-das) | 海光 vLLM 插件，回退线的上游；`patches/vllm/` 的 6 个 gfx928 正确性补丁针对它 |
| [HYGON-AI/inference-cookbook-das](https://github.com/HYGON-AI/inference-cookbook-das) | 海光官方推理示例与启动参数参考。注意其启动器面向 BW 卡（gfx936/938），**在 K100-AI 上照抄必崩** |

### FlagOS（智源 / flagos-ai）

| 仓库 | 在本项目中的作用 |
|---|---|
| [flagos-ai/DeepSeek-V4-FlagOS](https://github.com/flagos-ai/DeepSeek-V4-FlagOS) | 精度参考线的模型侧支持；**权重管线的反量化脚本 `convert_weight.py` 来自这里**（FP4/FP8 → BF16，我方以 `--device cpu` 运行） |
| [flagos-ai/vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) | FlagOS 线所用的 vLLM 插件（我方在其上打了 19 个补丁才跑通，性能不足但可作数值对齐基准） |
| [flagos-ai/community](https://github.com/flagos-ai/community) | FlagOS 社区仓库：路线图、支持矩阵与问题反馈入口 |
| [flagos-ai/EasyOfUse](https://github.com/flagos-ai/EasyOfUse) | FlagOS 的易用性工具与部署示例集合 |

### 上游框架

| 仓库 | 在本项目中的作用 |
|---|---|
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 主线框架上游。本仓库 `patches/sglang-0811/patch_triton_backend.py`（triton 路由）针对的是 `debug_flash_mla_adapter.py`，该缺口在上游与海光分支中至今均未修复 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 回退线框架上游 |

## 仓库结构

```
docs/
  在8张K100-AI上运行DeepSeek-V4-Flash-从零到成果.md   # 主文档：三条路线从零到结果，含全部失败尝试与原因
  0731升级与DSpark实战.md                              # 0731 升级 + DSpark 攻坚全过程（含测量陷阱）
  2026-08-14-当日总结.md                               # 【总账】轮次 26-32：5 项成果、9 项否决、方法论、数据勘误
  Bug报告-提交海光.md                                  # A/B/C/D/E/F 六类 20+ 项问题
  Bug报告-DSpark与triton路由.md                        # 代码层三问题：triton 路由缺失、DSpark 并发异常、bf16 KV 乱码
  Bug报告-gfx928构建与打包缺口.md                      # 构建/打包四问题：DeepEP 无 gfx928、EP 段错误、低比特算子缺失、humming 包未发布
  启动手册-sglang线.md / 启动手册-vLLM线.md            # 两条线的启动手册
  调优记录-全程.md                                     # 轮次 1–23 全记录（含所有否决项及证据）
  调优记录-轮次24.md                                   # DSpark 并发崩溃深挖（对照实验方法论）
  调优记录-轮次25.md                                   # 参数再优化三项全否 + bf16 KV 乱码陷阱
  调优记录-轮次26.md                                   # MoE 内核调优否决，瓶颈定位为权重访存带宽
  调优记录-轮次27.md                                   # cuda graph 假设被证伪 + 主线切换到 DSpark
  调优记录-轮次28.md                                   # EP 与 4-bit 两条路均卡在 gfx928 构建缺口
  调优记录-轮次29.md                                   # 采样崩溃定位到「拒绝重采样分支」
  调优记录-轮次30.md                                   # 接受阈值绕行失败，阈值管不到采样路径
  调优记录-轮次31.md                                   # 最后三项全否；发现我们在用 2 倍权重跑模型
  调优记录-轮次32.md                                   # 摸清原生 4-bit 每条路；humming 集成已发、库未发
patches/
  sglang/         # 0728 镜像：同步加载补丁 + 参数化启动器 ×5 + Dockerfile
  sglang-0811/    # 0811 镜像：triton 路由补丁 + dflash renorm 补丁 + 诊断补丁 + gfx928 启动器
  vllm/           # 6 个 gfx928 正确性补丁 diff + Dockerfile
tests/
  probe_arch.sh            # 【先跑这个】gfx928 构建目标探针：几十秒判断某功能内核是否为本卡编译
  probe_sampling_crash.py  # DSpark 采样崩溃最小复现（top_k=1 通过 / top_k=2 崩）
  probe_int4_triton.py     # Triton int4 MoE 探针（已确认为 gfx928 编译执行；数值验证待真实 AWQ 权重）
  run_ab_experiments.sh    # 顺序 A/B 对照编排（必带基线组、完整长度热身、每组换前缀）
  test_dv4.sh              # 7 项验收（health/对话/数学/素数/流式/速度）
  bench_concurrency.py     # 并发压测
  bench_conc_param.py      # 参数化并发压测（并发/温度/top_k，用于隔离崩溃）
  ttft_test.py             # TTFT 与前缀缓存收益
  longctx_test.py          # 长上下文测试（23K/98K 级）
  longctx_fresh.py         # 长上下文 prefill 测速（每次变前缀，避开缓存与 JIT 污染）
  test_coding_speed.py     # 编程任务单流速度
  preflight_0731.py        # 0731 权重结构预检（DSpark 张量分流核对）
  warmup_then_bench.sh     # 非贪心路径预热验证（预热后安全并发 2→4）
scripts/
  start_prod_dspark.sh       # 【主线】0731 + 0811 镜像 + DSpark（含贪心护栏说明）
  start_sglang_dsv4_prod.sh  # 旧主线（0728 + 4 月版 + MTP），现为回退线
  start_0731_base.sh         # 0731 + 0728 镜像（无投机基线）
  start_0811_nospec.sh       # 0731 + 0811 镜像 + 无投机（采样场景备用 / 对照实验）
  quant_w8a8_0731.py         # 0731 量化（含 DSpark 张量取舍开关）
  run_0731_pipeline.sh       # 反量化 → 量化 全流程
  run_quant_stage3.sh        # 单独重跑量化阶段
  moe_tuning/                # MoE 内核调优：调优器补丁、裁剪搜索空间、EP 启动器构造说明
```

## 关键技术点

1. **多线程 H2D 拷贝死锁**（加载假死 8 小时）：py-spy 定位到 sglang 异步加载 + DTK HIP 运行时的线程安全问题，`should_async_load → False` 补丁修复，加载缩至 ~3 分钟。
2. **vLLM 线 6 处 gfx928 正确性 bug**：融合算子写坏 KV 缓存、indexer 写读格式矛盾、转置 view 静默乱码等，逐一根因定位并给出补丁。
3. **KV cache dtype 与内核强绑定**：0728 线（MTP）必须 bf16；0811 线（dsv4 triton 解码）反过来必须 fp8，bf16 会静默乱码。**结论不可跨线套用**。
4. **投机解码三代对比**：EAGLE MTP（0728 线，1.55×）→ DSpark（0811 线，2.75×，accept len 4.38）；vLLM 线 MTP ≥8 并发 VMFault（已报障）。
5. **把"不支持"的镜像救活**：0811 镜像的 gfx928 障碍全在 env 选错路径与一处未接线的 triton 分支，逐关突破后可用。
6. **测量陷阱**：长上下文 prefill 曾先后测得 232 / 2144 / 437 tok/s ——分别是 triton 首次 JIT 污染、前缀缓存命中、真实值。单流解码同样中招且更阴：首测 11.4–12.3，**恰好接近"无投机"线的 12.3**，极易误判为投机没生效，热身充分后才是 32.5–33.2。**压测前必须用完整长度的请求热身两次**。
7. **性能变好要怀疑，变差也要怀疑**：bf16 KV 让指标全面变好（accept rate 冲到 1.00）却输出乱码；强开共享专家融合让指标全面变差（accept rate 崩到 0.03）同样输出乱码——但只看预填吞吐（持平）会漏掉。**任何配置改动都必须过一遍数学 / Think / Tool Call 再采信**。
8. **用对照实验定位崩溃**：面对看不到栈的 GPU 故障，最有效的手段是**固定其它变量、只切换嫌疑组件**。第 24 轮一次无投机对照排除了镜像/权重/注意力后端三个方向；第 29 轮用 `top_k=1` vs `top_k=2` 把采样崩溃钉到了拒绝重采样分支。
9. **醒目的可疑点未必是真问题，先取证再调优**：轮次 26 的"MoE 配置目录为空 + 满屏 sub-optimal 警告"实测只值 +1%；轮次 27 的"cuda-graph-max-bs 显然没覆盖验证批"被一行启动日志直接证伪。两次取证成本都在十几分钟。
10. **先查构建目标，再决定要不要做实验**：轮次 28 用 `strings <lib>.so | grep gfx` 与直接取 `torch.ops` 算子两招，几十秒否掉了两条原本各需半天到一天的路线。工具见 `tests/probe_arch.sh`。
11. **框架的默认值是保护性的**：轮次 31 强行覆盖三处框架自行关闭的优化（共享专家融合、MoE runner、预填图），三处全部失败——一处静默乱码、两处直接崩溃。
12. **零停机切换**：新配置以探针容器启动、跑完全套验收后 `docker rename` 成正式容器名——验证过的进程就是可用进程。

## 说明

- **模型权重不在本仓库**（4 月版 279GB / 0731 版 156GB 原始，需自行量化）。
- 文档中 `<NODE_A_IP>` / `<NODE_B_IP>` / `nodeA` / `nodeB` / `<internal-harbor>` / `/home/user` 为脱敏占位符。
- 基础镜像来自海光 sourcefind 仓库（`harbor.sourcefind.cn:5443`），补丁以 Dockerfile 固化为新镜像。

*文档基线日期：2026-08-14。*

---

Copyright © 2026 DaoTech Team. Licensed under the MIT License.
