# 在 8 张海光 K100-AI 上运行 DeepSeek-V4-Flash：从零到生产

在 8×海光 K100-AI 64GB（gfx928，DTK 26.04）上，将 **DeepSeek-V4-Flash**（284B MoE，激活 13B，自行量化 W8A8）部署为 OpenAI 兼容服务的完整实录：三条路线全部跑通，含全部补丁、启动器、测试脚本、27 轮调优记录，以及提交给海光的详细 Bug 报告。

## 生产现状：0731 + DSpark（2026-08-14 切换）

生产已从 4 月版 + MTP 切换到 **0731 正式版 + DSpark 投机解码**，单流解码提升 **+80%**：

| 指标 | 旧生产（4 月版 + MTP） | **现生产（0731 + DSpark）** | 变化 |
|---|---|---|---|
| 单流解码（编程任务） | 18.7 tok/s | **33.2 tok/s** | **+80%** |
| DSpark accept len / rate | — | **4.38 / 0.68** | 一次猜 5 个中 4 个多 |
| 23K prompt prefill | 343 tok/s | **437 tok/s** | **+27%** |
| KV 池容量 | 600,832 | **1,026,816** | **+71%** |
| 8 / 10 并发（贪心） | 59 / — | 8/8、10/10 全通 | 稳定 |
| 模型能力（DeepSWE） | 7.3 | **54.4** | 官方基准 |

切换手法：新配置以探针容器启动并跑完全套验收后，直接 `docker rename` 成生产容器名，
**零停机、不重跑 14 分钟的 JIT 与图捕获**。

需要三个补丁（见 `patches/sglang-0811/`）：同步加载、**triton 路由（自研，上游至今缺失）**、dflash renorm 兜底（移植自海光 2026-08-13 提交）。

### ⚠️ 生产护栏一：只能跑贪心

| 场景 | 可用性 |
|---|---|
| 贪心（temperature=0），1/2/4/8/10 并发 | ✅ 全部稳定 |
| 非贪心（temperature>0）+ 启动预热，≤4 并发 | ✅ 可用（32.4 tok/s） |
| 非贪心，≥8 并发 | ❌ `HSA_STATUS_ERROR_EXCEPTION 0x1016` → watchdog 超时、服务挂死 |

**网关层必须拦截高并发采样流量**，或路由到无投机备用线（`scripts/start_0811_nospec.sh`，
8 并发 52.2 tok/s，任意温度稳定）。

**决定性对照实验**：同镜像、同权重、**不开 DSpark** 时，temperature=0.7 × 8 并发 **8/8 成功（52.2 tok/s）**
——问题被完全归因于 DSpark 的投机接受路径，与镜像、权重、gfx928 triton 注意力后端无关。
六项缓解措施（draft 块大小、稀疏 top_k、运行批量限流、accept 内核 torch 回退、JIT moe_align、降并发）全部无效；
唯一有效的是**预热非贪心路径**，可把安全并发从 2 提到 4。完整矩阵与证据见
`docs/Bug报告-DSpark与triton路由.md`，排查全过程见 `docs/调优记录-轮次24.md`。

### ⚠️ 生产护栏二：Think 改为按请求显式开启

0731 线上 `reasoning_content` 不再默认产出，需要请求携带 `{"chat_template_kwargs": {"thinking": true}}`。
不带该参数时答案本身正确，但 `reasoning_content` 为空。Tool Call 无此要求。
**依赖 reasoning_content 的下游客户端需同步改造**（`docs/调优记录-轮次27.md`）。

### ⚠️ 生产护栏三：`--kv-cache-dtype bfloat16` 静默产出乱码

本线的解码内核 `triton_fp8_attention_fwd` 按 **fp8 布局**读 KV。改成 bf16 后服务照常启动、
不报任何错误，**性能指标全线"变好"**（单流 33.8→44.1、accept rate 0.68→**1.00**），
但输出是彻底的乱码。`accept rate 恒为 1.00` 是这类失效的关键告警信号——drafter 与 target
读同一份坏 KV，一起算错到同一处，于是"全部命中"。

**fp8 KV 是本线必需项**，与 0728 线（4 月版 + MTP，fp8 KV 慢 14% 且破坏 Think）结论相反：
两条线走不同内核，结论不可互相套用。详见 `docs/调优记录-轮次25.md`。

### 为什么停在 33 而不是 50

**应用层参数调优已经穷尽**，天花板不在调度而在**权重访存带宽**。四组实验从不同方向指向同一结论：

| 实验 | 结果 | 说明 |
|---|---|---|
| DSpark 块大小 3 / 5 / 8 | 5（模型训练值）最优 | 块大小不是性能旋钮，8 时 accept rate 崩到 0.40 |
| 接受阈值放宽到 0.9 / 0.8 | accept len 4.38→**4.55**，速度 33.8→33.3 | **命中率提高换不来速度** |
| 为 K100-AI 现场调优 MoE triton 配置 | 解码 **+1%**（噪声内）、预填 **0%** | 镜像里配置目录本是空的，调优后确认生效仍无收益 |
| 重调 cuda graph 解码批量 | **假设被日志证伪** | 框架早已按 `num_tokens_per_req=6` 自行换算，`bs=[1..16]` 全覆盖、无回退 |

解码每步只有约 6 个 token（DSpark 验证窗口）× topk 8 = 48 行，摊到 256 个专家上平均每专家不到 1 行——
这一步的耗时几乎全部是"把 256 个专家的权重从 HBM 读一遍"，与怎么切 tile、猜中多少、图捕不捕获都无关。
50 tok/s 对应单步 ≈88 ms（现为 ≈130 ms），需要的是**减少每步权重读取量**。

**还有量级空间的只剩两条**，都需要改变模型或并行布局本身：**专家并行 EP**（`--ep-size`，
理论上每步 MoE 权重读取量降到约 1/5，但受 `NCCL_P2P_DISABLE=1` 与 gfx928 无 DeepEP 内核制约）
和 **4-bit MoE 权重**（`quark_int4fp8_moe` 等，直接把瓶颈砍半，需重跑量化管线 + 精度验收）。
详见 `docs/调优记录-轮次27.md` 第五节。

全过程（含权重管线、四关突破、测量陷阱）见 **`docs/0731升级与DSpark实战.md`**。

## 成果一览

| 路线 | 状态 | 单流解码 | 聚合吞吐 | 定位 |
|---|---|---|---|---|
| **sglang 线 · 0811 镜像 + DSpark** | ✅ **生产主线**（限贪心） | **33.2 tok/s** | 8 并发 8/8、10 并发 10/10 | 0731 模型 + DSpark，1M 上下文 |
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
| [HuggingFace · deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 权重来源（本仓库不含权重，需自行下载后按 `scripts/` 量化）；`generation_config.json` 的本地部署推荐参数出处 |

### 海光（HYGON-AI）

| 仓库 | 在本项目中的作用 |
|---|---|
| [HYGON-AI/sglang-das](https://github.com/HYGON-AI/sglang-das) | 海光 sglang 分支，生产主线的上游。**报障目标仓库**（分支 `v0.5.15.post1_dev`）；`dflash renorm` 兜底补丁移植自其 2026-08-13 提交 `18e167b7`；MoE 调优器取自其 `benchmark/kernels/fused_moe_triton/`；本仓库 `docs/Bug报告-*.md` 即针对该仓库 |
| [HYGON-AI/vllm-plugin-das](https://github.com/HYGON-AI/vllm-plugin-das) | 海光 vLLM 插件，回退线的上游；`patches/vllm/` 的 6 个 gfx928 正确性补丁针对它 |
| [HYGON-AI/inference-cookbook-das](https://github.com/HYGON-AI/inference-cookbook-das) | 海光官方推理示例与启动参数参考。注意其启动器面向 BW 卡（gfx936/938），**在 K100-AI 上照抄必崩**，需按 `docs/0731升级与DSpark实战.md` 附一替换整套 env |

### FlagOS（智源 / flagos-ai）

| 仓库 | 在本项目中的作用 |
|---|---|
| [flagos-ai/DeepSeek-V4-FlagOS](https://github.com/flagos-ai/DeepSeek-V4-FlagOS) | 精度参考线的模型侧支持；**权重管线的反量化脚本 `convert_weight.py` 来自这里**（FP4/FP8 → BF16，我方以 `--device cpu` 运行，见 `docs/0731升级与DSpark实战.md` 第 2.2 节） |
| [flagos-ai/vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) | FlagOS 线所用的 vLLM 插件（我方在其上打了 19 个补丁才跑通，性能不足但可作数值对齐基准） |
| [flagos-ai/community](https://github.com/flagos-ai/community) | FlagOS 社区仓库：路线图、支持矩阵与问题反馈入口 |
| [flagos-ai/EasyOfUse](https://github.com/flagos-ai/EasyOfUse) | FlagOS 的易用性工具与部署示例集合 |

### 上游框架

| 仓库 | 在本项目中的作用 |
|---|---|
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 生产主线框架上游。本仓库 `patches/sglang-0811/patch_triton_backend.py`（triton 路由）针对的是 `debug_flash_mla_adapter.py`，该缺口在上游与海光分支中至今均未修复 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 回退线框架上游 |

## 仓库结构

```
docs/
  在8张K100-AI上运行DeepSeek-V4-Flash-从零到成果.md   # 主文档：三条路线从零到结果，含全部失败尝试与原因
  0731升级与DSpark实战.md                              # 0731 升级 + DSpark 攻坚全过程（含测量陷阱）
  Bug报告-提交海光.md                                  # A/B/C/D/E/F 六类 20+ 项问题，附根因与复现方式
  Bug报告-DSpark与triton路由.md                        # 三项新问题：triton 路由缺失、DSpark 并发硬件异常、bf16 KV 静默乱码
  启动手册-sglang线.md                                 # 生产主线启动手册（含 lpm 调度测试方法论）
  启动手册-vLLM线.md                                   # 回退线启动手册
  调优记录-全程.md                                     # 轮次 1–23 调优全记录（含所有否决项及证据）
  调优记录-轮次24.md                                   # 续篇：DSpark 并发崩溃深挖（对照实验方法论）
  调优记录-轮次25.md                                   # 续篇：参数再优化三项全否 + bf16 KV 乱码陷阱
  调优记录-轮次26.md                                   # 续篇：MoE 内核调优否决，瓶颈定位为权重访存带宽
  调优记录-轮次27.md                                   # 续篇：cuda graph 假设被证伪 + 生产切换到 DSpark
patches/
  sglang/         # 0728 镜像：同步加载补丁 + 参数化启动器 ×5 + Dockerfile
  sglang-0811/    # 0811 镜像：triton 路由补丁 + dflash renorm 补丁 + 诊断补丁 + gfx928 启动器
  vllm/           # 6 个 gfx928 正确性补丁 diff + Dockerfile
tests/
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
  start_prod_dspark.sh       # 【生产】0731 + 0811 镜像 + DSpark（含贪心护栏说明）
  start_sglang_dsv4_prod.sh  # 旧生产（0728 + 4 月版 + MTP），现为回退线
  start_0731_base.sh         # 0731 + 0728 镜像（无投机基线）
  start_0811_nospec.sh       # 0731 + 0811 镜像 + 无投机（采样场景备用 / 对照实验）
  quant_w8a8_0731.py         # 0731 量化（含 DSpark 张量取舍开关）
  run_0731_pipeline.sh       # 反量化 → 量化 全流程
  run_quant_stage3.sh        # 单独重跑量化阶段
  moe_tuning/                # K100-AI MoE 内核调优：调优器补丁、裁剪搜索空间、产出配置
```

## 关键技术点（详见主文档与 Bug 报告）

1. **多线程 H2D 拷贝死锁**（加载假死 8 小时）：py-spy 定位到 sglang 异步加载 + DTK HIP 运行时的线程安全问题，`should_async_load → False` 补丁修复，加载缩至 ~3 分钟。
2. **vLLM 线 6 处 gfx928 正确性 bug**：融合算子写坏 KV 缓存、indexer 写读格式矛盾、转置 view 静默乱码等，逐一根因定位并给出补丁。
3. **KV cache dtype 与内核强绑定**：0728 线（MTP）必须 bf16，量化 KV 会导致复读死循环且慢 14%；0811 线（dsv4 triton 解码）反过来必须 fp8，bf16 会静默乱码。**结论不可跨线套用**。
4. **投机解码三代对比**：EAGLE MTP（0728 线，1.55×）→ DSpark（0811 线，2.75×，accept len 4.38）；vLLM 线 MTP ≥8 并发 VMFault（已报障）。
5. **把"不支持"的镜像救活**：0811 镜像的 gfx928 障碍全在 env 选错路径与一处未接线的 triton 分支，逐关突破后可用（`docs/0731升级与DSpark实战.md` 第 4 节）。
6. **测量陷阱**：长上下文 prefill 曾先后测得 232 / 2144 / 437 tok/s ——分别是 triton 首次 JIT 污染、前缀缓存命中、真实值。单流解码同样中招：首次 12.26、第二次 33.21，而 12.26 恰好接近"无投机"的 12.3，极易被误判为投机没生效。**压测前必须热身两次**。
7. **性能变好要先怀疑正确性**：bf16 KV 让单流从 33.8 涨到 44.1（峰值 50.7，正好"达标"）、accept rate 冲到 1.00 —— 实际输出全是乱码。任何性能改动都必须过一遍数学 / Think / Tool Call 验收再采信。
8. **用对照实验定位崩溃**：面对 GPU 硬件异常这种"看不到栈"的故障，最有效的手段不是继续试参数，而是**固定其它变量、只切换嫌疑组件**。第 24 轮用一次无投机对照就排除了镜像/权重/注意力后端三个方向（`docs/调优记录-轮次24.md`）。
9. **醒目的可疑点未必是真问题，先取证再调优**：轮次 26 的"MoE 配置目录为空 + 满屏 sub-optimal 警告"实测只值 +1%；轮次 27 的"cuda-graph-max-bs 显然没覆盖验证批"被一行启动日志直接证伪（框架早已按 `num_tokens_per_req=6` 自行换算）。两次取证成本都在十几分钟，都省下了半天的无效调优。
10. **零停机切换**：新配置以探针容器启动、跑完全套验收后 `docker rename` 成生产名——验证过的进程就是生产进程，不必为了"正式启动"再花 14 分钟重跑 JIT 与图捕获。

## 说明

- **模型权重不在本仓库**（4 月版 279GB / 0731 版 273GB+292GB，需自行量化；`config.json` 需加入 sglang 兼容的 ignore 规则，方法见主文档与 `scripts/quant_w8a8_0731.py`）。
- 文档中 `<NODE_A_IP>` / `<NODE_B_IP>` / `nodeA` / `nodeB` / `<internal-harbor>` / `/home/user` 为脱敏占位符。
- 基础镜像来自海光 sourcefind 仓库（`harbor.sourcefind.cn:5443`），补丁以 Dockerfile 固化为新镜像，不含默认入口，参数全部经环境变量传入。

*文档基线日期：2026-08-14。*

---

Copyright © 2026 DaoTech Team. Licensed under the MIT License.
