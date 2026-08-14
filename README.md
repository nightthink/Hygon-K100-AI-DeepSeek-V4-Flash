# 在 8 张海光 K100-AI 上运行 DeepSeek-V4-Flash：从零到生产

在 8×海光 K100-AI 64GB（gfx928，DTK 26.04）上，将 **DeepSeek-V4-Flash**（284B MoE，激活 13B，自行量化 W8A8）部署为 OpenAI 兼容服务的完整实录：三条路线全部跑通，含全部补丁、启动器、测试脚本、24 轮调优记录，以及提交给海光的详细 Bug 报告。

## 最新进展：0731 版本 + DSpark 投机解码（2026-08-14）

把 7 月 31 日发布的 **DeepSeek-V4-Flash-0731** 正式版部署上机，并把官方判定"不支持 K100-AI"的 0811 镜像救活，拿到了新一代 **DSpark 投机解码**：

| 指标 | 原生产（4 月版 + MTP） | **0731 + DSpark** | 变化 |
|---|---|---|---|
| 单流解码（编程任务） | 18.7 tok/s（编程题 24） | **33.8 tok/s** | **+80%** |
| DSpark accept len / rate | — | **4.38 / 0.68** | 一次猜 5 个中 4 个多 |
| 23K prompt prefill | 343 tok/s | **437 tok/s** | **+27%** |
| KV 池容量 | 600,832 | **1,026,816** | **+71%** |
| 8 / 10 并发聚合（贪心） | 59 / — | 50.2 / 46.3 | −15% |
| 模型能力（DeepSWE） | 7.3 | **54.4** | 官方基准 |

需要三个补丁（见 `patches/sglang-0811/`）：同步加载、**triton 路由（自研，上游至今缺失）**、dflash renorm 兜底（移植自海光 2026-08-13 提交）。

### ⚠️ 已知限制：非贪心采样 + 并发触发 GPU 硬件异常

| 场景 | 可用性 |
|---|---|
| 贪心（temperature=0），1/2/4/8/10 并发 | ✅ 全部稳定 |
| 非贪心（temperature>0）+ 启动预热，≤4 并发 | ✅ 可用（32.4 tok/s） |
| 非贪心，≥8 并发 | ❌ `HSA_STATUS_ERROR_EXCEPTION 0x1016` → watchdog 超时、服务挂死 |

**决定性对照实验**：同镜像、同权重、**不开 DSpark** 时，temperature=0.7 × 8 并发 **8/8 成功（52.2 tok/s）**
——问题被完全归因于 DSpark 的投机接受路径，与镜像、权重、gfx928 triton 注意力后端无关。
六项缓解措施（draft 块大小、稀疏 top_k、运行批量限流、accept 内核 torch 回退、JIT moe_align、降并发）全部无效；
唯一有效的是**预热非贪心路径**，可把安全并发从 2 提到 4。完整矩阵与证据见
`docs/Bug报告-DSpark与triton路由.md`。

全过程（含权重管线、四关突破、测量陷阱）见 **`docs/0731升级与DSpark实战.md`**。

## 成果一览

| 路线 | 状态 | 单流解码 | 聚合吞吐 | 定位 |
|---|---|---|---|---|
| **sglang 线 · 0811 镜像 + DSpark** | ✅ 最快（贪心限定） | **33.8 tok/s** | 8 并发 50.2、10 并发 46.3 | 0731 模型 + DSpark，1M 上下文 |
| **sglang 线 · 0728 镜像 + MTP** | ✅ 生产主线 | 18.7–18.8 tok/s（编程类 ~24） | 8 并发 ~59、10 并发 ~66 | Think + Tool Call + 前缀缓存 + 1M 上下文 |
| **sglang 线 · 0811 镜像 + 无投机** | ✅ 采样场景备用 | ~12.3 tok/s | 8 并发 52.2（任意温度稳定） | 需要 temperature>0 且高并发时使用 |
| **vLLM 线**（hy3-0706 镜像 + 6 补丁） | ✅ 回退线 | ~9.3 tok/s | 128 并发 ~116 tok/s | 高并发吞吐强项 |
| **FlagOS 线**（vllm-plugin-fl + 19 补丁） | ✅ 精度参考 | 性能不足 | — | 数值对齐基准 |

生产定稿配置（0728 主线）：bf16 KV + MTP（EAGLE steps=3）+ 双解析器（reasoning/tool-call）+ radix 前缀缓存，`mem-fraction-static 0.85` / `chunked-prefill 4096` / `CUDA_GRAPH_MAX_BS 16`。长上下文实测：23K prompt prefill ~343 tok/s（TTFT ~45s），98K ~395 tok/s（~4.1min），前缀缓存命中 TTFT 0.8s。

## 仓库结构

```
docs/
  在8张K100-AI上运行DeepSeek-V4-Flash-从零到成果.md   # 主文档：三条路线从零到结果，含全部失败尝试与原因
  0731升级与DSpark实战.md                              # 0731 升级 + DSpark 攻坚全过程（含测量陷阱）
  Bug报告-提交海光.md                                  # A/B/C/D/E/F 六类 20+ 项问题，附根因与复现方式
  Bug报告-DSpark与triton路由.md                        # 新增两项：triton 路由缺失、DSpark 并发 GPU 硬件异常
  启动手册-sglang线.md                                 # 生产主线启动手册（含 lpm 调度测试方法论）
  启动手册-vLLM线.md                                   # 回退线启动手册
  调优记录-全程.md                                     # 参数调优全记录（含所有否决项及证据）
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
  start_sglang_dsv4_prod.sh  # 生产一键启动（0728 + 4 月版定稿配置）
  start_0731_base.sh         # 0731 + 0728 镜像（无投机基线）
  start_0731_dspark.sh       # 0731 + 0811 镜像 + DSpark
  start_0811_nospec.sh       # 0731 + 0811 镜像 + 无投机（对照实验 / 采样场景备用）
  quant_w8a8_0731.py         # 0731 量化（含 DSpark 张量取舍开关）
  run_0731_pipeline.sh       # 反量化 → 量化 全流程
  run_quant_stage3.sh        # 单独重跑量化阶段
```

## 关键技术点（详见主文档与 Bug 报告）

1. **多线程 H2D 拷贝死锁**（加载假死 8 小时）：py-spy 定位到 sglang 异步加载 + DTK HIP 运行时的线程安全问题，`should_async_load → False` 补丁修复，加载缩至 ~3 分钟。
2. **vLLM 线 6 处 gfx928 正确性 bug**：融合算子写坏 KV 缓存、indexer 写读格式矛盾、转置 view 静默乱码等，逐一根因定位并给出补丁。
3. **量化 KV cache（int8 与 fp8）均破坏 Think**：复读死循环 + 解码反而变慢 14%，生产必须 bf16 KV——gfx928 无 fp8 硬件指令是根源之一。
4. **投机解码三代对比**：EAGLE MTP（0728 线，1.55×）→ DSpark（0811 线，2.75×，accept len 4.38）；vLLM 线 MTP ≥8 并发 VMFault（已报障）。
5. **把"不支持"的镜像救活**：0811 镜像的 gfx928 障碍全在 env 选错路径与一处未接线的 triton 分支，逐关突破后可用（`docs/0731升级与DSpark实战.md` 第 4 节）。
6. **测量陷阱**：长上下文 prefill 曾先后测得 232 / 2144 / 437 tok/s ——分别是 triton 首次 JIT 污染、前缀缓存命中、真实值。测法见 `tests/longctx_fresh.py`。
7. **用对照实验定位崩溃**：面对 GPU 硬件异常这种"看不到栈"的故障，最有效的手段不是继续试参数，而是**固定其它变量、只切换嫌疑组件**。本仓库第 24 轮用一次无投机对照就排除了镜像/权重/注意力后端三个方向（`docs/Bug报告-DSpark与triton路由.md`）。

## 说明

- **模型权重不在本仓库**（4 月版 279GB / 0731 版 273GB+292GB，需自行量化；`config.json` 需加入 sglang 兼容的 ignore 规则，方法见主文档与 `scripts/quant_w8a8_0731.py`）。
- 文档中 `<NODE_A_IP>` / `<NODE_B_IP>` / `nodeA` / `nodeB` / `<internal-harbor>` / `/home/user` 为脱敏占位符。
- 基础镜像来自海光 sourcefind 仓库（`harbor.sourcefind.cn:5443`），补丁以 Dockerfile 固化为新镜像，不含默认入口，参数全部经环境变量传入。

*文档基线日期：2026-08-14。*

---

Copyright © 2026 DaoTech Team. Licensed under the MIT License.
