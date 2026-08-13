# 在 8 张海光 K100-AI 上运行 DeepSeek-V4-Flash：从零到生产

在 8×海光 K100-AI 64GB（gfx928，DTK 26.04）上，将 **DeepSeek-V4-Flash**（284B MoE，激活 13B，自行量化 W8A8）部署为 OpenAI 兼容服务的完整实录：三条路线全部跑通，含全部补丁、启动器、测试脚本、20 轮调优记录，以及提交给海光的详细 Bug 报告。

## 成果一览

| 路线 | 状态 | 单流解码 | 聚合吞吐 | 定位 |
|---|---|---|---|---|
| **sglang 线**（0728 镜像 + 补丁） | ✅ 生产主线 | **18.7–18.8 tok/s**（MTP，编程类输出实测可达 ~24） | 8 并发 ~59、10 并发 ~66 tok/s | Think + Tool Call + 前缀缓存 + 1M 上下文 |
| **vLLM 线**（hy3-0706 镜像 + 6 补丁） | ✅ 回退线 | ~9.3 tok/s | 128 并发 ~116 tok/s | 高并发吞吐强项 |
| **FlagOS 线**（vllm-plugin-fl + 19 补丁） | ✅ 精度参考 | 性能不足 | — | 数值对齐基准 |

生产定稿配置（sglang 线）：bf16 KV + MTP（EAGLE steps=3）+ 双解析器（reasoning/tool-call）+ radix 前缀缓存，`mem-fraction-static 0.85` / `chunked-prefill 4096` / `CUDA_GRAPH_MAX_BS 16`。长上下文实测：23K prompt prefill ~343 tok/s（TTFT ~45s），98K ~395 tok/s（~4.1min），前缀缓存命中 TTFT 0.8s。

## 仓库结构

```
docs/
  在8张K100-AI上运行DeepSeek-V4-Flash-从零到成果.md   # 主文档：三条路线从零到结果，含全部失败尝试与原因
  Bug报告-提交海光.md                                  # A/B/C/D/E/F 六类 20+ 项问题，附根因与复现方式
  启动手册-sglang线.md                                 # 生产主线启动手册（含 lpm 调度测试方法论）
  启动手册-vLLM线.md                                   # 回退线启动手册
  调优记录-20轮全程.md                                 # 20 轮参数调优全记录（含所有否决项及证据）
patches/
  sglang/    # 同步加载补丁（修复多线程 H2D 死锁）+ 参数化启动器 ×5 + Dockerfile
  vllm/      # 6 个 gfx928 正确性补丁 diff + Dockerfile
tests/
  test_dv4.sh              # 7 项验收（health/对话/数学/素数/流式/速度）
  bench_concurrency.py     # 并发压测
  ttft_test.py             # TTFT 与前缀缓存收益
  longctx_test.py          # 长上下文测试（23K/98K 级）
scripts/
  start_sglang_dsv4_prod.sh  # 生产一键启动脚本（定稿配置）
```

## 关键技术点（详见主文档与 Bug 报告）

1. **多线程 H2D 拷贝死锁**（加载假死 8 小时）：py-spy 定位到 sglang 异步加载 + DTK HIP 运行时的线程安全问题，`should_async_load → False` 补丁修复，加载缩至 ~3 分钟。
2. **vLLM 线 6 处 gfx928 正确性 bug**：融合算子写坏 KV 缓存、indexer 写读格式矛盾、转置 view 静默乱码等，逐一根因定位并给出补丁。
3. **量化 KV cache（int8 与 fp8）均破坏 Think**：复读死循环 + 解码反而变慢 14%，生产必须 bf16 KV——gfx928 无 fp8 硬件指令是根源之一。
4. **MTP/EAGLE 投机解码**：sglang 线稳定 1.55×（steps=3 最优）；vLLM 线 ≥8 并发 VMFault（已报障）。
5. **单流天花板分析**：应用层已穷尽（20 轮），到 50 tok/s 需海光内核层 2.7× 提升（NSA decode 内核、custom allreduce、树形投机），量化诉求见 Bug 报告 F-1。

## 说明

- **模型权重不在本仓库**（279GB W8A8，需自行量化；`config.json` 需加入 sglang 兼容的 ignore 规则，方法见主文档）。
- 文档中 `<NODE_A_IP>` / `<NODE_B_IP>` / `nodeA` / `nodeB` / `<internal-harbor>` 为脱敏占位符。
- 基础镜像来自海光 sourcefind 仓库（`harbor.sourcefind.cn:5443`），补丁以 Dockerfile 固化为新镜像，不含默认入口，参数全部经环境变量传入。

*文档基线日期：2026-08-13。*

---

Copyright © 2026 DaoTech Team. All rights reserved.
