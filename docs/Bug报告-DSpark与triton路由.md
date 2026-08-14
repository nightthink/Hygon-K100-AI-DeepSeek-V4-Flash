# 提交材料：K100-AI（gfx928）上 DeepSeek-V4-Flash + DSpark 的两个问题

> 目标仓库：`HYGON-AI/sglang-das`（分支 `v0.5.15.post1_dev`）
> 报告方环境：8×海光 K100-AI（gfx928）/ DTK 26.04 / DeepSeek-V4-Flash-0731 W8A8
> 使用镜像：`custom:sglang0.5.12-...-20260804-0006-deepseekV4-0811`（内含 sglang 0.5.15.post2.dev564+gb97d7df6e）
> 日期：2026-08-14（问题二于 8-14 补充对照实验与预热发现）

---

## 摘要

我方在 K100-AI 上把 DeepSeek-V4-Flash-0731 + DSpark 跑通，**单流解码从 12.3 tok/s 提升到 33.8 tok/s（2.75×）**，
比我方原生产配置（4 月版 + EAGLE MTP，18.7 tok/s）快 **80%**。过程中需要自行补一个补丁（问题一），
并发现一个尚无对应修复的 GPU 硬件异常（问题二）。

实测数据（8×K100-AI，W8A8，TP=8，`mem-fraction-static 0.85`，`chunked-prefill 4096`，`cuda-graph-max-bs 16`）：

| 指标 | 4 月版 + MTP | 0731 无投机 | **0731 + DSpark** |
|---|---|---|---|
| 单流解码（编程任务 600 tok） | 18.7 | 12.3 | **33.8** |
| DSpark accept len / rate | — | — | **4.38 / 0.68** |
| 8 并发聚合（贪心） | 59 | 55.9 | 50.2 |
| 10 并发聚合（贪心） | — | — | 46.3 |
| 23K prompt prefill | 343 tok/s | — | **437 tok/s** |
| 98K prompt prefill | 395 tok/s | — | 369 tok/s |
| KV 池 | 600,832 | 894,720 | **1,026,816** |

---

## 问题一【功能缺失】HCU 平台的 dsv4 注意力后端未接通 triton 解码路径

### 现象

在 gfx928 上，`dsv4` 注意力后端的解码路径没有可用的高性能实现：

| `SGLANG_HACK_FLASHMLA_BACKEND` | 结果 |
|---|---|
| `kernel`（默认） | 运行时拒绝：`Dense decode MLA is only supported on gfx936 or gfx938 architecture` |
| `triton` | `AssertionError: unsupported backend 'triton'` —— 分支未接入 |
| `torch` | 可运行但仅 **1.26 tok/s**，不具备可用性 |

### 根因

`DeepseekV4AttnBackend._call_flash_mla_with_kvcache()` 从
`sglang/srt/layers/attention/debug_flash_mla_adapter.py` 导入入口，而该入口只识别两种后端：

```python
def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if backend in {"torch", "native", "torch_native"}:
        return torch_native_flash_mla_with_kvcache(**kwargs)

    assert backend == "kernel", f"unsupported backend {backend!r}"
    import flash_mla
    return flash_mla.flash_mla_with_kvcache(**kwargs)
```

而同目录 `hip_flash_mla.py` 已实现完整的 triton 分支：

```python
    if backend == "triton":
        from sglang.kernels.ops.attention.nsa_triton_decode import (
            triton_fp8_attention_fwd,
        )
        return triton_fp8_attention_fwd(**kwargs)
```

平台分流在 `attention_registry.py::create_dsv4_backend`：`_is_hip and not _is_hcu` 才走
`DeepseekV4HipRadixBackend`（可用 triton），**HCU 平台落到只支持 kernel/torch 的
`DeepseekV4AttnBackend`**。gfx928 因此两头落空：kernel 内核明确拒绝该架构，triton 内核
就在镜像里却接不上。

`triton_fp8_attention_fwd` 的文档字符串本身声明它是 drop-in 设计：

> Accepts the same `**kwargs` dict that the caller builds for `flash_mla_with_kvcache` /
> `dpsk_v4_fp8_attention_fwd`... Unused keys (`block_table`, `cache_seqlens`,
> `tile_scheduler_metadata`, `num_splits`, `causal`, `is_fp8_kvcache`) are silently ignored.

调用方 `_build_flash_mla_input_dict()` 构造的键（`q` / `k_cache` / `head_dim_v` /
`softmax_scale` / `indices` / `topk_length` / `attn_sink` / `extra_*`）与该签名完全匹配。

### 建议补丁

```diff
 def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
     if backend in {"torch", "native", "torch_native"}:
         return torch_native_flash_mla_with_kvcache(**kwargs)
 
+    # gfx928（K100-AI）上 flash_mla 内核拒绝 dense decode MLA，
+    # 改走仓库自带的 triton 稀疏解码内核（同一 kwargs 约定，多余键忽略）。
+    if backend == "triton":
+        from sglang.kernels.ops.attention.nsa_triton_decode import (
+            triton_fp8_attention_fwd,
+        )
+
+        return triton_fp8_attention_fwd(**kwargs)
+
     assert backend == "kernel", f"unsupported backend {backend!r}"
     import flash_mla
 
     return flash_mla.flash_mla_with_kvcache(**kwargs)
```

等价方案：让 `DeepseekV4AttnBackend` 直接 import `hip_flash_mla` 的入口（其分发已覆盖
torch/tilelang/triton/kernel 四种）。我方选择前者是因改动面更小、不影响其它平台既有行为。

### 效果

打补丁后 gfx928 首次获得可用的 dsv4 解码路径：服务正常启动、Think/Tool Call/数学题全部正确、
单流从 1.26 tok/s（torch）提升到 33.8 tok/s（triton + DSpark）。

---

## 问题二【P0，新发现】DSpark 非贪心采样在 gfx928 上并发触发 GPU 硬件异常

### 现象

`--speculative-algorithm DSPARK` 下，**temperature > 0** 的请求并发到一定数量后：

```
Callback: Queue 0x... aborting with error : HSA_STATUS_ERROR_EXCEPTION:
An HSAIL operation resulted in a hardware exception. code: 0x1016
```

随后 `Scheduler watchdog timeout (self.watchdog_timeout=300)`，服务不可用，必须重启容器。

### 隔离矩阵（每格均为独立实验，崩溃后重启服务再测）

| 并发 | temperature=0（贪心） | temperature=0.7 冷启动 | temperature=0.7 **预热后** |
|---|---|---|---|
| 1 | ✅ 33.8 tok/s | ✅ | ✅ |
| 2 | ✅ | ✅ 13.4 tok/s | ✅ |
| 4 | ✅ 32.5 tok/s | ❌ GPU 硬件异常 | **✅ 32.4 tok/s** |
| 8 | ✅ 50.2 tok/s（8/8） | ❌ GPU 硬件异常（0–1/8） | ⚠️ 软挂起（无 HSA 异常，但请求不返回） |
| 10 | ✅ 46.3 tok/s（10/10） | 未测（已知会崩） | 未测 |

"预热"= 服务起来后先发若干条 **单路 temperature>0** 请求，把非贪心接受路径的 triton JIT
编译与首次分配走完，再上并发。

### 关键对照实验：确认是 DSpark 专属，而非镜像/权重/平台问题

| 组合 | temp 0.7 × 8 并发 | 结论 |
|---|---|---|
| 0728 镜像 + 4 月版权重 + EAGLE MTP | ✅ 正常 | 老投机路径无此问题 |
| **0811 镜像 + 0731 权重 + 无投机** | **✅ 8/8 成功，52.2 tok/s 聚合** | **同镜像同权重同 triton 后端下非贪心并发完全稳定** |
| 0811 镜像 + 0731 权重 + **DSpark** | ❌ HSA 硬件异常 | 唯一变量是 DSpark |

中间一行是本报告最重要的一条证据：它排除了 0811 镜像、0731 权重、gfx928 triton 注意力后端、
以及采样器本身（`top_k_renorm_prob` 的 torch 兜底）作为诱因，把问题完全钉在 DSpark 的
投机接受路径上。

### 已排除的因素（六项独立缓解措施，全部无效）

| # | 尝试 | 结果 |
|---|---|---|
| 1 | `--speculative-dspark-block-size 3`（验证窗口 6→4） | 同样崩溃，与 draft 块大小无关 |
| 2 | 显式 `top_k=50` 走稀疏路径 | 成功率 0/8 → 4/8，仍挂起；稠密全词表分支不是唯一诱因 |
| 3 | `--max-running-requests 4`（日志确认批量确被限制） | 依然崩溃，不是"运行批量过大" |
| 4 | `SGLANG_DSPARK_FORCE_TORCH_ACCEPT=1`（令 `kernels/dispatch.py::inputs_on_cuda` 恒返 False，强制走 torch 参考实现） | 依然崩溃，说明诱因不限于 `accept_sampling_triton` 单个内核 |
| 5 | 关闭 JIT `moe_align_block_size` | 无影响 |
| 6 | 降并发到 4（冷启动） | 崩溃（但预热后可用，见下） |

### 唯一有效的缓解：预热非贪心路径

先用单路 temperature>0 请求把该路径跑热，安全并发从 **2 提升到 4**（4 并发 32.4 tok/s，
冷启动时同配置必崩）；8 并发仍不可用，但失效形态从 **HSA 硬件异常 + watchdog 超时**
降级为**软挂起**（进程存活、无内核异常、请求不返回）。

这提示崩溃与该路径的**首次并发触发**（triton JIT 编译 / 首次显存分配 / 首次 kernel launch
与并发调度叠加）强相关，而非稳态计算本身的越界访存。建议贵方从这个方向排查：DSpark
非贪心接受路径的 JIT/首次调用是否在多请求并发进入时存在竞态。

### 一个需要澄清的干扰项（非诱因）

`moe_align_block_size_kernel` 的
`Launch params (1024, 1, 1) are larger than launch bounds (256)` 警告在**稳定的贪心运行中
同样出现**，因此它不是本问题的诱因，仅作为独立的次要建议列在附二。

### 相关但不同的一个问题（贵方已修，我方已移植验证）

在打补丁之前，同一路径先报的是：

```
File ".../speculative/dflash_utils.py", line 784, in build_dflash_verify_target_probs
    target_probs = top_k_renorm_prob(...)
TypeError: 'NoneType' object is not callable
```

我方移植了贵方 2026-08-13 的提交 `18e167b7`（"fix the client inference None errors in
temperature>0 with a torch implementation"）后该报错消失，**但随即暴露出上面的硬件异常**。
建议在合入该修复时一并回归 gfx928 上的多并发非贪心场景。

### 我方当前可用边界与规避

| 场景 | 可用性 |
|---|---|
| 贪心（temperature=0），1/2/4/8/10 并发 | ✅ 全部稳定，可直接生产 |
| 非贪心 + 预热，≤4 并发 | ✅ 可用（32.4 tok/s） |
| 非贪心，≥8 并发 | ❌ 不可用 |

生产侧限制为贪心解码。编程助手场景多数使用 temperature=0，因此 DSpark 仍可投入使用；
但任意一批 temperature>0 的并发请求就可能拖垮服务，需要网关层强制。

> 补充说明：DeepSeek-V4-Flash-0731 的 `generation_config.json` 为 `do_sample=true` /
> `temperature=1.0`，官方模型卡对本地部署推荐 `temperature=1.0`（官方 API 在思考模式下
> 忽略 temperature 参数，这是 API 侧行为，不等于本地权重不支持采样）。也就是说，
> **推荐配置恰好落在本问题的故障区间内**，这使该问题的优先级高于"少数用户偶尔调高温度"。

---

## 附一：gfx928 上跑通 DSpark 所需的完整配置

官方启动器面向 BW 卡，在 K100-AI 上照抄必崩。以下为我方验证可用的组合：

```bash
# 1. custom allreduce（不关会在权重加载后挂死；0.5.15 换了开关名，旧的 USE_DCU_CUSTOM_ALLREDUCE 已失效）
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
#    并在命令行加 --disable-custom-all-reduce

# 2. MHC 两条加速路径全关，落到 hc_pre_torch_impl
#    （否则 tilelang 报 HCU arch gfx928 not supported for MLS/GEMM_MLS）
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_USE_AITER_MHC_PRE=0
export SGLANG_OPT_USE_AITER_MHC_POST=0
export SGLANG_DSV4_MHC_PREWARM=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0

# 3. 本报告问题一的补丁启用的解码后端
export SGLANG_HACK_FLASHMLA_BACKEND=triton

# 4. 其余沿用官方 0728 k100ai 启动器的 env（GPU_MAX_HW_QUEUES=3、HIP_KERNEL_BATCH_CEILING=100、
#    SGLANG_USE_LIGHTOP=1、SGLANG_ROCM_USE_AITER_MOE=false 等约 20 项）
```

另需一个与本报告无关但必须的补丁：`model_loader/utils.py::should_async_load` 恒返 False
（DTK 26.04 上多线程 H2D 拷贝死锁，已在此前的 Bug 报告 A-1 中提交）。

## 附二：两个次要建议

1. **缺 K100-AI 的 MoE 内核配置**：日志提示
   `Config file not found at .../E=256,N=256,device_name=K100_AI,dtype=int8_w8a8,per_channel_quant=True.json`
   并回退默认配置（"Performance might be sub-optimal"）。建议随镜像附带 K100-AI 调优配置。
2. **`moe_align_block_size` launch bounds 警告**：
   `Launch params (1024, 1, 1) are larger than launch bounds (256)`，建议补 `__launch_bounds__`
   或用 `--gpu-max-threads-per-block` 重编。（如上文所述，此项与问题二无关。）

## 附三：验证支持

我方有 8×K100-AI 环境与完整的验收/压测/长上下文脚本（含 23K/98K 实测基线），
修复版镜像可在 1 天内完成回归对比。
