# DeepSeek-V4-Flash-0731 升级与 DSpark 投机解码实战（8×K100-AI）

> 时间：2026-08-13 ~ 08-14
> 环境：8×海光 K100-AI（gfx928）/ DTK 26.04 / 512GB 内存 / 96 核
> 结论先行：**单流解码从 18.7 tok/s 提升到 33.8 tok/s（+80%）**，长 prompt 预填充 +27%，
> KV 容量 +71%；代价是发现一个尚无修复的 GPU 硬件异常（非贪心 + 高并发）。

---

## 0. 一句话背景

我们此前在这台机器上把 4 月版 DeepSeek-V4-Flash 部署到生产（sglang 线，单流 18.7 tok/s，
详见《在8张K100-AI上运行DeepSeek-V4-Flash：从零到成果》）。7 月 31 日 DeepSeek 发布
0731 正式版，架构不变但重新做了后训练，编程与 agentic 能力大幅提升：

| 基准 | 4 月预览版 | 0731 正式版 |
|---|---|---|
| NL2Repo（仓库级代码任务） | 39.4 | 54.2 |
| DeepSWE（真实工程问题） | 7.3 | **54.4** |
| Terminal Bench（终端/Agent） | 61.8 | 82.7 |
| Toolathlon-Verified（工具调用） | 49.7 | 70.3 |

本文记录把 0731 落到这台机器上的完整过程，以及一个意外收获：**把官方判定"不支持
K100-AI"的 0811 镜像救活，拿到了 DSpark 投机解码**。

---

## 1. 第一个发现：0731 的投机模块换代了

对比新旧权重清单（`model.safetensors.index.json`）后发现：

- **主干完全一致**：43 层、256 专家、注意力结构、config 除新增字段外无差异。
- **MTP 模块整个换掉**：
  - 4 月版：`mtp.enorm / hnorm / e_proj / h_proj`（EAGLE 式 MTP，我们线上跑出 1.55× 加速的那个）
  - 0731：`mtp.main_norm / main_proj / markov_head / confidence_head`（**DSpark**，半自回归块级投机）
- DSpark 有 **3 个 draft 块**（`dspark_target_layer_ids: [40,41,42]`），每块带 256 专家，
  共 4705 个张量、2304 个专家权重——这就是权重从 149GB 涨到 167GB 的原因。
- config 新增 `dspark_block_size: 5`、`dspark_noise_token_id`、`dspark_target_layer_ids`、
  `dspark_markov_rank: 256`。

**后果**：我们生产用的 sglang 0728 镜像（sglang 0.5.12）只实现了旧式 MTP，全局搜不到任何
dspark 相关代码。0731 在其上**只能无投机运行**，单流从 18.7 掉到 12.3。

一个隐蔽的坑：镜像内启动器只要名字带 `mtp`（如 `run_ds_mtp_triton_logic_bf16_kv.sh`）就会
**强制开启 EAGLE**，而新权重里 mtp 张量已被剔除，它会去找不存在的 draft 权重。必须改用
`run_ds_nomtp_triton_logic_bf16_kv.sh`。

---

## 2. 权重管线：下载 → 反量化 → 双份量化

### 2.1 下载（167GB，约 4 小时）

经 hf-mirror 下载 `deepseek-ai/DeepSeek-V4-Flash-0731`，48 分片。两个坑：

1. **Xet 通道抛 401**：`huggingface_hub` 1.27 默认走 Xet CAS，hf-mirror 的代理返回
   `401 Unauthorized`，进程直接死亡。解法：`HF_HUB_DISABLE_XET=1` 回退普通 HTTP。
2. **并发过高触发 429**：16 并发被限流。降到 6 并发后稳定在 10-11 MB/s（该机出口带宽上限，
   实测 ModelScope 只有 3.3 MB/s，其它镜像站被网络策略挡掉）。

脚本加了 100 次重试循环，中途断了会自动续传。

### 2.2 反量化 FP4/FP8 → BF16（CPU，45 分钟）

用 FlagOS 的 `convert_weight.py --device cpu`（GPU 被生产服务占用）。

**先做了预检**，确认 DSpark 的新张量不会被漏掉或搞错：

| 分流 | 数量 | 说明 |
|---|---|---|
| MXFP4（expert）分支 | 2304 | DSpark 的 3 个 draft 块 × 256 专家 × 3 |
| blockwise FP8 分支 | 25 | attn / shared_experts |
| 直通（无 scale） | 47 | norm / gate / hc_* / attn_sink |

现有分流规则全覆盖，反量化不会漏也不会崩。产物 567GB / 48 分片。

### 2.3 量化 BF16 → W8A8（两份）

专家 INT8 per-channel 对称量化，注意力/共享专家/router 保持 BF16。因为要同时服务两个镜像，
量化脚本加了两个开关（见 `scripts/quant_w8a8_0731.py`）：

| 产物 | 参数 | 用途 | 大小 |
|---|---|---|---|
| `dsv4-0731-w8a8` | `--exclude-prefix mtp.`（默认） | 0728 镜像，剔除 DSpark | 273GB / 45 分片 |
| `dsv4-0731-w8a8-dspark` | `--exclude-prefix "" --keep-dspark-config` | 0811 镜像，保留 DSpark | 292GB / 46 分片 |

config.json 直接写入 sglang 兼容的 ignore 规则（负向先行断言 `re:^(?!.*\.experts\.).*`），
省掉后续手工打补丁。

### 2.4 管线三个坑（都值得写给后来者）

1. **容器漏挂 `/opt/hyhal`** → `ImportError: librocm_smi64.so.2`。凡在海光镜像里跑 torch，
   哪怕纯 CPU 任务也必须挂 hyhal。
2. **`${EXCL:-mtp.}` 吞掉显式空值**：想传"不排除任何张量"时传 `EXCL=""`，但 `:-` 会把空串
   当成"未设置"从而取默认值，导致 DSpark 张量仍被剔除。应用 `${EXCL-mtp.}`（无冒号）。
3. **`pkill` 杀不掉 docker 容器**：宿主机 shell 被杀后，`docker run` 起的容器继续跑到最后，
   **覆盖了正确版本的产物**（表现为"DSpark 权重"里 mtp 张量为 0）。长任务容器必须
   `--name` 固定命名，用 `docker rm -f` 收尾。

---

## 3. 第一轮：0731 在现有生产栈上（0728 镜像，无投机）

| 指标 | 0731 | 4 月版对照 | 结论 |
|---|---|---|---|
| 单流解码（编程题） | **12.28 tok/s** | 12.1（无投机）/ 18.7（+MTP） | 与基础解码完全持平——架构未变 |
| 8 并发聚合 | 55.9 | 56.5（无投机） | 持平 |
| KV 池 | **894,720** | 600,832 | **+49%**（剔除 DSpark 省下的显存转给 KV） |
| 数学 37×89−156 | 3137 ✅ | ✅ | 量化精度正常 |
| Think（9.11 vs 9.9） | 干净，推理 602 字 | 干净，推理 151 字 | 无复读，**0731 推理更充分** |
| Tool Call | ✅ | ✅ | |
| 权重加载 | 45 分片全载入，**无 key 不匹配** | — | 证明剔除 mtp 的策略正确 |

**小结**：功能完全正常，速度与 4 月版基础解码持平，但暂时失去 1.55× 的 MTP 加速。
要拿回速度，必须解决 DSpark 支持问题。

---

## 4. 第二轮：把"不支持 K100-AI"的 0811 镜像救活

### 4.1 起点：一个被判死刑的镜像

`custom:sglang0.5.12-...-20260804-0006-deepseekV4-0811` 我们此前判定为"对 K100-AI 完全不可用"
（tilelang 报 `HCU arch gfx928 not supported for MLS/GEMM_MLS`）。重新检查后发现两件事：

1. 它**自带完整 DSpark 实现**（`deepseek_v4_dspark.py`、`dspark.py`、
   `--speculative-algorithm DSPARK` + 4 个专用参数）。
2. 它的 tag 写着 0.5.12，实际装的是 **sglang 0.5.15.post2.dev564**（`deepseek_v4.py`
   3251 行 vs 0728 的 1885 行）。
3. 它的 `sgl_kernel` / `lightop` / `tilelang` / `triton` / `flash_mla` 二进制里
   **都含 gfx928 代码对象**。

当初判死刑，是因为照抄了官方启动器——那套 env 是给 BW 卡（gfx936/938）调的。

### 4.2 逐关突破（4 关）

| 关卡 | 现象 | 解法 |
|---|---|---|
| 1. custom allreduce 挂死 | AR 初始化后彻底无日志推进 | 0.5.15 换了开关名：`SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0` + `--disable-custom-all-reduce`（旧的 `USE_DCU_CUSTOM_ALLREDUCE=0` 已失效） |
| 2. 权重加载死锁 | 8 rank 静默（老问题 A-1） | 同步加载补丁在 0.5.15 上同样适用 |
| 3. tilelang MHC 崩溃 | `gfx928 not supported for MLS/GEMM_MLS` | 真正的开关是 `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=0` + `SGLANG_OPT_USE_AITER_MHC_PRE/POST=0` + `SGLANG_DSV4_MHC_PREWARM=0`，落到 `hc_pre_torch_impl` 纯 torch 实现 |
| 4. 注意力解码后端 | `kernel` 运行时拒绝 gfx928；`triton` 报 unsupported；`torch` 仅 1.26 tok/s | **自写补丁**：在 `debug_flash_mla_adapter.py` 增加 triton 分支，路由到镜像自带的 `nsa/triton_decode` |

第 4 关是关键。HCU 平台被硬编码走 `DeepseekV4AttnBackend`，它调用的入口只认 kernel/torch；
而隔壁 `hip_flash_mla.py` 里 triton 分支是现成的，且 `triton_fp8_attention_fwd` 的文档字符串
明确写着"接受与 flash_mla 相同的 kwargs，多余键静默忽略"——**就是 drop-in 设计，只是没接线**。

### 4.3 第 5 关：DSpark 专属崩溃（移植上游修复）

打通后跑 DSpark，temperature>0 立即崩：

```
File ".../speculative/dflash_utils.py", line 784, in build_dflash_verify_target_probs
    target_probs = top_k_renorm_prob(...)
TypeError: 'NoneType' object is not callable
```

根因：ROCm/HCU 上 sgl_kernel 只暴露 Python 包装器、没有 native 的 `top_{k,p}_renorm_probs`，
模块顶部 import 失败后把两者置为 None。

巧的是海光**当天（2026-08-13）**刚提交了修复（`18e167b7`，加 torch 版实现并按能力判定），
我们直接移植，报错消失。

---

## 5. DSpark 实测结果

### 5.1 三方对比

| 指标 | 4 月版 + MTP（原生产） | 0731 无投机 | **0731 + DSpark** |
|---|---|---|---|
| 单流解码（编程题 600 tok） | 18.7（编程题 24） | 12.28 | **33.8**（三次复现 33.59/33.82/34.97） |
| DSpark accept len / rate | — | — | **4.38 / 0.68** |
| 8 并发聚合（贪心） | 59 | 55.9 | 50.2（8/8 成功） |
| 10 并发聚合（贪心） | — | — | 46.3（10/10 成功） |
| 23K prompt prefill | 343 tok/s | — | **437 tok/s** |
| 27.5K prompt prefill | — | — | 412 tok/s |
| 98K prompt prefill | 395 tok/s（4.1 min） | — | 369 tok/s（4.5 min） |
| KV 池 | 600,832 | 894,720 | **1,026,816** |
| 上下文上限 | 1M | 1M | 1M |
| Think / Tool Call / 数学 | ✅ | ✅ | ✅ 全部正常，Think 无复读 |

**DSpark 一次猜 5 个 token 能中 4.38 个**（接受率 0.68），远强于 EAGLE MTP 的链式猜测
（steps=3、topk=1，实测 1.55×）。

### 5.2 一个测量陷阱（重要）

长上下文 prefill 我们先后测到三个差异巨大的数字：

| 测法 | 结果 | 真相 |
|---|---|---|
| 服务就绪后立刻测 | 232 tok/s | **triton 首次 JIT 编译污染** |
| 同一 prompt 再测一次 | 2144 tok/s | **前缀缓存命中** |
| 每次变前缀 + JIT 预热后 | **437 tok/s** | 真实值 |

教训：测长上下文必须用**每次变化前缀**的脚本（见 `tests/longctx_fresh.py`，用标签 md5
生成唯一前缀），且要在 JIT 预热之后测。否则结论会完全相反。

---

## 6. 唯一阻碍：非贪心 + 高并发触发 GPU 硬件异常

### 现象

```
Callback: Queue 0x... aborting with error : HSA_STATUS_ERROR_EXCEPTION:
An HSAIL operation resulted in a hardware exception. code: 0x1016
```

随后 `Scheduler watchdog timeout (300s)`，服务不可用，必须重启容器。

### 隔离矩阵（每格独立实验，崩溃后重启再测）

| 并发 | temperature=0（贪心） | temperature=0.7（非贪心） |
|---|---|---|
| 1 | ✅ 33.8 tok/s | ✅ |
| 2 | ✅ | ✅ |
| 4 | ✅ 32.5 tok/s | ❌ GPU 硬件异常 |
| 8 | ✅ 50.2 tok/s | ❌ GPU 硬件异常 |
| 10 | ✅ 46.3 tok/s | 未测（已知会崩） |

四种缓解手段全部无效：

- 缩小 draft 块（`--speculative-dspark-block-size 3`，验证窗口 6→4）：**同样崩溃**
- 显式 `top_k=50` 走稀疏路径（绕开稠密全词表兜底）：成功率 0/8 → 4/8，**仍会挂起**
- `--max-running-requests 4` 限流（日志确认 `#queue-req: 4`，运行批量确实被限制）：**依然崩溃**
- 降低并发到 2：可用，但对 8-10 路场景没有实用价值

**定位**：崩溃在 DSpark 非贪心接受路径
（`dspark_components/kernels/dspark_accept.py::accept_sampling_triton`）。贪心路径走纯 torch 的
`compute_dflash_correct_drafts_and_bonus`，完全稳定。查 HYGON-AI/sglang-das 的
`v0.5.15.post1_dev` 分支，2026-08-05 之后**没有任何针对 hang/deadlock/并发退化的 DSpark 提交**，
属新发现，已整理为 Bug 报告条目。

---

## 7. 生产化建议

**可以现在就用**，前提是锁定贪心解码：

- 编程助手场景绝大多数使用 temperature=0，而**贪心路径 1/2/4/8/10 并发全部稳定**。
- 需要在网关层强制 `temperature=0`，因为任意一个 temperature>0 的请求都可能拖垮服务。

**保守方案**：生产维持 4 月版 + MTP（18.7 tok/s，任意温度全稳），等海光修复内核后再切。

两个方案的取舍：

| | DSpark（锁贪心） | 4 月版 + MTP |
|---|---|---|
| 单流 | 33.8 | 18.7 |
| 23K prefill | 437 tok/s | 343 tok/s |
| 模型能力 | 0731（DeepSWE 54.4） | 4 月版（DeepSWE 7.3） |
| 温度自由度 | ❌ 必须 0 | ✅ 任意 |
| 稳定性 | 贪心下稳定 | 全场景稳定 |

---

## 8. 复现所需

- 补丁：`patches/sglang-0811/`（三个补丁 + 启动器）
- 量化：`scripts/quant_w8a8_0731.py`、`scripts/run_0731_pipeline.sh`
- 启动：`scripts/start_0731_dspark.sh`（0811+DSpark）、`scripts/start_0731_base.sh`（0728 基线）
- 测试：`tests/longctx_fresh.py`、`tests/bench_conc_param.py`、`tests/test_coding_speed.py`
- Bug 报告：`docs/Bug报告-DSpark与triton路由.md`

模型权重不在本仓库（需自行下载 `deepseek-ai/DeepSeek-V4-Flash-0731` 并按第 2 节量化）。
