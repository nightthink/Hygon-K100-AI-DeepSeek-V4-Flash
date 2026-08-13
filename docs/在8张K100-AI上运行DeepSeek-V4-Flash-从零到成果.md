# 在 8 张 K100-AI 上运行 DeepSeek-V4-Flash：从零到成果

> **验证主机**：nodeA（2×48 核 CPU / 512GB 内存 / 8×海光 K100-AI 64GB / gfx928 / DTK 26.04 / 内核 6.8.0-136-generic）
> **时间跨度**:2026-08 上旬（FlagOS 线与 vLLM 线 8/9–8/11，sglang 线 8/11–8/12）
> **模型**：DeepSeek-V4-Flash——284B MoE（激活 13B），43 层，256 路由专家，DSA 稀疏注意力（C4A + SWA + Lightning Indexer）+ mHC 超连接；原始权重注意力/密集层 FP8（e4m3, ue8m0 128×128 块 scale）、专家 MXFP4；上下文 1M。
> **最终状态**：**三条从零到 OpenAI 兼容服务的路线全部打通**。sglang 线为当前生产（单流 18.7 tok/s、8 并发 59.4 tok/s、Think/Tool Call/前缀缓存全开）；vLLM 线为热备回退（单流 9.3 tok/s、128 并发 116 tok/s）；FlagOS 线是唯一的**原始权重零重量化**路线（端到端正确，decode 0.1 tok/s，作精度上限参考与技术储备）。
> **配套文件**：详细 Bug 报告独立成文《K100AI-DeepSeek-V4-Flash-Bug报告-提交海光.md》；三条路线各有一个完整打包件（见第 6 节资产总表）。

---

## 0. 总览：三条路线的最终成绩

| 维度 | 主线一：vLLM 线 | 主线二：sglang 线（生产） |
|---|---|---|
| 镜像 | `<internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706` | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:dsv4-flash-k100ai-sglang0.5.12-20260728` |
| 推理框架 | vLLM 0.21.0 + vllm_hcu 平台插件 | sglang 0.5.12 + sglang-dsv4-flash-triton-patch 0.4.7（镜像内置） |
| 权重 | dsv4-flash-w8a8（同一份） | dsv4-flash-w8a8（同一份，config.json 增补 1 条 ignore 规则） |
| 我方补丁 | 6 个文件级补丁（344 行 diff） | 1 个加载器补丁 + 1 条 config 规则 + 启动器 2 处修改 |
| 单流解码 | 9.3 tok/s | **18.7 tok/s（2.0×）**，MTP EAGLE steps=3 |
| 8 并发聚合 | 23.1 tok/s | **59.4 tok/s（2.6×）** |
| 高并发聚合 | 128 并发 116 tok/s（吞吐王） | 未做 128 并发调优（8–10 路目标已满足） |
| TTFT | 未专项测量 | 短 prompt 0.3s；5K prompt 首次 8.3s、前缀缓存命中 **0.8s** |
| Think（推理） | 模型能力在，未接解析器 | **原生支持**：`--reasoning-parser deepseek-v4`，请求加 `chat_template_kwargs={"thinking":true}` |
| Tool Call | 未接解析器 | **原生支持**：`--tool-call-parser deepseekv4` |
| 上下文/KV 池 | 256K ctx 验证过（32K 生产默认） | 模型 1M ctx，KV 池 324,352 token（bf16 KV） |
| MTP 投机解码 | **失败**（8 并发 GPU VMFault，已回退） | **成功**（steps=3，8 并发稳定） |
| 定位 | 热备回退 + 高并发场景 | **生产主线**（编程助手：单流优先、8–10 路、≤20 用户、Think 必备） |

**路线三：FlagOS vLLM 线**（详见第 4 节）——镜像 `harbor.baai.ac.cn/flagos21-release/vllm-plugin-fl:v0.2.0-rc2-hygon`（FL vLLM 0.20.0 + flagtree 0.5.0+hcu + flag_gems 5.0.2），**原始 FP8+MXFP4 权重直接服务、零重量化**。历经 24 个错误/19 个补丁后端到端打通：输出连贯、`/v1/chat/completions` 正确返回；decode ≈ 0.1 tok/s（逐 token PyTorch 回退所致），故不作生产而作**精度上限参考**（三线中唯一不经任何重量化的），其平台无关补丁可移植到任何非 CUDA 平台的同版本 vLLM。

三线关系：FlagOS 线最早攻坚，趟出了"gfx928 无 FP8 硬件"等全部底层结论并提供了数值参考实现（vLLM 线补丁的逐位对账基准）；vLLM 线建立了**权重加工流水线（orig→bf16→w8a8）**；sglang 线复用该流水线现货权重——官方 K100-AI 路线恰好要求 INT8-W8A8-per-channel，这是 sglang 线一天内打通的关键。

---

## 1. 公共基础（vLLM 线与 sglang 线共用；FlagOS 线直接用 1.2 下载的原始权重，无需 1.3 的转换）

### 1.1 前置条件

| 项 | 要求 |
|---|---|
| GPU | 8×K100-AI（64GB，gfx928），DTK 26.04，`/opt/hyhal` 存在 |
| 磁盘 | ≥1.2TB（orig 150GB + bf16 中间产物 568GB + w8a8 成品 299GB），挂 `/data1` |
| 网络 | 可访问 ModelScope、GitHub、harbor.sourcefind.cn:5443（公开匿名拉取）|
| 约定 | 容器内跑 python/torch 必带 `-v /opt/hyhal:/opt/hyhal`，否则 torch 报 `librocm_smi64.so.2` 缺失（纯 CPU 工作也一样） |

### 1.2 下载原始权重与工具仓库

```bash
pip install modelscope

# DeepSeek 官方原始权重（FP8 + MXFP4 专家，~150GB）
modelscope download --model deepseek-ai/DeepSeek-V4-Flash \
  --local_dir /data1/models/DeepSeek-V4-Flash-orig

# FlagOS 适配仓库（提供 FP4/FP8→BF16 的 convert_weight.py）
git clone https://github.com/flagos-ai/DeepSeek-V4-FlagOS /data1/DeepSeek-V4-FlagOS
```

### 1.3 权重加工流水线：orig → BF16 → W8A8（两线共用的核心资产）

**为什么必须转换**：K100-AI（gfx928）**没有 FP8/FP4 硬件指令**（本项目实测硬结论，见第 7.2 节），原始权重直接喂任何一线都走不通。可行路径：专家量化为 INT8 W8A8（compressed-tensors 格式，逐通道对称权重 + 动态 per-token 激活，无需校准），注意力/共享专家/router/lm_head 保持 BF16。

**第一步：FP4/FP8 → BF16 反量化**（产物 568GB，DCU 加速）：

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 --entrypoint bash \
  <internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706:latest
# 容器内：
cd /data1/DeepSeek-V4-FlagOS
python convert_weight.py \
  --input-fp4-hf-path  /data1/models/DeepSeek-V4-Flash-orig \
  --output-bf16-hf-path /data1/models/dsv4-flash-bf16 --device cuda
```

转换后**必须删掉 config.json 残留的 `quantization_config`**（否则 vLLM 走错加载路径）：

```python
import json
p = "/data1/models/dsv4-flash-bf16/config.json"
cfg = json.load(open(p)); cfg.pop("quantization_config", None)
json.dump(cfg, open(p, "w"), indent=2)
```

**第二步：BF16 → W8A8 INT8 量化**（产物 299GB，纯 CPU 流式，1–2 小时）：

```bash
docker run --rm -v /opt/hyhal:/opt/hyhal -v /data1:/data1 --entrypoint python3 \
  <internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706:latest \
  /data1/quant_w8a8_stream.py \
    --input /data1/models/dsv4-flash-bf16 --output /data1/models/dsv4-flash-w8a8
# 完成标志：quantized expert weights: 33792，46 个分片，约 299GB
```

`quant_w8a8_stream.py`（部署套件内）要点：只量化名字含 `.experts.`、以 `.weight` 结尾的二维张量；`scale = absmax/127`（输出通道维）；写入 `quantization_config`：`targets: ["re:.*\\.experts\\..*"]`、weights int8/channel/symmetric、input_activations int8/token/dynamic、ignore lm_head。

**第三步（仅 sglang 线需要）：config.json 增补 ignore 规则**。sglang 的 compressed-tensors 实现要求**每一个 Linear 层都必须命中 targets 或 ignore**（vLLM 对未命中层默认不量化，sglang 直接报错 `Unable to find matching target for model.layers.0.self_attn.wqkv_a`）。修复（原件备份为 `config.json.bak_sglang`）：

```python
import json, shutil
p = '/data1/models/dsv4-flash-w8a8/config.json'
shutil.copy(p, p + '.bak_sglang')
c = json.load(open(p))
# 负向前瞻正则：experts 之外的所有层都进 ignore（ignore 优先级高于 targets，
# 且不覆盖 experts，故量化范围不变）
c['quantization_config']['ignore'] = ['lm_head', 're:^(?!.*\\.experts\\.).*$']
json.dump(c, open(p, 'w'), indent=2)
```

> 该修改对 vLLM 线无害（这些层本来就是 BF16 不量化），两线可共用同一份权重目录。

---

## 2. 主线一：vLLM 线（热备回退，高并发吞吐王）

### 2.1 镜像

`<internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706:latest`
（vLLM 0.21.0 + vllm_hcu 平台插件 + DAS 版 torch 2.10/triton/lightop 0.6.0(0616 build)/tilelang，来自海光私有渠道 0706 版）

### 2.2 六个文件级补丁（全部以 docker -v 挂载覆盖，不改镜像）

补丁成品是 6 个 `*_patched.py` 完整文件；unified diff 共 344 行在 `/data1/patches/`（打包件 `dsv4_full_kit_20260811.tar.gz` 内均有）。对应关系与根因：

| # | diff | 容器内目标 | 根因一句话 |
|---|---|---|---|
| 01 | 01-mhc.diff (47行) | `vllm/model_executor/layers/mhc.py` | gfx928 TileLang 不支持 PDL，海光漏删 5 处 |
| 02 | 02-deepseek_v4_attention.diff (176行) | `vllm/model_executor/layers/deepseek_v4_attention.py` | 三处：转置布局误判 / `_C` 融合算子写坏 KV / indexer FP8 布局 |
| 03 | 03-vllm_hcu-rocm_aiter_mla_sparse.diff (66行) | `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py` | indexer 写 FP8 读 bf16 矛盾 |
| 04 | 04-vllm-rocm_aiter_mla_sparse.diff (28行) | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | wo_a 转置 view 静默乱码（**最终元凶**）⚠️ 与 03 是两份文件都要打 |
| 05 | 05-kv_cache_utils.diff (13行) | `vllm/v1/core/kv_cache_utils.py` | max_concurrency 除零（仅日志） |
| 06 | 06-worker_utils.diff (14行) | `vllm/v1/worker/utils.py` | "No common block size" 处加诊断 |

**补丁 01（mhc.py，PDL 移除）**——5 处同型修改，全文模式：

```diff
     with T.Kernel(num_tokens, threads=96) as i:
-        T.pdl_sync()
+        pass  # T.pdl_sync()  # patched: no PDL on gfx928
...（共 4 处 pdl_sync + 1 处 pdl_trigger，全部替换为 pass）
-        T.pdl_trigger()
+        pass  # T.pdl_trigger()  # patched: no PDL on gfx928
```

**补丁 02（deepseek_v4_attention.py）三处根因与修法**：

- (a) compressor 输入投影：海光可插拔 Linear 把 `fused_wkv_wgate.weight` 存成转置布局 `[in,out]`，upstream 直接 `weight.T` 导致 shape 不匹配。修复：按实际形状自适应决定是否转置。
- (b) C++ 融合算子 `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 在 gfx928 写出的 FP8 KV + scale 无法被 upstream triton 反量化内核解读（收集出的 KV 815/2560 个 NaN → 第 0 层注意力全 NaN → 全模型乱码）。修复：保留算子调用（无害），q 侧用 python 重实现 per-head RMSNorm + GPT-J RoPE（与 FlagOS 参考逐位对齐 cos=0.999996），KV 侧 python 做 RoPE 后用 upstream triton `quantize_and_insert_k_cache` **重写**缓存条目（与读取端同源同布局）。
- (c) indexer 缓存维持 FP8 uint8 布局（`head_dim + head_dim//128*4` 字节），配合补丁 03 的 FP8 端到端路径。

**补丁 03（vllm_hcu 份 mla_sparse）**：海光把 Lightning Indexer 插入/收集内核换成 bf16 版，但 V4 的 indexer 缓存实际由 compressor 融合内核以 FP8 布局写入（indexer 创建时 `skip_k_cache_insert=True`，海光改的插入路径根本不执行）——写 FP8 读 bf16。修复：收集分支改回 upstream FP8 内核 `cp_gather_indexer_k_quant_cache_triton`（带 k_scale 与 token_to_seq），`fp8_dtype` 用 `current_platform.fp8_dtype()`，q 保持 FP8（lightop `mqa_logits` 支持 FP8×FP8）。

**补丁 04（vllm 份 mla_sparse，最终乱码元凶）**：`rocm_inv_rope_einsum` 里 `wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim)`——运行时权重实为转置布局 `[4096,1024]`，直接 view 是对转置矩阵的**静默内存重解释**，o 投影输出全乱但不报错（逐层 cos 对账：注意力核心 >0.999，过 o 投影骤降到 0.017）。修复：

```diff
-    wo_a_w = self.wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim)
+    w = self.wo_a.weight
+    if w.shape[0] == n_local_groups * o_lora_rank:   # 原生布局
+        wo_a_w = w.view(n_local_groups, o_lora_rank, hidden_dim)
+    else:                                             # 海光转置布局 [hidden, groups*o_lora]
+        wo_a_w = w.view(hidden_dim, n_local_groups, o_lora_rank).permute(1, 2, 0)
```

**关键陷阱**：该文件在 vllm 和 vllm_hcu **各有一份**，`deepseek_v4_attention.py` import 的是 vllm 那份——只补 vllm_hcu 份时补丁不生效（实际踩过）。

### 2.3 启动（生产调优版 `/data1/start_dsv4_tuned.sh`）

```bash
#!/bin/bash
set -e
IMG=<internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706:latest
docker rm -f vllm-dsv4 2>/dev/null || true
docker run -d --name vllm-dsv4 --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --restart unless-stopped \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 \
  -v /data1/mhc_patched.py:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/mhc.py \
  -v /data1/dsv4_attn_patched.py:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/deepseek_v4_attention.py \
  -v /data1/mla_sparse_patched.py:/usr/local/lib/python3.10/dist-packages/vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py \
  -v /data1/vllm_mla_sparse_patched.py:/usr/local/lib/python3.10/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py \
  -v /data1/kv_cache_utils_patched.py:/usr/local/lib/python3.10/dist-packages/vllm/v1/core/kv_cache_utils.py \
  -v /data1/worker_utils_patched.py:/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/utils.py \
  -e NCCL_P2P_DISABLE=1 \
  $IMG \
  vllm serve /data1/models/dsv4-flash-w8a8 \
    --tensor-parallel-size 8 --kv-cache-dtype fp8 --block-size 256 \
    --enforce-eager --disable-custom-all-reduce \
    --disable-hybrid-kv-cache-manager \
    --served-model-name deepseek-v4-flash \
    --max-model-len 262144 --max-num-seqs 128 \
    --gpu-memory-utilization 0.92 --port 8000
```

非默认参数逐条原因（**不要随意去掉**）：

| 参数 | 原因 |
|---|---|
| `--kv-cache-dtype fp8` | DeepseekV4 硬性要求 fp8_ds_mla 缓存格式 |
| `--block-size 256` | IndexerBackend/FlashMLASparse 只声明支持 256（默认 16 报 "No common block size"） |
| `--enforce-eager` | CUDA graph 实测负优化（见 2.5） |
| `--disable-custom-all-reduce` + `NCCL_P2P_DISABLE=1` | 本机 PCIe 拓扑下 custom allreduce/P2P 在权重加载后挂死（GPU 0%、无日志） |
| `--disable-hybrid-kv-cache-manager` | 混合 KV 管理器 page 断言在本模型多 spec 组下不成立 |
| `--max-num-seqs 128` | 调优结果：32→128 是并发吞吐 +66% 的唯一来源 |

### 2.4 验收与性能

```bash
bash /data1/test_dv4.sh          # 7 项验收（脚本全文见第 5 节）
python3 /data1/bench_concurrency.py 128 128   # 并发压测
```

结果（全部通过）：自我介绍正常、37×89−156=**3137**（t=0）、素数函数正确（6k±1 实现）、流式正常。吞吐曲线（128-token 短请求）：

| 并发 | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| 聚合 tok/s | 9.1–9.3 | 23.1 | 69.6 | 76.5 | **116.3** | 117.9（饱和） |

### 2.5 失败的尝试与放弃原因（vLLM 线）

1. **去掉 `--enforce-eager`（CUDA graph）**：能捕获能跑、正确性过，但 32 并发吞吐**降 14%**（59.6 vs 69.6）。gfx928 上 captured graph 对本负载无益 → 回滚保留 eager。
2. **`max-num-seqs` 提到 256**：128 并发无增益（117.9 vs 116.3）只推高排队延迟 → 定格 128。
3. **MTP 投机解码（EAGLE）**：单流 12.5 tok/s（+34%）、Think 14.3 tok/s，诱人；但 **≥8 并发触发 GPU VMFault**（spec=1 与 spec=2 均崩，可稳定复现），栈级 bug 补丁层面不可修 → 全部回退。实验配置归档 `/data1/start_dsv4_mtp_experimental.sh`，最小复现 `/data1/mtp_vmfault_repro.sh`（供海光排障）。附注：MTP 是分布无损的投机解码（draft 由目标模型验证），对 Think 质量无影响，t=0 因批验证数值并列裁决非逐字节复现但质量等价——该结论在 sglang 线复用。
4. **python→Triton 逆 RoPE 优化**（`inv_rope_heads_` 内核，逐位对齐、单层 3.7×）：数值正确但端到端收益被 CUDA graph FULL_DECODE_ONLY 吸收（decode 回放期 python 开销本就不在关键路径），且 eager 才是生产态 → 保留代码（`/data1/dsv4_fused_ops.py`）未上生产。

---

## 3. 主线二：sglang 线（当前生产，单流王 + 全能力）

### 3.1 镜像的来历（harbor 探索）

通过 harbor 匿名 API 枚举海光公开仓库发现了官方 DSv4 sglang 适配线：

```bash
# 列出 dcu 项目所有镜像 tag（匿名可访问）
curl -sk "https://harbor.sourcefind.cn:5443/api/v2.0/projects/dcu/repositories/admin%252Fbase%252Fcustom/artifacts?with_tag=true&page_size=40"
```

相关 tag 两个：

| tag | 判定 |
|---|---|
| `sglang0.5.12-...-20260804-0006-deepseekV4-0811` | ❌ 面向新一代 BW 卡（gfx936/938）：其 tilelang 的 HCU GEMM 仅支持 gfx938/92a/946（MLS 指令），deepgemm 无 gfx928 —— **K100-AI 不可用**（实测两条路全死，见 3.2） |
| `dsv4-flash-k100ai-sglang0.5.12-20260728` | ✅ **K100-AI 专用**（tag 带 k100ai），内置 DSv4 专用补丁包，最终生产采用 |

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:dsv4-flash-k100ai-sglang0.5.12-20260728
```

### 3.2 试错全记录（六幕，均为后来者避坑）

**第一幕：0811 镜像 + 原始权重 → 权重加载死锁 8 小时**。8 个 rank 串行打出 `Execute dequant fp8 wo_a` 后全体静默，health 000。py-spy 两次采样（间隔 20s）帧完全相同：主线程 `as_completed()` 等 futures，ThreadPoolExecutor 工作线程冻结在 `fused_moe layer.py:584 expert_data.copy_(loaded_weight)`（H2D 拷贝）；CPU 0%、磁盘 0 I/O、VRAM 均衡 37–42%、dmesg 无 VMFault。**结论：多线程 H2D 权重拷贝在该 DTK/HIP 栈死锁**。→ 催生同步加载补丁（3.3.1），加载从死锁变 3 分钟。

**第二幕：0811 + 同步补丁 → CUDA graph 捕获时 tilelang 编译失败**：`HCU arch gfx928 not supported for MLS/GEMM_MLS; supported: gfx938, gfx92a, gfx946`，出自 mhc_pre 的 tilelang splitk 内核。且 mhc_pre 另一分支依赖 deepgemm（该镜像 deepgemm 仅 gfx92a/936/938）。两分支全死 → **判定 0811 镜像面向 BW 卡，放弃**，转 0728。

**第三幕：0728 + 原始权重 + 沿用 vllm 线挂载习惯 → deepgemm dlopen 失败**：`Failed to load .../deep_gemm/_C.so: libcudart.so.13: cannot open shared object file`——镜像内 deep_gemm 竟是 **CUDA 编译版**（链接 libcudart），国产平台不可加载。mhc_pre 默认走 deepgemm 分支（`SGLANG_OPT_DEEPGEMM_HC_PRENORM` 默认 True）。→ 环境变量绕过：`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` + `SGLANG_ENABLE_JIT_DEEPGEMM=0`。

**第四幕：绕过后 → 首次 forward 段错误**（CUDA graph bs=256/bs=16 捕获、以及 `--disable-cuda-graph` 纯 eager 均崩）：`Fatal Python error: Segmentation fault`，栈在 `triton/compiler/compiler.py:474 _init_handles` ← `rms_normalize_triton`，伴随 HIP 运行时 `HOSTQUEUE` 报错。一度怀疑宿主 hyhal 挂载与镜像不配 → 去掉挂载后 torch 直接 `librocm_smi64.so.2` 缺失，证伪（**该挂载必须保留**）。

**第五幕：破案**。镜像 `/root/.bash_history` 里留有海光工程师的完整调试史，揭示官方用法三要素我们全没用：
1. 镜像内置补丁包 `sglang-dsv4-flash-triton-patch 0.4.7`（已 install，25 文件，含 FlashMLA triton/triton_logic decode 后端、DSV4 sparse prefill Triton 接口、MQA/paged MQA 内核、packed KV 解量化、gfx928 MoE align 的 LightOp 路由、K100_AI/INT8-W8A8-per-channel MoE 调优配置）；
2. `sglang-dsv4-flash-patch write-launchers --output-dir ...` 可生成**官方启动器**（含全套必需环境变量——缺它们正是段错误根源）；
3. **权重必须是 INT8-W8A8-per-channel**（`--quantization compressed-tensors`），不是原始 FP8/FP4 —— 我们 vLLM 线的 w8a8 权重正好现货匹配。

```bash
# 提取官方启动器（7 个脚本）
docker run --rm -v /data1:/data1 --entrypoint bash <0728镜像> -c \
  "sglang-dsv4-flash-patch status && \
   sglang-dsv4-flash-patch write-launchers --output-dir /data1/sglang_patches/launchers"
# 产出：run_ds_mtp.sh（主脚本）、run_ds_mtp_triton_logic_bf16_kv.sh、
#       run_ds_nomtp_triton_logic_bf16_kv.sh、run_ds_nomtp_torch_native_bf16_kv.sh、
#       run_ds_mtp_triton_logic.sh、run_cilent.sh、run_evalscope.sh
```

**第六幕：官方启动器 + w8a8 权重 → 一个 config 报错后跑通**。`Unable to find matching target for model.layers.0.self_attn.wqkv_a in the compressed-tensors config` → 第 1.3 节第三步的 ignore 规则修复 → **服务起来，验收全过**。

### 3.3 我方补丁（共 3 处，内容全文）

**3.3.1 加载器同步化补丁**（防多线程 H2D 死锁；对 0811/0728 两镜像同代码）。从镜像拷出 `sglang/srt/model_loader/utils.py`，将 `should_async_load` 改为恒返 False，保存为 `/data1/sglang_patches/model_loader_utils_patched_0728.py`，启动时 `-v` 挂载覆盖：

```diff
 def should_async_load(weight: torch.Tensor) -> bool:
     """Return True if we should load the given weight asynchronously. ..."""
-    device = getattr(weight, "device", None)
-    if device is None:
-        return False
-    return device.type == "cpu"
+    # PATCH(gfx928/K100-AI): multithreaded H2D weight copies deadlock on this
+    # DTK/HIP stack (all 8 ranks hung in expert_data.copy_ for 2h+, 0 CPU).
+    # Force synchronous loading.
+    return False
```

代价：加载不并行。实测无痛——页缓存热时 46 分片 7–11 秒读完，全程加载 178 秒。

**3.3.2 权重 config.json ignore 规则**：见第 1.3 节第三步（`'re:^(?!.*\\.experts\\.).*$'`）。

**3.3.3 官方启动器 run_ds_mtp.sh 的 2 处修改**（原件备份 `.bak`/`.bak2`）：

```diff
   --moe-a2a-backend none \
-  --disable-radix-cache \
+  --reasoning-parser deepseek-v4 \
+  --tool-call-parser deepseekv4 \
   --chunked-prefill-size 8192 \
   --quantization compressed-tensors \
-  --chat-template ./tool_chat_template_deepseekv3.jinja \
   --kv-cache-dtype "$KV_CACHE_DTYPE" \
```

理由：删 `--chat-template`——模型目录无该 jinja 文件（DSv4 用 `encoding/encoding_dsv4.py` 规范而非 jinja，sglang 内置模板工作正常）；删 `--disable-radix-cache`——启用前缀缓存后同前缀 TTFT 10×改善且未见副作用；加双解析器——Think 与 Tool Call 经 OpenAI API 输出。

### 3.4 官方启动器的关键内容（run_ds_mtp.sh 环境变量全集）

```bash
# 平台稳定性
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
# allreduce：禁 DCU 定制版，强制 torch allreduce
export USE_DCU_CUSTOM_ALLREDUCE=0
export FORCE_TORCH_AR=1
# 算子路由（gfx928 关键）
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0   # deepgemm 为 CUDA 版不可用
export SGLANG_USE_FP8_W8A8_MOE=false      # 专家是 INT8，禁 FP8 MoE 分支
export SGLANG_GROUPGEMM=true
export SGLANG_USE_LIGHTOP=1               # lightop topK/moe_align/moe_sum
export SGLANG_ROCM_USE_AITER_MOE=false    # AITER moe_sorting_ck 无 gfx928
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_LIGHTOP_TOPK=true
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_NSA_FUSE_TOPK=false         # 官方注释：绕过以隔离崩溃
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_DEEPEP_TP_SHARD_QUANT=0
# 变体开关（由包装脚本设定）
#   bf16 KV 变体：SGLANG_DSV4_INT8_KV_CACHE=false + KV_CACHE_DTYPE=bfloat16
#   MTP 开关：   ENABLE_MTP=true/false，SPECULATIVE_NUM_STEPS=3
#   注意力后端：  SGLANG_HACK_FLASHMLA_BACKEND=triton_logic

sglang serve \
  --port "$PORT" --trust-remote-code --model-path /models --tp 8 \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --nsa-prefill-cp-mode round-robin-split \
  --moe-a2a-backend none \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --chunked-prefill-size 8192 \
  --quantization compressed-tensors \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --disable-flashinfer-autotune \
  [MTP 时追加] --speculative-algorithm EAGLE --speculative-num-steps 3 \
               --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

### 3.5 生产启动（`/data1/start_sglang_dsv4_prod.sh` 全文）

```bash
#!/bin/bash
# DeepSeek-V4-Flash 生产启动脚本（sglang 线，2026-08-12 调优定稿）
set -e
docker rm -f sglang-dsv4 2>/dev/null || true
IMG=harbor.sourcefind.cn:5443/dcu/admin/base/custom:dsv4-flash-k100ai-sglang0.5.12-20260728
docker run -d --name sglang-dsv4 --restart unless-stopped --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 \
  -v /data1/models/dsv4-flash-w8a8:/models \
  -v /data1/sglang_patches/model_loader_utils_patched_0728.py:/usr/local/lib/python3.10/dist-packages/sglang/srt/model_loader/utils.py:ro \
  -e NCCL_P2P_DISABLE=1 -e PORT=8000 \
  -w /data1/sglang_patches/launchers \
  --entrypoint bash "$IMG" run_ds_mtp_triton_logic_bf16_kv.sh
echo "服务启动中，约 6 分钟就绪：curl http://127.0.0.1:8000/health"
```

要点：模型以 `-v .../dsv4-flash-w8a8:/models` 挂到官方脚本的固定路径；工作目录设为启动器目录（脚本用 `$PWD` 放 logs/moe_configs）；`--ulimit memlock=-1` 对应脚本内 `ulimit -l unlimited`。

### 3.6 验收与性能（生产配置：MTP steps=3 + triton_logic + bf16 KV + 前缀缓存 + 双解析器）

| 测试 | 结果 |
|---|---|
| /health | 200，加载全程约 5.5–6 分钟（页缓存热） |
| 自我介绍 | "你好！我是DeepSeek，一个由深度求索公司创造的AI助手……" ✓ |
| 37×89−156（t=0） | **3137** ✓ |
| 素数函数（t=0） | 标准 6k±1 实现 ✓ |
| **单流解码** | **18.7 tok/s**（写 200 字任务，256 max_tokens；无 MTP 时 12.1） |
| **8 并发聚合** | **59.4 tok/s**（per-request 7.4 tok/s），健康检查通过、无 VMFault |
| TTFT 短 prompt | 0.26–0.34 s |
| TTFT 5K prompt | 首次 8.3s；**前缀缓存命中 0.82s（10×）**，日志 `#cached-token: 2560` 佐证 |
| Think | `chat_template_kwargs={"thinking":true}` → `reasoning_content` 正确分离，9.11 vs 9.9 推理正确（101 reasoning tokens，无复读） |
| Tool Call | get_weather 冒烟：`tool_calls` 解析出正确函数名/JSON 参数，`finish_reason=tool_calls` ✓ |
| KV 容量 | max_total_num_tokens=324,352（bf16 KV），模型 context_len=1M |
| 稳态 | 运行 29 分钟复测全绿，8 卡 VRAM 均衡 80% 无泄漏迹象 |

**客户端用法示例**（Think 开关按请求粒度控制）：

```bash
# 普通对话（chat 模式，直接回答）
curl http://nodeA:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "你好"}]}'

# Think 模式（返回 reasoning_content + content 两段）
curl http://nodeA:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "9.11 和 9.9 哪个大？"}],
  "chat_template_kwargs": {"thinking": true}}'
```

### 3.7 失败的尝试与放弃原因（sglang 线调优阶段）

1. **SPECULATIVE_NUM_STEPS=4**：单流 19.06 tok/s（仅 +2%，噪声级），8 并发 **53.1 tok/s（−11%）**——投机步数加大后验证开销在并发下反噬。→ 定格官方默认 steps=3。
2. **int8 KV cache**（官方 run_ds_mtp.sh 默认）：容量仅 +17%（379,136 vs 324,352，因 DSA 压缩 KV 本身很小，int8 只作用于部分缓存），且 **Think 推理出现复读循环**（"实际上，9.9更大。实际上……"重复到耗尽 token，content 为空；bf16 KV 同题干净利落 101 token）。Think 是硬需求 → **否决 int8 KV，bf16 KV 定稿**。
3. **0811 镜像全线**（第 3.2 节第一、二幕）：多线程加载死锁（补丁可修）+ tilelang 无 gfx928 GEMM（不可修）→ 放弃该镜像，此判断节约了继续深挖的时间。
4. **原始 FP8/FP4 权重直接喂 sglang**：段错误根源之一；官方路线本就要求 W8A8 → 放弃，改用现货 w8a8。


---

## 4. 路线三：FlagOS vLLM 线（成功——原始权重零重量化直服，性能受限）

> **结论先行**：这条线是**成功的**——2026-08-11 在 nodeA 上以 vLLM OpenAI 兼容服务完整拉起，`/v1/chat/completions` 端到端返回正确、输出连贯。它是三条线中唯一**不经任何重量化**（原始 FP8 + MXFP4 权重直接服务）的路线，精度上限最高；代价是 decode ≈ 0.1 tok/s（关键算子逐 token PyTorch 回退），因此定位为精度参考与技术储备而非生产。

### 4.1 环境与镜像

| 项 | 值 |
|---|---|
| 镜像 | `harbor.baai.ac.cn/flagos21-release/vllm-plugin-fl:v0.2.0-rc2-hygon`（FL vLLM 0.20.0 + flagtree 0.5.0+hcu + flag_gems 5.0.2，OOT 平台插件） |
| 权重 | `/data1/models/DeepSeek-V4-Flash-orig` 原始权重直用（容器内挂载为 `/model`），**无需任何转换** |
| 服务 | 端口 8001，`served_model_name=deepseek-v4-flash`，max_seq_len 8192，enforce_eager |
| 附加依赖 | tilelang wheel 目录 `/home/user/tilelang_pkg/`（tilelang-0.1.8 whl + apache-tvm-ffi==0.1.2；tilelang 已不参与计算，但 mhc.py 导入守卫依赖 `has_tilelang()` 为真，目前仍需安装） |

平台核心难点（该线趟出的底层结论，反哺另两条线）：gfx928 无 FP8/MXFP4 硬件指令且 FL 插件注册为 OOT 平台（`is_cuda_alike()=False`，大量上游平台分支走不进）；上游为 V4 写的关键算子全是 CUDA-only 编译扩展（DeepGEMM、FlashMLA `_flashmla_C`、`_moe_C`/`_C` 融合内核），镜像里没有；**tilelang 0.1.8 官方 wheel 只含 CUDA codegen**（`strings libtilelang.so` 无 `target.build.tilelang_hip`）——确认后彻底放弃 tilelang，全面转向 flagtree Triton + PyTorch 回退。

### 4.2 最终架构（各计算路径落点）

| 计算路径 | 最终方案 |
|---|---|
| FP8 block-scaled 线性层 | vLLM 自带 Triton 内核 `w8a8_triton_block_scaled_mm`（flagtree hcu 实测可用，相对误差 ~1.5%）；UE8M0 权重 scale 转 float32 |
| MXFP4 MoE 专家 | vLLM EMULATION 后端（Triton 在线反量化到 BF16），`QUARK_MXFP4_IMPL=triton` |
| MoE 路由 / Router GEMM | 纯 PyTorch 参考实现 / fp32 `F.linear` 回退（CUDA 快路径缺失） |
| fused_qk_rmsnorm | 重写内核结构绕过 hcu 编译器分支合并指针 bug（Patch 11，bit-exact） |
| Q-norm+RoPE+KV 量化入缓存 | 纯 PyTorch 逐字节复刻 CUDA 融合核语义（Patch 18） |
| o-proj fp8_einsum（DeepGEMM） | 纯 PyTorch 反量化 + einsum 回退（Patch 13） |
| Lightning Indexer | CUDA 编译算子复用（`_C.top_k_per_row_*` 等镜像内存在），仅 DeepGEMM 的 MQA logits 用 PyTorch 回退（Patch 16） |
| FlashMLA 稀疏注意力 | 纯 PyTorch 回退（Patch 19）：prefill topk 注意力 + decode 双 fp8 分页缓存反量化合并 + attention sink |

### 4.3 24 个错误 / 19 个补丁（摘要；逐条全文见包内 DEPLOYMENT.md）

- **第一阶段（Error 1–10 → 基础补丁）**：让 OOT 平台认识模型——MXFP4 EMULATION 后端注册/直通/设备检查放行、mhc tilelang 导入守卫与 PyTorch 回退、DeepGEMM `tf32_hc_prenorm_gemm` 回退等。
- **第二阶段（Error 11–12，tilelang 之死）**：`target.build.tilelang_hip` 在官方 wheel 中不存在（非注册问题，是功能不存在）→ 放弃 tilelang，FP8 线性改 `TritonFp8BlockScaledMMKernel` 子类（shim 文件类名不变，其余组件零改动）。
- **第三阶段（Error 13–18 → Patch 11–15）**：hcu 编译器 if/else 分支合并不同指针崩溃（重写内核结构）、`launch_pdl` 参数不识别（删除）、o-proj einsum 回退、topk_softplus_sqrt 落参考实现、rocminfo PATH、router GEMM 回退。
- **第四阶段（Error 19–22 → Patch 16–19）**：稀疏注意力全链路——indexer forward 放行 OOT + MQA logits 回退；`bind_kv_cache` OOT 放行；`_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 纯 PyTorch 逐字节复刻（q per-head RMSNorm→GPT-J RoPE；KV RoPE→bf16 舍入→7×64 块 UE8M0 FP8 量化→V4 块布局写入）；`_flashmla_C` 的 prefill/decode 全 PyTorch 回退。
- **第五阶段（Error 23–24 + 超时）**：首请求 Triton 现场编译超时→`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600`；flag_gems 对空张量布尔/整数索引崩溃→clamp+masked_fill 与 numel==0 守卫。

### 4.4 启动与验证

```bash
# 文件放置（包内 flagos_succ1/）：
cp start_flagos_fl.sh /home/user/ ; cp patch_vllm_moe.py vllm_linear_init_patched.py /tmp/
mkdir -p /home/user/tilelang_kernels && cp tilelang_kernels/*.py /home/user/tilelang_kernels/
# 启动（entrypoint 内自动：装依赖→打 19 补丁→起服务；全流程约 20–25 分钟）
bash /home/user/start_flagos_fl.sh
docker logs -f flagos_fl_deepseek 2>&1 | grep -E "KV cache size|Available routes|Worker failed"
# 验证
curl http://localhost:8001/v1/models
curl http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
```

**结果**：端到端正确（返回连贯中文，usage 正常）；7 项验收变体脚本 `test_dv4_8001.sh`。FP8 线性内核实测误差 ~1.5%（fp8 正常水平）；Patch 18 量化路径与 CUDA 内核逐字节同语义。

### 4.5 性能现状与翻盘路径

decode ≈ 0.1 tok/s，瓶颈排序：① Patch 19 decode 注意力逐 token Python 循环 + 全量反量化（Triton 化预计提升 1–2 个数量级）；② Patch 16 MQA logits 逐 head 循环；③ Patch 13/18 回退可合并/Triton 化；④ MoE EMULATION 每次前向在线反量化（可常驻 BF16 专家权重，显存换速度）。首请求慢（Triton 按 shape 现场编译）可挂载持久化 `~/.triton` 缓存缓解。

---

## 5. 测试脚本（全文）

### 4.1 七项验收 `test_dv4.sh`（两线通用，端口 8000）

```bash
#!/bin/bash
# DeepSeek-V4-Flash 服务验收脚本  用法: bash test_dv4.sh [host]
HOST="${1:-127.0.0.1}"; BASE="http://${HOST}:8000"; M="deepseek-v4-flash"

echo "== 1. 健康检查 =="
curl -s -o /dev/null -w "GET /health -> HTTP %{http_code}\n" ${BASE}/health

echo "== 2. 模型列表 =="
curl -s ${BASE}/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin); print('models:', [m['id'] for m in d['data']])"

echo "== 3. 基础对话 =="
curl -s ${BASE}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$M"'", "messages": [{"role":"user","content":"你好，请用一句话介绍你自己。"}], "max_tokens": 128
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print('usage:', d['usage'])"

echo "== 4. 数学正确性（量化精度抽查）=="
curl -s ${BASE}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$M"'", "messages": [{"role":"user","content":"计算 37 * 89 - 156，只给出最终数字。"}],
  "max_tokens": 512, "temperature": 0
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('回答:', d['choices'][0]['message']['content'].strip()); print('期望包含: 3137')"

echo "== 5. 代码能力 =="
curl -s ${BASE}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$M"'", "messages": [{"role":"user","content":"用Python写一个判断素数的函数，只给代码。"}],
  "max_tokens": 256, "temperature": 0
}' | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

echo "== 6. 流式输出 =="
curl -sN ${BASE}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$M"'", "messages": [{"role":"user","content":"从1数到10，用顿号分隔。"}],
  "max_tokens": 64, "stream": true }' | head -20

echo "== 7. 解码速度粗测 =="
python3 - <<PYEOF
import json, time, urllib.request
req = urllib.request.Request("${BASE}/v1/chat/completions",
    data=json.dumps({"model":"$M","messages":[{"role":"user","content":"写一段200字左右介绍人工智能发展史的文字。"}],
                     "max_tokens":256,"temperature":0.7}).encode(),
    headers={"Content-Type":"application/json"})
t0=time.time(); resp=json.load(urllib.request.urlopen(req,timeout=600)); dt=time.time()-t0
u=resp["usage"]
print(f"completion_tokens={u['completion_tokens']} 耗时={dt:.1f}s 解码约 {u['completion_tokens']/dt:.1f} tok/s（单请求）")
PYEOF
```

### 4.2 并发压测 `bench_concurrency.py`（用法 `python3 bench_concurrency.py <并发> <max_tokens>`）

```python
import json, time, urllib.request, concurrent.futures, sys

BASE = "http://127.0.0.1:8000/v1/chat/completions"
M = "deepseek-v4-flash"
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 128

PROMPTS = [
    "写一段150字介绍人工智能发展史的文字。",
    "解释一下什么是注意力机制，用通俗的语言。",
    "用Python写一个快速排序函数。",
    "简述量子计算与经典计算的区别。",
    "写一首关于秋天的五言绝句并解释。",
    "什么是数据库索引？举例说明。",
    "介绍一下太阳系的八大行星。",
    "解释TCP三次握手的过程。",
]

def one(i):
    req = urllib.request.Request(BASE, data=json.dumps({
        "model": M, "messages": [{"role":"user","content":PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": MAXTOK, "temperature": 0.7}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    return r["usage"]["completion_tokens"], time.time() - t0

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
    results = list(ex.map(one, range(CONC)))
wall = time.time() - t0
total = sum(c for c, _ in results)
print(f"concurrency={CONC} max_tokens={MAXTOK}")
print(f"total completion tokens={total}  wall={wall:.1f}s")
print(f"aggregate throughput={total/wall:.1f} tok/s   per-request avg={total/CONC/wall:.2f} tok/s")
```

### 4.3 TTFT 测量（流式首包，`/data1/ttft_test.py` 的等价整洁版）

```python
import json, time, urllib.request

def ttft(content, mt=32):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps({"model": "m", "messages": [{"role": "user", "content": content}],
                         "max_tokens": mt, "stream": True}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.startswith(b"data:") and b"content" in line:
                return time.time() - t0

print("short TTFT= %.2f s" % ttft("你好"))
long_text = "人工智能的发展历史可以追溯到20世纪50年代，图灵提出了著名的图灵测试。" * 150
print("long(~5K tok) TTFT= %.2f s" % ttft(long_text + "请用一句话总结上文。"))
# 连跑两次：第二次 long 应大幅下降（前缀缓存命中，日志出现 #cached-token > 0）
```

---

## 6. 用到的全部软件包与资源清单

### 6.1 Docker 镜像（按用途）

| 镜像 | 用途 | 结论 |
|---|---|---|
| `<internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706:latest` | vLLM 线运行时 + 权重转换/量化环境 | ✅ 生产（回退线） |
| `harbor.sourcefind.cn:5443/dcu/admin/base/custom:dsv4-flash-k100ai-sglang0.5.12-20260728` | sglang 线运行时 | ✅ 生产（主线） |
| `harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-...-20260804-0006-deepseekV4-0811` | sglang 新版尝试 | ❌ BW 卡专属（tilelang 无 gfx928 GEMM、deepgemm 无 gfx928），已排除 |
| `harbor.baai.ac.cn/flagrelease-public/flagrelease-hygon-deepseek-v4-flash:202604242039` / `harbor.baai.ac.cn/flagos21-release/vllm-plugin-fl:v0.2.0-rc2-hygon` | FlagOS 线（归档）+ 本项目当 root 文件工具容器用 | 端到端通但 0.1 tok/s，归档 |

### 6.2 sglang 0728 镜像内关键包（pip list 摘录，运行时依赖全景）

| 包 | 版本 | 说明 |
|---|---|---|
| sglang | 0.5.12+g9a3a3e5f9 | 主框架 |
| **sglang-dsv4-flash-triton-patch** | **0.4.7** | 海光 DSv4 K100-AI 专用补丁包（已安装激活，25 文件；CLI：`sglang-dsv4-flash-patch status/install/verify-moe/verify-attention/write-launchers`） |
| sglang-kernel | 0.4.2.post2 | |
| sglang-router | 0.3.2+das.dtk2604 | |
| triton | 3.6.0+dtk2604.torch2100 | DAS 版 |
| tokenspeed-triton | 3.7.10.post20260531 | |
| torch | 2.10.0+das（dtk2604） | |
| flash-attn | 2.8.3+das.opt1.dtk2604 | |
| flash-mla | 1.2.0+das.opt1.dtk2604 | 含 gfx928 code objects |
| lightop | 0.6.0+das.dtk2604（build 2607051235） | topK/moe_align/moe_sum 算子 |
| sgl-deep-gemm / deep_gemm | 0.1.0 | ⚠️ CUDA 编译版（链接 libcudart.so.13），国产平台不可用，必须 env 关闭 |

### 6.3 vLLM 0706 镜像内关键包

vLLM 0.21.0；vllm_hcu 平台插件；DAS torch 2.10 / triton / tilelang（PDL 不支持 gfx928）/ lightop 0.6.0（build 0616）；MoE 调优 JSON `E=256,N=256,device_name=K100_AI,dtype=int8_w8a16.json`（8/10 镜像已内置，单流从 3.6–4 提到 9.1 tok/s 的来源）。

### 6.4 服务器上的资产总表（nodeA）

| 路径 | 内容 |
|---|---|
| `/data1/models/DeepSeek-V4-Flash-orig` | 原始权重 149G（重建源头，必留） |
| `/data1/models/dsv4-flash-w8a8` | **两线共用生产权重** 279G（含 .bak_sglang config 备份） |
| `/data1/models/dsv4-flash-bf16` | 中间产物 543G（可删，可重建） |
| `/data1/models/dsv4-flash-fp8-mp8` | mp8 探测线权重 164G（留作金标准素材） |
| `/data1/start_sglang_dsv4_prod.sh` | **sglang 生产一键启动** |
| `/data1/start_dsv4_tuned.sh` | vLLM 线生产启动（回退用） |
| `/data1/sglang_patches/` | 加载器补丁 + 官方启动器 7 个（含我方 2 处修改，.bak/.bak2 备份） |
| `/data1/*_patched.py` ×6 + `/data1/patches/*.diff` ×6 | vLLM 线补丁 |
| `/data1/test_dv4.sh`、`/data1/bench_concurrency.py`、`/data1/ttft_test.py` | 测试三件套 |
| `/data1/TUNING_RECORD_20260811.md` | 调优与试错全记录（第 1–11 轮+稳态） |
| `/data1/start_dsv4_mtp_experimental.sh`、`/data1/mtp_vmfault_repro.sh` | vLLM MTP 实验归档 + VMFault 最小复现 |
| `/data1/das_tilelang_pkg/` | DAS tilelang 0.1.9 提取包（230MB，含 tvm_ffi/z3） |
| `/home/user/dsv4_full_kit/dsv4_full_kit_20260811.tar.gz` | 全路线打包交付件 |
| `/home/user/flagos.succ1/flagos_succ1.tar.gz` | FlagOS 线交付件（19 补丁+文档） |

### 6.5 外部资源与 API

- ModelScope：`deepseek-ai/DeepSeek-V4-Flash`（原始权重）
- GitHub：`flagos-ai/DeepSeek-V4-FlagOS`（convert_weight.py）
- harbor 镜像枚举 API（匿名）：`GET https://harbor.sourcefind.cn:5443/api/v2.0/projects/dcu/repositories/admin%252Fbase%252Fcustom/artifacts?with_tag=true`
- 光合社区下载站 API：`GET https://download.sourcefind.cn:65024/api-static/file/ListFile?CategoryID=4&Path=/...`，文件下载 `GET /file/4/<path>`（DAS1.8 各包实测均为 gfx936/938 专属，对 K100-AI 无用——已逐包验证过 tilelang 0.1.6.post2、triton 3.5.1、lightop 0.7.0、deepgemm 2.1.0）

---

## 7. 其他失败/探测路线（简述与放弃原因）

### 7.1 mp8 离线线——权重就绪，栈缺口，转为金标准素材

原始权重 → fp8-mp8 转换成功（10 分钟出 164GB；修复 `convert.py` 对 FP8 checkpoint `.scale` 张量的切分 assert，修复版 `/data1/convert_fp8mp8.py`）。但 FlagOS 离线栈 `model.py` 的 fp8 分支是死代码（tilelang import 被注释、分支函数无定义，官方只验证过 BF16-mp16 双节点）。**放弃原因**：离线不是服务目标；补 fp8 逐层反量化（几十行）留作未来做精度金标准时再做。

### 7.2 tilelang FP8 路线——硬件级死路（重要负面结论）

- DAS tilelang 0.1.9 的 `fp8_gemm` 在 gfx928 编译失败：`MmacTraits` 无 fp8 特化 —— **gfx928 MMAC 矩阵核心不支持 fp8 输入**；
- tilelang `act_quant` 的 fp8 cast 内核产出 ~6.6% NaN（scale 正确，cast 错误）——**gfx928 无 fp8 转换硬件指令**，软件模拟有缺陷；
- 新版 tilelang（0811 镜像内）HCU GEMM 路径干脆只支持 MLS 架构（gfx938/92a/946），gfx928 连 bf16 GEMM 都编译不过。
**结论**：K100-AI 上任何 FP8 计算只能走 Triton 软转换或反量化到 BF16/INT8，这是所有路线选型的底层依据。

### 7.3 vLLM 线 MTP——见 2.5 第 3 条（8 并发 VMFault，sglang 线已替代实现同能力）

---

## 8. 对海光的 Bug 修正要求（详见独立报告）

完整报告独立成文：**《K100AI-DeepSeek-V4-Flash-Bug报告-提交海光.md》**（随本文档一同交付），按 DTK/HIP 运行时（A 类）、vLLM-0706 镜像（B 类）、sglang 镜像（C 类）、编译器与算子库生态（D 类）四类共 **17 项**逐条给出现象、证据、根因分析、复现方式与我方临时规避方案，并附复现材料清单（E 类）。

最高优先级速览：

| 编号 | 级别 | 问题 | 一句话 |
|---|---|---|---|
| A-1 | P0 | 多线程 H2D 拷贝死锁 | sglang 权重加载假死 8 小时的根因，py-spy 实锤 |
| A-2 | P0 | vLLM 线 MTP ≥8 并发 VMFault | 复现脚本随附；sglang 线 MTP 正常，指向 vllm_hcu 实现 |
| B-1 | P0 | `_C` 融合算子写坏 FP8 KV 缓存 | 815/2560 NaN → 全模型乱码 |
| C-1 | P0 | 0728 镜像 deep_gemm 为 CUDA 编译版 | dlopen 必败且默认路径踩雷 |
| B-3 | P1 | wo_a 转置 view 静默乱码 | 无报错的正确性事故，最难查的一类 |
| C-3 | P1 | int8 KV 损伤 DSv4 Think 质量 | 长推理复读循环，bf16 正常 |

## 9. 新机器最短复现清单

**共用（一次性，约半天）**：第 1.2 节下载 → 第 1.3 节三步权重加工（orig→bf16→w8a8→ignore 规则）。

**sglang 生产线（推荐）**：
1. 拉 0728 k100ai 镜像；
2. 容器内 `sglang-dsv4-flash-patch write-launchers --output-dir /data1/sglang_patches/launchers` 提取官方启动器，按 3.3.3 打 2 处修改；
3. 按 3.3.1 生成加载器同步补丁；
4. `bash /data1/start_sglang_dsv4_prod.sh`，约 6 分钟 health 200；
5. `bash /data1/test_dv4.sh` + `python3 /data1/bench_concurrency.py 8 100`。预期：单流 ~18.7 tok/s，8 并发 ~59 tok/s，Think/Tool Call 可用。

**vLLM 回退线**：
1. 拉 hy3-0706 镜像；解包 6 个 `*_patched.py`；
2. `bash /data1/start_dsv4_tuned.sh`；
3. 验收同上。预期：单流 ~9.3 tok/s，128 并发 ~116 tok/s。

**FlagOS 精度参考线**：
1. 拉 vllm-plugin-fl 镜像；按 4.4 节放置补丁、启动脚本与 tilelang wheel；
2. `bash /home/user/start_flagos_fl.sh`（entrypoint 自动装依赖+打 19 补丁+起服务，约 20–25 分钟）；
3. 8001 端口验证。预期：输出正确连贯，decode ~0.1 tok/s（仅精度对照用）。

---

*文档生成：2026-08-12。全部数据来自 nodeA 实测；调优过程逐轮记录见 `/data1/TUNING_RECORD_20260811.md`。*
