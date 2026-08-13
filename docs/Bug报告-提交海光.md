# 海光 K100-AI（gfx928）DeepSeek-V4-Flash 适配 Bug 报告

> **报告方环境**：nodeA —— 2×48 核 CPU / 512GB 内存 / 8×海光 K100-AI 64GB（gfx928）/ DTK 26.04 / 内核 6.8.0-136-generic
> **背景**：我方在该机上将 DeepSeek-V4-Flash（284B MoE，激活 13B）通过三条路线（海光 vLLM 线、海光 sglang 线、FlagOS vLLM 线）全部部署为 OpenAI 兼容服务，过程中发现并定位以下问题。每项均附现象、证据、根因分析、复现方式与我方临时规避方案，供贵方修复参考。
> **涉及镜像**：
> ① `<internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706:latest`（下称 vLLM-0706 镜像）
> ② `harbor.sourcefind.cn:5443/dcu/admin/base/custom:dsv4-flash-k100ai-sglang0.5.12-20260728`（下称 sglang-0728 镜像）
> ③ `harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811`（下称 sglang-0811 镜像）
> ④ `harbor.baai.ac.cn/flagos21-release/vllm-plugin-fl:v0.2.0-rc2-hygon`（FlagOS FL 镜像，涉及 flagtree hcu 后端时与贵方相关）

---

## A 类：DTK/HIP 运行时问题（跨框架，最高优先）

### A-1【P0】多线程 H2D 拷贝死锁（sglang 权重加载假死 8 小时）

- **现象**：sglang-0811/0728 镜像加载 DeepSeek-V4 权重时，8 个 TP rank 全部在打印 `Execute dequant fp8 wo_a` 后永久静默；服务 health 一直 000；进程不退出不报错。首次遭遇时假死超过 8 小时。
- **证据（py-spy dump，间隔 20 秒两次采样，帧完全一致，判定真死锁而非慢）**：
  - 主线程：`concurrent.futures as_completed → threading.wait`（等待 futures，位于 `deepseek_v4.py load_weights`）；
  - ThreadPoolExecutor 工作线程：冻结于 `sglang/srt/layers/moe/fused_moe_triton/layer.py:584` 的 `expert_data.copy_(loaded_weight)`（Host→Device 拷贝）及 `linear.py:497 weight_loader`；
  - 系统侧：CPU ~0%（top 94% idle）、磁盘 0 I/O（iostat）、8 卡 VRAM 均衡停在 37–42%、功耗 100–130W、`dmesg`/`journalctl -k` 无 VMFault 无报错。
- **根因分析**：sglang `model_loader/utils.py::should_async_load()` 对 CPU 张量返回 True，权重经 ThreadPoolExecutor **多线程并发**执行 `tensor.copy_()` H2D。该并发模式在 DTK 26.04 HIP 运行时上死锁（推测为 HIP 流/staging buffer 的线程安全问题）。
- **复现**：任一 sglang DSv4 镜像 + 默认参数启动 TP8 加载即可复现（概率接近 100%，我方连续 2 次全中）。
- **建议修复**：HIP 运行时修复多线程 H2D 拷贝的线程安全；短期可在 sglang HCU 平台分支将 `should_async_load` 默认返回 False。
- **我方临时规避**：挂载覆盖 `model_loader/utils.py`，`should_async_load` 恒返 False（强制同步加载）。实测无性能痛点：页缓存热时 46 分片 7–11 秒读完，整体加载 178 秒。

### A-2【P0】vLLM 线 MTP 投机解码 ≥8 并发 GPU VMFault

- **现象**：vLLM-0706 镜像 + W8A8 权重 + EAGLE/MTP 投机解码（`num_speculative_tokens` 取 1 或 2 均复现）：单流与低并发正常（单流 12.5 tok/s，+34%），**并发数达到 8 时 GPU VMFault，引擎崩溃**。
- **证据**：dmesg 出现 VMFault 记录；bench 脚本 8 路并发稳定触发；spec=1 与 spec=2 两种配置均崩，排除 draft 深度因素。
- **对照**：**sglang-0728 镜像上同一模型 MTP（EAGLE steps=3）8 并发完全稳定**（本报告方生产配置），说明是 vllm_hcu 栈的实现问题而非硬件/驱动天花板。
- **复现**：`/data1/mtp_vmfault_repro.sh`（随附最小复现脚本：启动 MTP 配置 + 8 路并发请求）。
- **影响**：vLLM 线单流 +34% 的收益不可用。
- **我方临时规避**：vLLM 线回退非 MTP 配置；MTP 需求转 sglang 线满足。

### A-3【P1】custom allreduce / P2P 在本机 PCIe 拓扑下加载后挂死

- **现象**：vLLM-0706 镜像不加 `--disable-custom-all-reduce` + `NCCL_P2P_DISABLE=1` 时，权重加载完成后进程挂死：GPU 0%、无日志推进。
- **旁证**：sglang 官方启动器同样内置 `USE_DCU_CUSTOM_ALLREDUCE=0` + `FORCE_TORCH_AR=1`，说明贵方已知 custom allreduce 在部分拓扑不可靠。
- **建议**：custom allreduce 初始化时做拓扑探测失败即自动回退 torch allreduce，而不是挂死。

### A-4【P1】非官方 env 组合下 triton 内核加载段错误（应报错而非 SIGSEGV）

- **现象**：sglang-0728 镜像在**不带**官方启动器环境变量全集（`GPU_MAX_HW_QUEUES=3`、`HIP_KERNEL_BATCH_CEILING=100`、`SGLANG_USE_LIGHTOP=1` 等约 20 项）时，首次 forward 即 `Fatal Python error: Segmentation fault`，栈位于 `triton/compiler/compiler.py:474 _init_handles`（内核模块加载），伴随 HIP 运行时 `HOSTQUEUE <0x...>: device id / base_address` 输出；CUDA graph 捕获（bs=256、bs=16）与 `--disable-cuda-graph` 纯 eager 均崩。
- **根因分析**：缺省 env 下某个资源路径（疑与 HW queue 数量或 lightop 路由相关）越界，SIGSEGV 于 hipModuleLoad 阶段。
- **建议**：关键运行前置条件缺失时给出明确报错/自动设置默认值，而不是段错误；并请在镜像 README 中把必需 env 显式文档化（目前只能从 `write-launchers` 生成的脚本反推）。

---

## B 类：vLLM-0706 镜像（vllm_hcu 栈）正确性问题

### B-1【P0】`_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 在 gfx928 写坏 KV 缓存

- **现象**：该 C++ 融合算子写出的 SWA FP8 缓存字节 + scale，无法被 upstream triton 反量化内核 `_dequantize_and_gather_k_kernel` 正确解读：收集出的 KV 2560 项中 815 项 NaN → 第 0 层注意力全 NaN → 全模型输出乱码。
- **隐蔽性**：启动时的 profile run 不做真实注意力，问题被掩盖到首个真实请求才爆发。
- **定位方法**（供贵方复核）：逐层插桩 absmean/nan 统计定位到第 0 层 attention；dump 缓存字节与 upstream triton 写入端逐字节比对确认布局不符。
- **建议修复**：统一该算子与 upstream triton 反量化内核的 FP8 布局（数据区/块尾 scale 区的字节序与 scale 编码）。
- **我方临时规避**：保留算子调用（写入无害），q 侧 python 重实现 per-head RMSNorm + GPT-J RoPE（与 FlagOS 参考逐位对齐 cos=0.999996），KV 侧 python RoPE 后用 upstream triton `quantize_and_insert_k_cache` 重写缓存条目。

### B-2【P0/P1】indexer 内核写读格式矛盾（写 FP8 读 bf16）

- **现象**：贵方把 Lightning Indexer 的插入/收集内核换成 bf16 版本，但 V4 的 indexer 缓存实际由 compressor 融合内核以 **FP8 布局**写入（indexer 创建时 `skip_k_cache_insert=True`，贵方修改的插入路径根本不执行）。启动即触发 `k_cache 必须是 bf16 类型` 断言。
- **附带问题**：bf16 收集分支把**未初始化的 `k_scale`** 传给 lightop 内核。
- **建议修复**：收集分支恢复 upstream FP8 内核 `cp_gather_indexer_k_quant_cache_triton`（带 k_scale 与 token_to_seq），或让插入路径真正生效并全链路 bf16。
- **我方临时规避**：改回 upstream FP8 收集内核，`fp8_dtype` 用 `current_platform.fp8_dtype()`，q 保持 FP8（lightop `mqa_logits` 支持 FP8×FP8）。

### B-3【P1】`rocm_inv_rope_einsum` 的 `wo_a.weight.view(...)` 与可插拔 Linear 转置布局冲突（静默乱码）

- **现象**：`wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim)` 假设权重为原生布局；运行时贵方可插拔 Linear 实际存成转置布局 `[4096, 1024]`。对转置矩阵直接 view 是**静默内存重解释**——o 投影输出完全乱序但无任何报错。
- **证据**：逐层 cos 对账——q/kv/qr/mHC/注意力核心全部 >0.999，经 o 投影后骤降至 **0.017**；dump 运行时权重与 checkpoint 各切分假设比对实锤转置存储。
- **同类问题**：`fused_wkv_wgate.weight.T`（compressor 输入投影）——`mat1 and mat2 shapes cannot be multiplied`。
- **建议修复**：凡绕过 `Linear.forward` 直接访问 `.weight` 的代码，一律按实际 shape 自适应布局；建议贵方对可插拔 Linear 的转置存储行为做全仓审计。
- **我方临时规避**：按 `weight.shape[0]` 判断布局，转置时走 `view(hidden, groups, o_lora).permute(1,2,0)`。

### B-4【P1】`mhc.py` PDL 调用移除不完整（编译期崩溃）

- **现象**：gfx928 TileLang 不支持 PDL（Programmatic Dependent Launch），贵方补丁漏删 5 处（4× `T.pdl_sync()` + 1× `T.pdl_trigger()`），编译 mhc 内核报 "PDL is not supported"，启动即崩。
- **我方临时规避**：5 处全部替换为 `pass`。离线数值比对确认 mHC sinkhorn 输出与 FlagOS 参考一致（cos≈1.0）。

### B-5【P2】vllm 与 vllm_hcu 双份同名文件导致修复只落一半

- **现象**：`rocm_aiter_mla_sparse.py` 在 `vllm/v1/attention/ops/` 与 `vllm_hcu/v1/attention/ops/` 各有一份，且 `deepseek_v4_attention.py` 实际 import 的是 **vllm 那份**（与"平台插件优先"的直觉相反）。我方补丁一度只打了 vllm_hcu 份而不生效。
- **建议**：平台差异代码单点维护（vllm_hcu 份 re-export），或至少在两份文件头注明真实 import 关系。

### B-6【P2】小问题两处

- `kv_cache_utils.py`：启动日志的 `max_concurrency` 计算除零。
- `worker/utils.py`："No common block size" 报错信息不含各 backend 声明的 block size 列表，排障困难（DeepseekV4IndexerBackend/FlashMLASparse 实际要求 256，默认 16 必失败——建议模型侧自动设置或报错时列出）。

---

## C 类：sglang 镜像问题

### C-1【P0】sglang-0728 镜像内 deep_gemm 为 CUDA 编译版（dlopen 必败）

- **现象**：`/usr/local/lib/python3.10/dist-packages/deep_gemm/_C.so` 链接 `libcudart.so.13`，在海光平台 dlopen 报 `libcudart.so.13: cannot open shared object file`。而 `SGLANG_OPT_DEEPGEMM_HC_PRENORM` 默认为 True，使 mhc_pre 默认路径 `import deep_gemm` 直接崩溃（CUDA graph 捕获期报 `Capture cuda graph failed: Failed to load dynamic shared library`）。
- **建议修复**：镜像内置 HIP/DTK 编译版 deep_gemm，或在 HCU 平台把 `SGLANG_OPT_DEEPGEMM_HC_PRENORM` 默认置 0（官方启动器已这么做，但裸启动的用户必踩）。

### C-2【P1】sglang-0811 镜像对 K100-AI 完全不可用且无架构标注

- **现象**：tag `sglang0.5.12-...-deepseekV4-0811` 无任何架构标注。实测其 tilelang 的 HCU GEMM 路径仅支持 MLS 架构：`HCU arch gfx928 not supported for MLS/GEMM_MLS; supported: gfx938, gfx92a, gfx946`（mhc_pre tilelang splitk 内核编译失败）；其 deepgemm 亦只含 gfx92a/936/938 code objects。即该镜像在 K100-AI 上 mhc 两条路径全死。
- **建议**：镜像 tag 或 README 标明支持的 gfx 架构（0728 的 `dsv4-flash-k100ai-*` 命名是好实践）；tilelang 报错信息保持现状即可（已很明确）。

### C-3【P1】量化 KV cache（int8 与 fp8 均复现）在 DSv4 上损伤 Think（长推理）质量，且解码变慢

- **现象一（int8）**：sglang-0728 官方默认配置（`SGLANG_DSV4_INT8_KV_CACHE=true`）下，thinking 模式长推理出现**复读循环**："……实际上，9.9更大。实际上，9.9等于9.90……实际上……"重复直至耗尽 max_tokens，`content` 为空。bf16 KV 同题推理干净（101 reasoning tokens 一次到位）。
- **现象二（fp8，2026-08-12 补测）**：改用纯 fp8（`KV_CACHE_DTYPE=fp8_e4m3` + `SGLANG_DSV4_INT8_KV_CACHE=false`，其余生产配置不变）**同样复读**："但问题可能是在问哪个数字更大，所以答案是9.9更大。"无限重复至 max_tokens；另见 `</think>` 泄漏进 content、同请求下 Tool Call 不再触发。即问题不特定于 int8 打包，**凡量化 KV 均触发**，指向 gfx928 上 KV 反量化路径的数值/精度问题。
- **性能复核**：量化 KV 不仅无速度收益反而更慢——fp8 单流 16.0–16.3 tok/s vs bf16 18.7–18.8（**-14%**），反量化开销超过带宽节省；容量收益也仅 +17%（702,720 vs 600,832 tokens，DSA 压缩 KV 本身很小，量化只作用于部分缓存）。
- **建议**：在 K100-AI 上复测 int8/fp8 KV 的长推理精度与反量化 kernel 性能；修复前文档标注"reasoning 场景必须 bf16 KV"。
- **我方选择**：生产采用 bf16 KV 变体（`run_ds_mtp_triton_logic_bf16_kv.sh`）。

### C-4【P2】官方启动器引用镜像内不存在的 chat template 文件

- **现象**：`write-launchers` 生成的 `run_ds_mtp.sh` 含 `--chat-template ./tool_chat_template_deepseekv3.jinja`，但该文件既不在镜像内也不随 launchers 生成，照抄脚本必失败。
- **旁注**：DeepSeek-V4 模型目录以 `encoding/encoding_dsv4.py`（代码式编码规范）取代 jinja 模板；实测删掉该参数后 sglang 内置模板工作正常，Think/Tool Call 经 `--reasoning-parser deepseek-v4` / `--tool-call-parser deepseekv4` 均正确。
- **建议**：launchers 里删掉该参数或随包提供 jinja 文件；并考虑把双解析器加入官方启动器默认参数（DSv4 的 Think/ToolCall 是核心卖点，当前官方脚本不带解析器，OpenAI API 用户拿不到 reasoning_content / tool_calls 结构化输出）。

### C-5【P2】`--disable-radix-cache` 作为官方默认值的疑问

- **现象**：官方启动器默认禁用 radix cache（前缀缓存）。我方删除该参数后：正确性验收全过、稳态运行无异常，**同前缀 5K prompt 第二次 TTFT 从 8.25s 降到 0.82s（10×）**，多轮对话体验差异巨大。
- **建议**：若禁用是因已知 bug，请在脚本注释标明触发条件（我方需评估风险）；若只是保守默认，建议放开或至少文档说明代价。

### C-6【P2】CUDA graph 大 batch 捕获段错误（次要，官方 env 下不触发）

- **现象**：非官方 env 下 bs=256 捕获段错误（avail_mem 仅 7.5GB 时）。归并入 A-4 的"应报错不应段错误"诉求；另建议对捕获期 OOM 风险给出显式检查。

---

## D 类：编译器与算子库生态（经 FlagOS 线发现，涉及 hcu Triton 后端）

以下问题在 FlagOS FL 镜像（flagtree 0.5.0+hcu）上发现。flagtree 由 FlagOS 维护，但 **hcu 后端**与贵方生态直接相关，建议协同修复：

### D-1【P1】hcu Triton 编译器：if/else 分支合并不同基指针/不同 constexpr 时 PassManager 崩溃

- **现象**：fused_qk_rmsnorm Triton 内核编译报 `PassManager::run failed`（make_ttgir 阶段）。二分定位：hcu 后端无法处理 if/else 两臂各自使用**不同基指针或不同 constexpr** 再汇合的模式（pointer phi）。
- **我方规避**：把 RMSNorm body 抽成 `@triton.jit` 内联辅助函数、在每个分支臂内各自展开——修复版 bit-exact。此改写模式可作为贵方修复前的官方 workaround 模板。

### D-2【P2】hcu Triton 不识别 `launch_pdl` 启动参数（KeyError）

- NVIDIA 专属参数；建议 hcu 后端对未知 launch 参数忽略并告警，而非 KeyError。

### D-3【硬件结论确认请求】gfx928 FP8 支持边界

我方实测三项硬结论，请贵方确认并写入公开文档（可为社区节约大量试错）：
1. **MMAC 矩阵核心无 fp8 输入特化**（DAS tilelang 0.1.9 `fp8_gemm` 编译失败于 `MmacTraits` 无 fp8 特化）；
2. **无 fp8 转换（cast）硬件指令**（tilelang `act_quant` fp8 cast 内核产出 ~6.6% NaN，scale 计算正确、cast 本身错误——软件模拟缺陷）；
3. 新版 tilelang（0811 镜像内 build）HCU GEMM 路径已完全放弃 gfx928（仅 MLS 架构）。
即 K100-AI 上任何 FP8 计算只能走 Triton 软件转换或先反量化为 BF16/INT8——这决定了 DSv4 类 FP8 原生模型在该卡的全部可行路线。

### D-4【P2】DAS1.8 系列包全部仅含 gfx936/938 目标

- 光合社区下载站的 DAS1.8 包（tilelang 0.1.6.post2、triton 3.5.1、lightop 0.7.0、deepgemm 2.1.0，各 dtk2604 build）经逐包解包验证均不含 gfx928 code objects。建议下载页标注适用架构，避免 K100-AI 用户下载后无法使用（我方为验证花费了完整一轮排查）。

---

---

## F 类：性能诉求（量化，2026-08-12 补充）

### F-1【诉求】编程助手场景单流解码 50 tok/s

- **业务背景**：本机服务编程助手（平均输入 23K token、最长 98K、输出 ≤8K、8-10 路并发）。单流解码速度直接决定用户感观，目标 50 tok/s。
- **现状与差距分解**（应用层调优已穷尽，18.7 tok/s 为当前栈天花板）：
  - 基础解码 12.1 tok/s × MTP 链式（EAGLE steps=3）1.55× = 18.7 tok/s；
  - 到 50 需基础解码 ~32 tok/s，即 **2.7× 内核层提升**；
  - 对照证据：同机 Qwen3.6-27B dense BF16 单流 24 tok/s——DSv4 激活参数（13B，专家还是 INT8）只有其一半却更慢，说明 DSA 稀疏注意力/indexer/mHC 的 decode 开销占大头。
- **建议优化点（按预期收益排序）**：
  1. **NSA/DSA decode 内核**（当前 triton_logic 补丁后端）：稀疏注意力 + indexer 的 decode 路径深度优化或专用汇编内核；
  2. **custom allreduce 修复**（本报告 A-3）：现被迫 `USE_DCU_CUSTOM_ALLREDUCE=0` + torch AR 走 PCIe，43 层 × 每层多次同步的延迟在 decode 关键路径上；
  3. **DSv4 树形投机支持**：当前 sglang 断言 `Only EAGLE ... topk == 1 is supported for DeepseekV4ForCausalLM`，放开 topk>1 预计再乘 10-20%；
  4. prefill 侧同样受 NSA 内核限制（实测 343-395 tok/s，23K prompt TTFT ~45s、98K ~4.1min）——长 prompt TTFT 对编程场景同样关键。
- **验证承诺**：我方验收/压测/长上下文脚本齐备（含 23K/98K 实测基线），修复版镜像可在 1 天内完成回归对比。

---

## E 类：随附复现材料清单

| 材料 | 位置（nodeA） | 对应问题 |
|---|---|---|
| MTP VMFault 最小复现 | `/data1/mtp_vmfault_repro.sh` | A-2 |
| MTP 实验配置归档 | `/data1/start_dsv4_mtp_experimental.sh` | A-2 |
| 加载死锁临时补丁（对照可复现原问题） | `/data1/sglang_patches/model_loader_utils_patched_0728.py` | A-1 |
| vLLM 线 6 补丁 diff（每个对应一处 bug 的修复） | `/data1/patches/01~06-*.diff` | B-1…B-6 |
| int8 KV Think 复读实录 | `/data1/TUNING_RECORD_20260811.md` 第 11 轮 | C-3 |
| fp8 KV Think 复读 + 单流 -14% 实录 | `/data1/TUNING_RECORD_20260811.md` 第 20 轮 | C-3 |
| FlagOS 线 19 补丁与 24 错误全记录 | `/home/user/flagos.succ1/`（DEPLOYMENT.md） | D-1、D-2 |
| 三条路线完整文档 | 《在8张K100-AI上运行DeepSeek-V4-Flash：从零到成果》 | 全部 |

> 联系与验证：以上所有问题在 nodeA 上均可复现或已留存日志/记录；欢迎提供修复版镜像，我方可在 1 天内完成回归验证（验收脚本与压测脚本齐备）。

---

Copyright © 2026 DaoTech Team. Licensed under the MIT License.
