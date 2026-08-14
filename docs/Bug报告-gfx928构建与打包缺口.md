# 提交材料（二）：K100-AI（gfx928）上阻断 4-bit 与专家并行的构建 / 打包缺口

> 目标仓库：`HYGON-AI/sglang-das`（分支 `v0.5.15.post1_dev`）
> 报告方环境：8×海光 K100-AI（gfx928）/ DTK 26.04 / DeepSeek-V4-Flash-0731
> 使用镜像：`custom:sglang0.5.12-...-20260804-0006-deepseekV4-0811`
> 日期：2026-08-14
> 配套材料：`Bug报告-DSpark与triton路由.md`（问题一至三，均为代码层问题）

---

## 摘要

前一份报告提交的是三个**代码层**问题。本报告提交四个性质不同的问题：
**功能代码都写好了，但没能在 gfx928 上用起来**——或是编译目标漏了本卡，或是依赖库没随镜像发布。

这四项共同构成我方当前性能天花板的主因。最关键的量化依据是：

| | 目录大小 | 每参数（284B 计） |
|---|---|---|
| 原始 `DeepSeek-V4-Flash-0731` | **156 GB** | ≈ 4.4 bit |
| 我方 W8A8 | **292 GB** | ≈ 8.2 bit |

模型的路由专家原生就是 4-bit（抽样 safetensors 头部：专家张量占 **90.3%** 字节）。
我方实测确认解码瓶颈是**每步从 HBM 搬运的专家权重字节数**（详见前份报告附三），
因此我方目前是**以模型原生设计约 1.9 倍的权重流量在运行**。

这不是"希望增加一项优化"，而是**希望能按模型本来的格式运行**。

---

## 问题四【P1】DeepEP / RocSHMEM 的设备代码未包含 gfx928

### 现象

启用专家并行后，在 MoE 首次 dispatch 时整组 rank abort（exit code −6）：

```
Error: hipGetSymbolAddress(reinterpret_cast<void**>(&print_lock_addr),
       HIP_SYMBOL(print_lock)): invalid kernel file (218)
       at RocSHMEM::/builds/dcutoolkit/deeplearing/DeepEP/third-party/rocshmem/...
```

调用栈：`token_dispatcher/deepep.py → dispatch` ← `ep_moe/layer.py → forward_deepep`
← `models/deepseek_v4.py`。

### 复现

```bash
--speculative-algorithm DSPARK --ep-size 8 \
--moe-a2a-backend deepep --deepep-mode low_latency
```

### 根因（已定位）

```bash
$ strings .../deep_ep/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so | grep -oE 'gfx[0-9]{3,4}' | sort -u
gfx936
gfx938
```

本卡为 `gfx928:sramecc+:xnack-`。`invalid kernel file (218)` 是 HIP 的
"胖二进制中无匹配当前架构的设备代码"。失败发生在第一次符号查找（`print_lock`）上，
说明**整个 RocSHMEM 设备侧对本卡不可用**，而非个别内核缺失。

对照：`sgl_kernel` 编译目标为 `gfx906 gfx926 gfx928 gfx936 gfx938`，覆盖本卡。

### 诉求

把 gfx928 加入 DeepEP / RocSHMEM 的 offload-arch 列表。**功能代码与 ROCm 移植均已完成，
只差为本卡出一份构建产物。**

### 收益

TP=8 下每卡需读取全部 256 个专家权重的 1/8（等效 32 个专家）；
EP=8 下每卡只持有 32 个专家且每步只读被路由到的少数几个。
按我方负载（每步 6 token × topk 6 = 36 行）估算，**每步 MoE 权重读取量可降到约 1/5**。

### 附带观察

`deepep + low_latency` 模式下，验证图的 cuda graph 捕获批量从
`[1,2,3,4,5,6,7,8,10,12,14,16]` 缩为 `[4,8,12,16]`——**bs=1 未被捕获**。
即便架构问题解决，单流解码仍会回退 eager。建议一并检查。

另：该模式启动时预分配约 38 GB 缓冲（`Load weight begin` 时可用显存从 62.69 GB 降至 24.58 GB），
在 64 GB 卡上需要相应下调 `--mem-fraction-static`。

---

## 问题五【P1】EP 朴素 dispatcher 在 gfx928 上非法地址访问

### 现象

绕开 DeepEP、使用朴素 dispatcher 时走得更远（权重加载正常、cuda graph 捕获列表恢复完整），
但在图捕获阶段段错误（exit code −11）：

```
Invalid address access: 0x7825611c0000, Error code: 1
  at sglang/srt/layers/moe/fused_moe_triton/layer.py:1350 forward_impl
```

### 复现

```bash
--speculative-algorithm DSPARK --ep-size 8 --moe-a2a-backend none
```

### 说明

这是独立于问题四的 gfx928 正确性缺陷——EP 分片下 fused_moe_triton 前向存在非法地址访问。
与问题四互为备选路径：任一修复即可让我方用上专家并行。

---

## 问题六【P0】原生 4-bit 专家在 gfx928 上无可用后端

这是本报告优先级最高的一项，包含两个独立缺口。

### 6.1 低比特算子族未编进 ROCm 构建

`sgl_kernel` 的 Python 包装里可以看到这些函数：

```
cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data,
qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm, gptq_gemm, awq_dequantize
```

但直接取 torch 算子命名空间（惰性 `__getattr__`，`dir()` 不枚举算子，**必须直接取属性**）：

| 算子 | 是否注册 |
|---|---|
| `cutlass_w4a8_moe_mm` | ❌ 缺 |
| `get_cutlass_w4a8_moe_mm_data` | ❌ 缺 |
| `qserve_w4a8_per_chn_gemm` | ❌ 缺 |
| `qserve_w4a8_per_group_gemm` | ❌ 缺 |
| `gptq_gemm` | ❌ 缺 |
| `awq_dequantize` | ❌ 缺 |
| `moe_align_block_size`（对照组，生产在用） | ✅ 在 |

对照组存在，证明探针方法有效。**`--quantization` 的取值列表接受 `w4afp8`/`gptq`/`awq` 等，
但本卡上没有对应内核。**

（附注：`cutlass_w4a8_moe_mm` 的 docstring 明写 "leverages NVIDIA Hopper architecture
features"，ROCm 侧需要的是重新实现而非重新编译。）

### 6.2 `humming` 后端：集成代码已发布，依赖库未发布

sglang 会自动识别本模型的专家布局：

```
Auto-detected DSV4 routed-expert layout: is_fp4_experts=True
```

`layers/quantization/fp8.py` 中的后端映射：

| `--moe-runner-backend` | fp4 专家路径 | gfx928 可用性 |
|---|---|---|
| `auto`（默认） | **反量化 fp4 → fp8**，走 `Fp8MoEMethod` | 能跑，但等于 8-bit |
| `marlin` | `Mxfp4MarlinMoEMethod` | ❌ CUDA 专属 |
| **`humming`** | **`Mxfp4HummingMoEMethod`** | ⚠️ 见下 |
| `flashinfer_mxfp4` | SM90 / SM100 分支 | ❌ CUDA 专属 |

sglang 侧集成完整存在于镜像中：

```
/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/quantization/mxfp4_humming_moe.py  (3178 B, 2026-08-11)
/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/quantization/humming_utils.py      (5177 B, 2026-08-11)
```

但 `humming_utils.py` 第 5 行：

```python
from humming.layer import HummingInputSchema, HummingMethod
```

而该包在镜像中不存在：

```python
>>> importlib.util.find_spec('humming')   # None
>>> importlib.util.find_spec('lightop')   # 存在（对照）
```

**结论：gfx928 上唯一可能承载原生 mxfp4 专家的后端，其实现库没有随镜像发布。**

### 诉求（按性价比排序）

1. **随镜像发布 `humming` 包，并为 gfx928 构建** —— 集成代码已完成，
   对贵方可能只是打包清单遗漏；对我方是权重流量直接减半、且**无需重新量化**。
2. 低比特算子族补进 ROCm 构建（`w4a8` 需 ROCm 侧实现，非单纯重编）。

### 我方已验证的自助替代路径（供参考）

`moe_wna16` + AWQ int4 走 Triton（`use_int4_w4a16`）。该路径**已能在 gfx928 上编译执行**
（配置查找打出 `dtype=int4_w4a16`，无异常），且 `moe_wna16.py` 的 AWQ 分支写有
`if not _is_hcu:` 才做能力检查——即贵方已为 HCU 放行该路径。

我方尚未完成数值验证（合成张量构造 AWQ 打包布局失败，需走真实加载器）。
即便该路径可用，它也需要我方重新量化，且 `a16` 意味着激活为 bf16 而非当前的 int8；
相比之下问题 6.2 的修复对双方成本都更低。

---

## 附：问题二的定位补充（DSpark 采样崩溃）

前份报告的问题二列出六项无效缓解措施。本次用一个判别实验把范围显著收窄。

**探针设计**：`temperature=0.7` + `top_k=1` 在数学上等价于贪心（候选集只有一个 token），
但**走的是完整的采样代码路径**——`do_sample` 分支、接受判定、残差缓冲全部经过，
唯独不会发生"拒绝后重采样"。

| 配置 | 8 并发结果 | accept rate |
|---|---|---|
| `temperature=0`（基线） | ✅ 8/8 | 0.60–0.77 |
| **`temperature=0.7` + `top_k=1`** | ✅ **8/8 通过**（聚合 48.9 tok/s） | 0.60–0.77 |
| `temperature=0.7` + `top_k=2` | ❌ `HSA_STATUS_ERROR_EXCEPTION 0x1016` | **0.25** |
| `temperature=0.7`（全词表） | ❌ detokenizer 卡死，2/8 超时 | 0.26–0.27 |

**结论：采样代码路径本身与随机性本身均非诱因，触发条件是「拒绝后重采样」分支被密集执行。**
`top_k` 从 1 到 2 就是分界线；accept rate 同步从 0.6+ 崩到 0.25（七成以上草稿 token 被拒绝）。

这个定位解释了前份报告中一个说不通的疑点：`SGLANG_DSPARK_FORCE_TORCH_ACCEPT=1` 无效，
是因为该开关覆盖的是**接受判定**内核，而非**拒绝后重采样**分支——回退换掉的不是出问题的代码。

**另请注意故障表现不唯一**：`top_k=2` 复现经典的 HSA 硬件异常队列 abort；
全词表采样则表现为进程全部存活（8 个 scheduler + detokenizer 均在 `Sl` 睡眠态）、
detokenizer 不再响应、健康检查报
`Server couldn't get a response from detokenizer for last 20 seconds`。
两者是同一缺陷的两种表现，建议不要当作独立问题处理。

**补充否决**：我方另试了放宽接受阈值（`--speculative-accept-threshold-acc 0.1`
`--speculative-accept-threshold-single 0.05`）以减少拒绝事件，无效——
且采样下的 accept rate 仍然是 0.27，纹丝不动，说明**这两个阈值参数不作用于采样接受路径**。
（顺带确认：阈值放到极激进值对贪心性能无影响，33.15 vs 33.2 tok/s。）

**最小复现**：`--speculative-algorithm DSPARK`、`temperature=0.7`、**`top_k=2`**、并发 8。
对照组 `top_k=1` 不应复现。判据：`accept rate` 从 0.6+ 掉到 0.25 时故障出现。
复现脚本见我方仓库 `tests/probe_sampling_crash.py`。

---

## 附：诉求汇总（两份报告合并，按性价比排序）

| # | 诉求 | 出处 | 我方收益 |
|---|---|---|---|
| 1 | 发布 `humming` 包并为 gfx928 构建 | 本报告 6.2 | 原生 4-bit，**零重新量化**，权重流量减半 |
| 2 | DeepEP / RocSHMEM 加 gfx928 编译目标 | 本报告 四 | 每步权重读取量约降至 1/5 |
| 3 | 修 DSpark 拒绝重采样分支 | 前报告 二 + 本报告附 | 解锁 `temperature>0` |
| 4 | 低比特算子族补进 ROCm 构建 | 本报告 6.1 | 4-bit 备选路 |
| 5 | 修 EP 朴素 dispatcher 非法地址访问 | 本报告 五 | 专家并行备选路 |
| 6 | KV dtype 与解码后端的兼容性校验 | 前报告 三 | 消除静默乱码风险 |
| 7 | 接通 HCU 平台的 triton 解码路径 | 前报告 一 | 已自行打补丁，建议上游合入 |

第 1 条成本最低、收益最直接。

---

## 验证支持

我方有 8×K100-AI 环境与完整的验收 / 压测 / 长上下文脚本，
另有 `tests/probe_arch.sh`（几十秒核对各库的 gfx 编译目标与算子注册情况），
修复版镜像可在 1 天内完成回归对比。
