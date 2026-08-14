# AWQ int4 路径验证工装

目的：判定 sglang 的 **Triton `use_int4_w4a16`** MoE 路径在 gfx928 上是否可用。
若可用，则可把 DSV4 的专家权重降到 4-bit，直接砍掉一半权重流量——
而权重流量正是轮次 26 定位的解码瓶颈（详见 `docs/调优记录-轮次31.md`、`轮次32.md`）。

## 结论：被一道与厂商自身补丁相矛盾的 ROCm 闸门拦住

```
ValueError: moe_wna16 quantization is currently not supported in ROCm.
  at sglang/srt/configs/model_config.py:1463  _verify_quantization
```

**这是本次验证的最终结果。** 它的矛盾之处在于：

- `layers/quantization/moe_wna16.py` 的 AWQ 分支写有 `if not _is_hcu:` 才做能力检查——
  即海光**专门为自家硬件放行了这条路**（本机 `is_hcu()` 返回 True）
- 但 `configs/model_config.py::_verify_quantization` 在更上游**直接拒绝 ROCm 上的 moe_wna16**

也就是说，**那段 HCU 放行代码永远执行不到**。二者必有一处需要修正。

对海光的诉求（可并入 `docs/Bug报告-gfx928构建与打包缺口.md` 第 6 节）：
既然已经为 HCU 放行了 `moe_wna16 + awq` 的能力检查，请一并放开
`_verify_quantization` 里的 ROCm 拦截（或改为按 `is_hcu()` 判定），
否则 Triton int4 这条不依赖任何预编译低比特算子的路径无法启用。

## 附带发现：bf16 基线自己就 VM fault

在拿到上面的结论之前，我们先跑**未量化的 bf16 参考模型**建立基线，结果它也崩了：

```
Invalid address access: 0x7a599b1ec000, Error code: 1.
（Triton 内核阶段，UnquantizedFusedMoEMethod）
```

- 与量化**无关**（这一版没有任何量化）
- 与维度无关：hidden 256/head_dim 64 与 hidden 1024/head_dim 128 两组均复现
- 生产用的 DSV4（同样走 fused_moe_triton，但走 CompressedTensorsW8A8Int8MoE）稳定运行 33 tok/s

**这个签名与轮次 28 的 EP 实验完全一致**（`fused_moe_triton/layer.py:1350 forward_impl`
非法地址访问）。提示 fused_moe_triton 在某些配置组合下于 gfx928 存在共性缺陷，
值得单列排查——但那是独立于"int4 能不能用"的另一个问题。

## 已完成且可复用的部分

| 脚本 | 作用 | 状态 |
|---|---|---|
| `build_tiny_moe.py` | 用 transformers 造 2 层 Qwen2MoE（301.5M 参数），复用 DSV4 tokenizer | ✅ |
| `pack_awq.py` | 专家权重量化为 AWQ int4，并**用 sglang 自己的还原逻辑做 CPU 往返校验** | ✅ **48/48 层往返一致** |
| `run_infer.py` | 离线 Engine 跑推理，贪心解码 | ✅ 可用（模型侧受上述两个问题阻断） |

### 打包已被独立验证，可直接用于全量 DSV4

`pack_awq.py` 逐字复刻 `moe_wna16.py::convert_awq_tensor` 的还原步骤，把打包结果
还原回整数权重与量化前逐元素比对：

```
量化专家线性层: 48 个
往返校验: ✅ 全部一致
量化最大相对误差: 0.0605      # 随机权重 + 无校准 RTN，属正常
```

产出形状（hidden=1024, moe_intermediate=512, group=128）：

```
qweight (1024, 64)  int32     # (in, out/8)
qzeros  (8, 64)     int32     # (in/group, out/8)
scales  (8, 512)    bfloat16  # (in/group, out)
```

**这是本次工作中最有价值的可复用产出**：一旦厂商放开上述闸门，
这套打包逻辑可直接套用到 DSV4 的全量专家权重，无需再赌 AWQ 的 nibble 排列。

## 沿途踩到并已固化的坑

### 1. 非 MoE 层必须走 `modules_to_not_convert`

`MoeWNA16Config.get_quant_method` 对普通 `LinearBase` 会路由到 `AWQConfig`，
需要 `awq_dequantize` / `awq_gemm` 等算子——而这些**没编进本镜像的 ROCm 构建**（轮次 28/32）。
因此注意力、shared_expert、router、lm_head 全部列入 `modules_to_not_convert` 保持 bf16。

### 2. `is_layer_skipped_quant` 是纯子串匹配

```python
return any(module_name in prefix for module_name in modules_to_not_convert)
```

必须写 `"mlp.gate"` 而不是 `"gate"`——后者会误伤专家的 `gate_proj`，
把本该量化的权重也跳过，整个实验失去意义。

### 3. transformers 5.6 的三个构造陷阱

- 向 `Qwen2MoeConfig(...)` 传参会提前触发 rope 标准化，报
  `'PreTrainedConfig' object has no attribute 'max_position_embeddings'`
  → 先默认构造、再逐项赋值
- `layer_types` 默认 24 项，与 `num_hidden_layers=2` 冲突
  → 显式设为 `["full_attention"] * num_hidden_layers`
- `AutoTokenizer.from_pretrained(DSV4目录)` 会去解析 stock transformers 不认识的
  `deepseek_v4` model_type，同样炸在 rope 上
  → 只拷贝 tokenizer 文件，`vocab_size` 直接从 config.json 读

### 4. 本镜像的分页 KV 约束

离线 Engine 起 Qwen2MoE 时依次撞到：

```
Error: 4-D KV buffer's dim-1 must equal page_size; got shape[1]=2, page_size=1
Error: Paged KV cache block size must be 64
```

→ 必须显式 `page_size=64`，且不要强制 `attention_backend="triton"`。

### 5. `sgl.Engine` 必须放在 `__main__` 保护里

否则报 "An attempt has been made to start a new process before the current process
has finished its bootstrapping phase"。

## ⚠️ 运行须知

- **必须先停掉推理服务**：本工装会触发 VM fault，而 VM fault 会把同一批 GPU 上
  正在服务的进程一并带崩（轮次 32 已因此多重启一次）。
- 模型与中间产物写在 `/home/cl/awq_probe/`（`/data1` 对普通用户不可写）。

## 下一步

1. **（推荐）** 把上述 `_verify_quantization` 的 ROCm 拦截作为诉求提给海光——
   它与厂商自己的 HCU 放行代码直接矛盾，属于明确的一致性缺陷
2. 或本地打补丁绕过该拦截，验证 Triton int4 内核的真实数值表现
   （但需先解决 bf16 基线的 VM fault，否则无对照基准）
3. 单列排查 `fused_moe_triton` 在 gfx928 上的非法地址访问（与轮次 28 EP 同签名）
