# AWQ int4 路径验证工装（进行中）

目的：判定 sglang 的 **Triton `use_int4_w4a16`** MoE 路径在 gfx928 上是否数值正确。
若可用，则可把 DSV4 的专家权重降到 4-bit，直接砍掉一半权重流量——
而权重流量正是轮次 26 定位的解码瓶颈（详见 `docs/调优记录-轮次31.md`、`轮次32.md`）。

## 为什么不用 tests/probe_int4_triton.py 那条路

那个探针手搓张量直接喂内核，结果全零 / VM fault，**无法区分"内核有问题"和"我打包错了"**。
AWQ 的 nibble 有特定排列顺序，手工构造等于盲写一遍打包逻辑。

本工装换了思路：**造一个结构真实的小 MoE，走 sglang 真实加载器**，
并且在上 GPU 之前先在 CPU 上把打包验证掉。

## 三个阶段

| 脚本 | 作用 | 状态 |
|---|---|---|
| `build_tiny_moe.py` | 用 transformers 造 2 层 Qwen2MoE（70.1M 参数），复用 DSV4 tokenizer | ✅ 完成 |
| `pack_awq.py` | 专家权重量化为 AWQ int4 并**用 sglang 自己的还原逻辑做 CPU 往返校验** | ✅ **48/48 层往返一致** |
| `run_infer.py` | 分别跑 bf16 版与 AWQ 版，贪心解码对比输出 | ⏸ 未完成（连接中断） |

## 关键成果：打包已被独立验证

`pack_awq.py` 逐字复刻了 `moe_wna16.py::convert_awq_tensor` 的还原步骤，
把打包结果还原回整数权重，与量化前的整数**逐元素比对**：

```
量化专家线性层: 48 个
往返校验: ✅ 全部一致
量化最大相对误差: 0.0636      # 随机权重 + 无校准 RTN，属正常
```

产出形状（in=out=256，group=128）：

```
qweight (256, 32)  int32     # (in, out/8)
qzeros  (2, 32)    int32     # (in/group, out/8)
scales  (2, 256)   bfloat16  # (in/group, out)
```

**这意味着后续若仍失败，责任在内核侧而非打包侧**——这正是上一个探针无法给出的结论。

## 沿途踩到并已固化的坑

### 1. 非 MoE 层必须走 `modules_to_not_convert`

`MoeWNA16Config.get_quant_method` 对普通 `LinearBase` 会路由到 `AWQConfig`，
需要 `awq_dequantize` / `awq_gemm` 等算子——而这些**没编进本镜像的 ROCm 构建**（轮次 28/32）。
因此注意力、shared_expert、router、lm_head 全部列入 `modules_to_not_convert` 保持 bf16。

### 2. `is_layer_skipped_quant` 是纯子串匹配

```python
return any(module_name in prefix for module_name in modules_to_not_convert)
```

所以必须写 `"mlp.gate"` 而不是 `"gate"`——后者会误伤专家的 `gate_proj`，
把本该量化的专家权重也跳过，整个实验失去意义。

### 3. transformers 5.6 的两个构造陷阱

- 向 `Qwen2MoeConfig(...)` 传参会提前触发 rope 标准化，报
  `'PreTrainedConfig' object has no attribute 'max_position_embeddings'`
  → 改为先默认构造、再逐项赋值
- `layer_types` 默认是 24 项，与 `num_hidden_layers=2` 冲突，保存时报
  `must be equal to the number of layer types`
  → 显式设为 `["full_attention"] * num_hidden_layers`
- `AutoTokenizer.from_pretrained(DSV4目录)` 会去解析 stock transformers 不认识的
  `deepseek_v4` model_type，同样炸在 rope 上
  → 只拷贝 tokenizer 文件，vocab_size 直接从 config.json 读

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

## 未完成部分

`run_infer.py` 的 bf16 参考版已能加载（`UnquantizedFusedMoEMethod`，0.26 秒）并分配 KV，
但在补齐 `page_size=64` 后的那次运行中会话与服务器断连，未取到生成结果。

**下一步**：重跑 bf16 版取基线输出 → 跑 AWQ 版（`quantization="moe_wna16"`）→ 逐 token 对比。
判据：两版输出在贪心解码下应高度一致（小模型随机权重，允许因量化误差在若干 token 后分叉，
但不应出现乱码或全零）。

## ⚠️ 运行须知

- **必须先停掉推理服务**：本工装可能触发 VM fault，而 VM fault 会把同一批 GPU 上
  正在服务的进程一并带崩（轮次 32 已因此多重启一次）。
- 模型与中间产物写在 `/home/cl/awq_probe/`（`/data1` 对普通用户不可写）。
