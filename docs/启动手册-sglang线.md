# DeepSeek-V4-Flash sglang 服务启动手册（8×K100-AI）

> **镜像**：`lzd/dsv4-flash-k100ai-sglang:0728-patched-v2`
> 基于海光官方 `dsv4-flash-k100ai-sglang0.5.12-20260728`，已固化：①权重同步加载补丁（防多线程 H2D 拷贝死锁）；②**参数化版**官方启动器 7 个（含 Think/Tool Call 双解析器、前缀缓存、MEM_FRACTION_STATIC/PREFILL_CHUNK/NSA_FUSE_TOPK/SCHED_POLICY 参数）置于镜像内 `/opt/dsv4/launchers/`。**无需再挂载任何补丁文件；未设默认入口，启动方式与参数完全由使用者掌控。**
> **权重**：`/data1/models/dsv4-flash-w8a8`（279GB，专家 INT8 W8A8-per-channel + 注意力 BF16，config.json 已含 sglang 所需的 compressed-tensors ignore 规则）
> 文档版本：2026-08-12

---

## 1. 主机前置条件

| 项 | 要求 | 快速自检 |
|---|---|---|
| GPU | 8×K100-AI（gfx928），DTK 26.04 驱动就绪 | `ls /dev/kfd` 存在；`rocm-smi` 能列出 8 卡 |
| hyhal | 宿主 `/opt/hyhal` 存在（容器必须挂载，否则 torch 报 `librocm_smi64.so.2` 缺失） | `ls /opt/hyhal` |
| 权重 | `/data1/models/dsv4-flash-w8a8` 完整 | `ls /data1/models/dsv4-flash-w8a8/*.safetensors \| wc -l` → 46 |
| docker | 当前用户在 docker 组 | `docker ps` 不报权限错（新加组需重新登录，临时可用 `sg docker -c '...'`） |
| 显卡占用 | 8 卡空闲（本服务独占 8 卡） | 停掉其他占卡容器 |
| 端口 | 8000（或自选）未被占用 | `ss -tlnp \| grep 8000` |

首次启动建议预热页缓存（把加载从 10+ 分钟缩到约 6 分钟）：

```bash
cat /data1/models/dsv4-flash-w8a8/*.safetensors > /dev/null &
```

## 2. 启动命令（生产推荐配置）

```bash
docker rm -f sglang-dsv4 2>/dev/null || true
docker run -d --name sglang-dsv4 --restart unless-stopped \
  --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 \
  -v /data1/models/dsv4-flash-w8a8:/models \
  -e NCCL_P2P_DISABLE=1 -e PORT=8000 \
  -e MEM_FRACTION_STATIC=0.85 -e CUDA_GRAPH_MAX_BS=16 -e PREFILL_CHUNK=4096 \
  -w /opt/dsv4/launchers \
  --entrypoint bash lzd/dsv4-flash-k100ai-sglang:0728-patched-v2 \
  run_ds_mtp_triton_logic_bf16_kv.sh
```

要点：权重必须挂到容器内 `/models`（启动器内固定该路径）；v2 镜像内 `/opt/dsv4/launchers` 已是参数化版启动器，全部参数经 `-e` 生效；`--ulimit memlock=-1` 对应启动器内的 `ulimit -l unlimited`。

**就绪判断**：约 6 分钟（页缓存热）后 `curl http://127.0.0.1:8000/health` 返回 200。加载期间无响应属正常（279GB 权重 + 8 rank 初始化 + CUDA graph 捕获），跟踪日志：`docker logs -f sglang-dsv4`。

## 3. 可调参数（-e 环境变量传入，无需改镜像）

| 环境变量 | 默认 | 调优定稿值 | 作用与依据 |
|---|---|---|---|
| `PORT` | 8000 | 8000 | 服务端口 |
| `MEM_FRACTION_STATIC` | 0.725 | **0.85** | 静态显存占比。0.85 → KV 池 600,832 token（+85%）；0.90 账面 711K 但长 prompt prefill OOM，禁用 |
| `PREFILL_CHUNK` | 8192 | **4096** | prefill 分块。8192 在 0.85 下长 prompt 峰值 4GB OOM；4096 是"大 KV 池 + 98K 长 prompt 稳定"的关键 |
| `CUDA_GRAPH_MAX_BS` | 64 | **16** | 匹配 8–10 路并发上限，省下的捕获显存转给 KV 池 |
| `SPECULATIVE_NUM_STEPS` | 3 | 3 | 实测 3 最优；4 时单流仅 +2% 而 8 并发 −11% |
| `NSA_FUSE_TOPK` | false | false | 实测无收益（23K prefill 335 vs 343 tok/s），保持官方默认 |
| `SCHED_POLICY` | fcfs | fcfs | 调度策略。`lpm` 为待验证优化项，见第 10 节 |
| `ENABLE_MTP` | 由所选脚本决定 | true | 直接跑 `run_ds_mtp.sh` 时可设 false 关闭投机解码 |

## 4. 启动变体（镜像内 /opt/dsv4/launchers/ 现成 7 个）

把启动命令末尾的脚本名替换即可：

| 脚本 | 配置 | 何时用 |
|---|---|---|
| `run_ds_mtp_triton_logic_bf16_kv.sh` | **MTP + bf16 KV（生产默认）** | 单流最优 18.7 tok/s，长推理输出干净 |
| `run_ds_nomtp_triton_logic_bf16_kv.sh` | 无 MTP + bf16 KV | 排查 MTP 相关问题时的对照组（单流 12.1） |
| `run_ds_mtp.sh` | MTP + **int8 KV**（上游默认） | ⚠️ 实测 Think 长推理出现复读循环，不建议 reasoning 场景 |
| `run_ds_mtp_triton_logic.sh` | MTP + int8 KV + triton_logic | 同上告诫 |
| `run_ds_nomtp_torch_native_bf16_kv.sh` | 无 MTP + torch native 注意力 | 最保守的排障配置 |
| `run_cilent.sh` / `run_evalscope.sh` | 客户端 / 评测工具 | 测试用 |

## 5. 客户端调用（OpenAI 兼容，含 Think / Tool Call）

```bash
# 普通对话
curl http://<host>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role":"user","content":"你好"}]}'

# Think 模式（回复中 reasoning_content 与 content 分离）
curl http://<host>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role":"user","content":"9.11 和 9.9 哪个大？"}],
  "chat_template_kwargs": {"thinking": true}}'

# Tool Call：标准 OpenAI tools 参数即可，返回 tool_calls，finish_reason=tool_calls
```

注：model 字段可任填（服务端未设 served-model-name，`/v1/models` 显示为 `/models`）。要显示正规名，可用 `-v` 挂一份加了 `--served-model-name deepseek-v4-flash` 的启动器覆盖 `/opt/dsv4/launchers/run_ds_mtp.sh`，不必重做镜像。

## 6. 预期指标（8×K100-AI 实测）

| 指标 | 值（调优定稿配置实测） |
|---|---|
| 单流解码 | 18.7 tok/s（MTP EAGLE steps=3） |
| 8 / 10 并发聚合 | ~59 / 66.4 tok/s，稳定无崩溃 |
| TTFT | 短 prompt ~0.3s；5K 首次 ~8s、前缀缓存命中 ~0.8s |
| 长 prompt 实测 | 23K 级：45s（prefill 343 tok/s）；98K 级：4.1 分钟（395 tok/s），服务全程健康 |
| KV 池容量 | **600,832 token**（bf16 KV，共享；单请求 98K+8K 富余 5 倍） |
| 上下文上限 | 模型 1M；bf16 KV 实际安全上限约 600K（更高需 int8 KV，但损伤 Think，不建议） |

> 编程负载参考（平均输入 23K / 最长 98K / 输出 ≤8K）：容量对 10 路平均负载富余近 2 倍；重 prefill 期间 /health 可能短暂无响应，属单引擎忙碌，自行恢复。

## 7. 验收与压测

```bash
bash test_dv4.sh                      # 7 项验收：health/models/对话/数学3137/素数/流式/速度
python3 bench_concurrency.py 8 100    # 8 并发压测，预期 ~59 tok/s
python3 ttft_test.py                  # TTFT；连跑两次第二次 long 应大幅下降（前缀缓存）
```

（三脚本见交付打包件 tests/ 目录。）验收判据：数学题回答含 3137（t=0）；Think 测试 reasoning 连贯无复读；素数函数为标准 6k±1 实现。

## 8. 常见问题速查

| 现象 | 原因与处置 |
|---|---|
| torch 导入报 `librocm_smi64.so.2` | 忘挂 `-v /opt/hyhal:/opt/hyhal` |
| 报 `Unable to find matching target ... wqkv_a` | 权重 config.json 缺 ignore 规则。本份权重已修；若换新权重，在其 `quantization_config.ignore` 增加 `"re:^(?!.*\\.experts\\.).*$"` |
| 加载假死、8 rank 打完 dequant 日志后静默 | 同步加载补丁未生效——确认镜像是 `patched-v2` 而非官方 0728 原版 |
| Think 输出复读循环 | 用了 int8 KV 变体——换 `*_bf16_kv.sh` |
| 首次请求明显慢 | Triton 按 shape 现场编译，属正常；预热一条请求即可 |
| health 长时间 000 | 属加载中；超过 15 分钟看 `docker logs` 是否有 Traceback |

## 9. 镜像血缘与重建

基础镜像 `harbor.sourcefind.cn:5443/dcu/admin/base/custom:dsv4-flash-k100ai-sglang0.5.12-20260728`；构建上下文（Dockerfile + 同步加载补丁 + 7 个启动器）在 `/home/user/image_bake/sglang/`。重建：

```bash
cd /home/user/image_bake/sglang && docker build -t lzd/dsv4-flash-k100ai-sglang:0728-patched-v2 .
```

跨机复制：`docker save -o x.tar lzd/dsv4-flash-k100ai-sglang:0728-patched-v2` → `scp` 到目标机 → `docker load -i x.tar`（建议 md5sum 校验）。

## 10. 待验证优化项：schedule_policy = lpm（附完整测试方法）

**是什么**：sglang 默认调度策略 `fcfs`（先来先服务）；`lpm`（Longest Prefix Match）在等待队列中**优先调度与 KV 缓存前缀匹配最长的请求**。编程助手场景下多个请求常共享大段前缀（同一系统提示词、同一代码库上下文、同一会话历史），lpm 让"能吃到缓存的请求先跑"，减少缓存被逐出后的重算，理论收益是**降低平均 TTFT、提高 cached-token 命中率**；潜在代价是排队公平性（无共享前缀的请求可能被插队）。

**开关**：启动命令加 `-e SCHED_POLICY=lpm`（v2 镜像启动器已支持）。切换需重启服务（约 6 分钟）。

**为什么普通压测测不出它**：`bench_concurrency.py` 的各路 prompt 互不相同、无共享前缀，lpm 与 fcfs 行为完全一致。有效测试必须使用**带共享前缀的并发负载**。

**方法一：合成模拟（半小时，方向性结论）**

1. 构造负载：10 个并发"用户"，每个请求 = 共享 20K token 公共前缀（模拟代码库上下文）+ 各自不同的 3K 尾部（模拟具体问题）；每用户串行发 3-5 轮。
2. 分别在 `SCHED_POLICY=fcfs` 与 `lpm` 下各跑一遍（两次重启，注意每次先发一条预热请求）。
3. 对比三个指标：平均 TTFT、P95 TTFT、服务日志中 `#cached-token` 占比（`docker logs sglang-dsv4 | grep -o "#cached-token: [0-9]*"` 汇总）。
4. 判读：差异 >15% 才算有效信号；<15% 视为噪声维持 fcfs。局限：合成前缀结构是假设的，结论只能作方向参考。

**方法二：真实流量影子测试（数天，结论可靠，推荐）**

1. 真实用户接入运行数天，保留服务日志（每条请求都记录 cached-token 与排队耗时）。
2. 选两个负载相近的使用时段（如相邻两个工作日的同时段），一个跑 fcfs、一个跑 lpm。
3. 从日志统计对比：平均/P95 TTFT、cached-token 命中率、排队等待分布；同时收集用户主观体感。
4. lpm 显著占优 → 写入生产配置（`-e SCHED_POLICY=lpm` 加进启动脚本）；否则维持 fcfs。

**判停条件**：无论哪种方法，若 lpm 下出现个别请求长时间饥饿（P99 排队时间显著上升），即使平均值占优也建议维持 fcfs——编程助手场景下"个别用户等很久"比"平均快一点"伤害更大。

---

Copyright © 2026 DaoTech Team. Licensed under the MIT License.
