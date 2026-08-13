# DeepSeek-V4-Flash vLLM 服务启动手册（8×K100-AI）

> **镜像**：`lzd/dsv4-flash-k100ai-vllm:0706-patched-v1`（镜像 ID `f42b725280f8`）
> 基于海光 `vllm-ubuntu22.04-dtk26.04-hy3-0706`（vLLM 0.21.0 + vllm_hcu 平台插件），已固化 6 个 gfx928 正确性补丁（mhc PDL 移除 / `_C` 融合算子写坏 KV 的绕行 / indexer FP8 写读统一 / wo_a 转置 view 修复 / 两处日志修复）。**无需再挂载任何补丁文件。**
> **权重**：`/data1/models/dsv4-flash-w8a8`（279GB，专家 INT8 W8A8-per-channel + 注意力 BF16）
> 文档版本：2026-08-12

---

## 1. 主机前置条件

| 项 | 要求 | 快速自检 |
|---|---|---|
| GPU | 8×K100-AI（gfx928），DTK 26.04 驱动就绪 | `ls /dev/kfd` 存在；`rocm-smi` 能列出 8 卡 |
| hyhal | 宿主 `/opt/hyhal` 存在（容器必须挂载，否则 torch 报 `librocm_smi64.so.2` 缺失） | `ls /opt/hyhal` |
| 权重 | `/data1/models/dsv4-flash-w8a8` 完整 | `ls /data1/models/dsv4-flash-w8a8/*.safetensors \| wc -l` → 46 |
| docker | 当前用户在 docker 组 | `docker ps` 不报权限错 |
| 显卡占用 | 8 卡空闲（本服务独占 8 卡） | 停掉其他占卡容器 |
| 端口 | 8000（或自选）未被占用 | `ss -tlnp \| grep 8000` |

首次启动建议预热页缓存（把加载从 ~10 分钟缩到 ~4 分钟）：

```bash
cat /data1/models/dsv4-flash-w8a8/*.safetensors > /dev/null &
```

## 2. 启动命令（调优定稿配置）

```bash
docker rm -f vllm-dsv4 2>/dev/null || true
docker run -d --name vllm-dsv4 --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --restart unless-stopped \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 \
  -e NCCL_P2P_DISABLE=1 \
  lzd/dsv4-flash-k100ai-vllm:0706-patched-v1 \
  vllm serve /data1/models/dsv4-flash-w8a8 \
    --tensor-parallel-size 8 --kv-cache-dtype fp8 --block-size 256 \
    --enforce-eager --disable-custom-all-reduce \
    --disable-hybrid-kv-cache-manager \
    --served-model-name deepseek-v4-flash \
    --max-model-len 262144 --max-num-seqs 128 \
    --gpu-memory-utilization 0.92 --port 8000
```

**就绪判断**：日志出现 `Application startup complete`（页缓存热约 4 分钟，冷盘约 10 分钟），`curl http://127.0.0.1:8000/health` 返回 200。跟踪日志：`docker logs -f vllm-dsv4`。

## 3. 参数说明

### 3.1 红线参数（每条都有实测依据，勿动）

| 参数 | 原因 |
|---|---|
| `--kv-cache-dtype fp8` | DeepseekV4 硬性要求 fp8_ds_mla 缓存格式，不给直接报错 |
| `--block-size 256` | DeepseekV4Indexer/FlashMLASparse 后端只声明支持 256（默认 16 报 "No common block size"） |
| `--enforce-eager` | CUDA graph 实测负优化（32 并发吞吐 −14%） |
| `--disable-custom-all-reduce` + `-e NCCL_P2P_DISABLE=1` | 不加则权重加载完成后进程挂死（GPU 0%、无日志） |
| `--disable-hybrid-kv-cache-manager` | 混合 KV 管理器 page 断言在本模型多 spec 组下不成立 |

### 3.2 可调参数

| 参数 | 定稿值 | 调整建议 |
|---|---|---|
| `--max-model-len` | 262144 | 业务上下文短则调小，KV 显存转给并发容量 |
| `--max-num-seqs` | 128 | 实测最优（32→128 使高并发吞吐 +66%；256 无增益徒增延迟） |
| `--gpu-memory-utilization` | 0.92 | 显存吃紧可降 |
| `--port` | 8000 | |

### 3.3 禁用项

**不要启用任何投机解码/MTP 参数**（`--speculative-config` 等）：该栈在 ≥8 并发时触发 GPU VMFault 崩溃（已向海光报障，待修复版镜像）。

## 4. 客户端调用（OpenAI 兼容）

```bash
curl http://<host>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role":"user","content":"你好"}],
  "max_tokens": 128}'
```

流式（`"stream": true`）正常支持。`/v1/models` 显示 `deepseek-v4-flash`。

## 5. 预期指标（8×K100-AI 实测）

| 指标 | 值 |
|---|---|
| 单流解码 | ~9.3 tok/s |
| 8 并发聚合 | ~23 tok/s |
| 32 / 64 并发聚合 | ~70 / ~77 tok/s |
| **128 并发聚合** | **~116 tok/s**（饱和点，高并发吞吐是本配置的强项） |
| 上下文 | 262,144 token（启动参数可调） |

## 6. 验收与压测

```bash
bash test_dv4.sh                        # 7 项验收：health/models/对话/数学3137/素数/流式/速度
python3 bench_concurrency.py 128 128    # 高并发压测，预期 ~116 tok/s
```

（脚本见交付打包件 tests/ 目录。）验收判据：数学题回答含 3137（t=0）；素数函数为标准 6k±1 实现；输出无乱码。**若输出乱码/NaN**：确认镜像确为 `patched-v1`（补丁自检：`docker run --rm --entrypoint grep lzd/dsv4-flash-k100ai-vllm:0706-patched-v1 -c "patched: no PDL" /usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/mhc.py` 应输出 5）。

## 7. 常见问题速查

| 现象 | 原因与处置 |
|---|---|
| torch 导入报 `librocm_smi64.so.2` | 忘挂 `-v /opt/hyhal:/opt/hyhal` |
| 加载完成后挂死、GPU 0%、无日志 | 少了 `--disable-custom-all-reduce` 或 `-e NCCL_P2P_DISABLE=1` |
| 报 "No common block size" | `--block-size` 不是 256 |
| 启动即报 fp8_ds_mla 相关错误 | 少了 `--kv-cache-dtype fp8` |
| 全模型输出乱码 | 用了未打补丁的原版镜像（wo_a 转置 view 静默乱码是最典型元凶） |
| 并发时 GPU VMFault | 启用了投机解码——按 3.3 节禁用 |

## 8. 镜像血缘与重建

基础镜像 `<internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706:latest`；构建上下文（Dockerfile + 6 个补丁文件）在 `/home/user/image_bake/vllm/`。重建：

```bash
cd /home/user/image_bake/vllm && docker build -t lzd/dsv4-flash-k100ai-vllm:0706-patched-v1 .
```

跨机复制：`docker save -o x.tar lzd/dsv4-flash-k100ai-vllm:0706-patched-v1` → `scp` 到目标机 → `docker load -i x.tar`（建议 md5sum 校验）。补丁的 unified diff（6 个共 344 行）与根因分析见交付打包件 patches/ 目录及部署文档。

---

Copyright © 2026 DaoTech Team. Licensed under the MIT License.
