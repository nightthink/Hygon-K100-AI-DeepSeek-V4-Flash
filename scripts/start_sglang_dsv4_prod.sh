#!/bin/bash
# DeepSeek-V4-Flash 生产启动脚本（sglang 线，2026-08-12 镜像固化版 v2）
# 镜像：lzd/dsv4-flash-k100ai-sglang:0728-patched-v2
#   = 官方 dsv4-flash-k100ai-sglang0.5.12-20260728
#   + 同步加载补丁已固化（防多线程 H2D 死锁，无需再挂载）
#   + 官方启动器（含双解析器/前缀缓存修改）固化于 /opt/dsv4/launchers/
# 性能：单流 18.3-18.7 tok/s（MTP EAGLE steps=3）/ 8 并发 59.4 tok/s / 前缀缓存命中 TTFT 0.8s@5K
# 能力：Think（请求加 chat_template_kwargs {"thinking":true}）/ Tool Call / 1M ctx（KV 池 324K, bf16）
# 可调：-e SPECULATIVE_NUM_STEPS=N（默认3）、-e CUDA_GRAPH_MAX_BS=N（默认64）、-e PORT=N，
#       或把末行脚本换成 /opt/dsv4/launchers/ 下其他变体（nomtp / torch_native / int8KV 等）
# 注意：int8 KV 已否决（Think 复读）；steps=4 已否决（8 并发 -11%）。
# 构建上下文：/home/user/image_bake/sglang/（Dockerfile 可复现）
set -e
docker rm -f sglang-dsv4 2>/dev/null || true
docker run -d --name sglang-dsv4 --restart unless-stopped --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 \
  -v /data1/models/dsv4-flash-w8a8:/models \
  -e NCCL_P2P_DISABLE=1 -e PORT=8000 \
  -e MEM_FRACTION_STATIC=0.85 -e CUDA_GRAPH_MAX_BS=16 -e PREFILL_CHUNK=4096 \
  -w /opt/dsv4/launchers \
  --entrypoint bash lzd/dsv4-flash-k100ai-sglang:0728-patched-v2 run_ds_mtp_triton_logic_bf16_kv.sh
echo "服务启动中，约 6 分钟就绪：curl http://127.0.0.1:8000/health"
