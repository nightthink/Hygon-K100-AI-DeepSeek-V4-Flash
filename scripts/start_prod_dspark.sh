#!/bin/bash
# 生产启动器：0731 权重 + 0811 镜像 + DSpark 投机解码
#
# 定稿于 2026-08-14（轮次27）。单流 33.2 tok/s，8 并发 8/8、10 并发 10/10 稳定。
#
# ⚠️ 硬约束：本线只在 temperature=0（贪心）下稳定。
#    temperature>0 且并发 >=8 会触发 GPU 硬件异常 HSA_STATUS_ERROR_EXCEPTION 0x1016，
#    watchdog 超时后服务挂死，必须重启（约 14 分钟）。
#    网关层必须拦截 temperature>0 的高并发流量，或路由到无投机备用线
#    （start_0811_nospec.sh，8 并发 52.2 tok/s，任意温度稳定）。
#
# ⚠️ 行为差异：Think 需要请求显式携带 chat_template_kwargs={"thinking": true}，
#    不再默认开启。Tool Call 无需额外参数。
#
# ⚠️ 不要改 --kv-cache-dtype：本线解码内核按 fp8 布局读 KV，
#    改 bf16 会静默产出乱码且各项指标"变好"（accept rate 恒为 1.00 是告警信号）。
set -euo pipefail

IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811
MODEL=/data1/models/dsv4-0731-w8a8-dspark

docker rm -f sglang-dsv4 sglang-0731base sglang-0811probe >/dev/null 2>&1 || true
sleep 4

docker run -d --name sglang-dsv4 --restart unless-stopped \
  --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 -v "$MODEL":/models \
  -e NCCL_P2P_DISABLE=1 -e PORT=8000 -e SPEC_ALGO=dspark \
  -e MEM_FRACTION_STATIC=0.85 -e CUDA_GRAPH_MAX_BS=16 -e PREFILL_CHUNK=4096 \
  -w /data1/sglang_patches/launchers_0811 --entrypoint bash \
  "$IMAGE" run_0811_probe.sh

echo "已启动。首次就绪约 14 分钟（含 triton JIT + cuda graph 捕获）。"
echo "就绪判断：curl http://127.0.0.1:8000/health"
echo ""
echo "⚠️ 就绪后第一次生成必然偏慢（JIT 污染，实测 12.3 tok/s），"
echo "   第二次才是真实值（33.2 tok/s）。压测前务必先热身两次。"
