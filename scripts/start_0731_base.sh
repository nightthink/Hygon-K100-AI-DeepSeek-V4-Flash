#!/bin/bash
# 第一轮实测：0728-patched-v2 镜像 + 0731 新权重（无投机，因 0728 镜像不认 DSpark）
# 与生产定稿配置一致，只换权重目录，便于与 4 月版无投机基线 12.1 tok/s 对比
docker rm -f sglang-dsv4 sglang-0731base sglang-0811probe 2>/dev/null
sleep 3
docker run -d --name sglang-0731base \
  --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 \
  -v /data1/models/dsv4-0731-w8a8:/models \
  -e NCCL_P2P_DISABLE=1 -e PORT=8000 \
  -e MEM_FRACTION_STATIC=0.85 -e CUDA_GRAPH_MAX_BS=16 -e PREFILL_CHUNK=4096 \
  -e ENABLE_MTP=false \
  -w /opt/dsv4/launchers \
  --entrypoint bash lzd/dsv4-flash-k100ai-sglang:0728-patched-v2 \
  run_ds_nomtp_triton_logic_bf16_kv.sh
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | head -2
echo "就绪判断：curl http://127.0.0.1:8000/health（约 4-6 分钟）"
