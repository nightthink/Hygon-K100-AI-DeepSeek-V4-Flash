#!/bin/bash
# 第二轮实测：0811 镜像（sglang 0.5.15 + DSpark）+ 0731 权重（保留 mtp/DSpark）
# 启动器 run_0811_probe.sh 内含：同步加载补丁、triton 路由补丁、gfx928 全套 env 回退
docker rm -f sglang-dsv4 sglang-0731base sglang-0811probe 2>/dev/null
sleep 3
docker run -d --name sglang-0811probe \
  --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 \
  -v /data1/models/dsv4-0731-w8a8-dspark:/models \
  -e NCCL_P2P_DISABLE=1 -e PORT=8000 \
  -e SPEC_ALGO=${SPEC_ALGO:-dspark} \
  -e DSPARK_BLOCK=${DSPARK_BLOCK:-} \
  -e MAX_RUNNING=${MAX_RUNNING:-} \
  -e MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85} \
  -e CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-16} \
  -e PREFILL_CHUNK=${PREFILL_CHUNK:-4096} \
  -w /data1/sglang_patches/launchers_0811 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  run_0811_probe.sh
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | head -2
echo "就绪判断：curl http://127.0.0.1:8000/health（首次含 triton JIT，约 8-12 分钟）"
