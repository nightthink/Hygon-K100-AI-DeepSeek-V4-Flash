#!/bin/bash
# 对照实验：0811 镜像 + 0731 权重（已剔除 DSpark 张量）+ 不开投机
#
# 目的：判断"非贪心高并发 GPU 硬件异常"是 DSpark 特有，还是 0811 镜像在 gfx928 上的通病。
# 实测结论（2026-08-14）：本配置下 temperature=0.7 × 8 并发 **8/8 成功、聚合 52.2 tok/s**，
# 即同镜像、同权重、同 triton 注意力后端下非贪心并发完全稳定 —— 问题归因于 DSpark。
#
# 副产品：本配置也可作为"必须支持采样 + 高并发"场景的备用生产配置
# （代价是失去投机加速，单流约 12.3 tok/s）。
#
# 依赖：/data1/sglang_patches/launchers_0811/run_0811_probe.sh 及三个补丁。
docker rm -f sglang-dsv4 sglang-0731base sglang-0811probe 2>/dev/null
sleep 3
docker run -d --name sglang-0811probe \
  --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 \
  -v /data1/models/dsv4-0731-w8a8:/models \
  -e NCCL_P2P_DISABLE=1 -e PORT=8000 \
  -e SPEC_ALGO=none \
  -e MEM_FRACTION_STATIC=0.85 -e CUDA_GRAPH_MAX_BS=16 -e PREFILL_CHUNK=4096 \
  -w /data1/sglang_patches/launchers_0811 \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  run_0811_probe.sh
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | head -2
echo "对照实验：0811 + 无投机，就绪后测 temp=0.7 × 8 并发"
