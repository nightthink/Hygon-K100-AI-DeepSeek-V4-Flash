#!/bin/bash
# 容器化 MoE 调优入口：按批量列表自动选用对应的裁剪搜索空间。
#
#   bash run_tune.sh "1 2 4 6 8 16"                       -> space_small.json（解码档，约 3.5 分钟）
#   bash run_tune.sh "32 64 128 256 512 1024 2048 4096"   -> space_large.json（预填档，约 9.5 分钟）
#
# 注意：
#   - 独占全部 GPU，需先停掉推理服务。
#   - 调优器每轮会用"本轮 batch 列表"整体覆盖同名 json，两轮之间要备份，最后用
#     merge_configs.py 合并：
#       bash run_tune.sh "1 2 4 6 8 16"
#       cp "E=256,...json" decode_configs_backup.json
#       bash run_tune.sh "32 64 ... 4096"
#       python3 merge_configs.py
#   - 首次使用前先跑 patch_tuner.py（per-channel 量化会让调优器崩溃）。
set -x
BS=${1:-"1 2 4 6 8 16"}
D=$(cd "$(dirname "$0")" && pwd)

# 批量最大值 <= 16 视为解码档
MAXBS=$(echo $BS | tr ' ' '\n' | sort -n | tail -1)
if [ "$MAXBS" -le 16 ]; then
  SPACE=space_small.json
else
  SPACE=space_large.json
fi

docker rm -f moe-tune 2>/dev/null
sleep 2
docker run --rm --name moe-tune \
  --network=host --ipc=host --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal -v /data1:/data1 -v "$D":/tune \
  -e SGLANG_USE_LIGHTOP=1 -e SGLANG_ROCM_USE_AITER_MOE=false \
  -w /tune \
  --entrypoint bash \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811 \
  -c "python3 -u tuning_fused_moe_triton.py \
        --model /data1/models/dsv4-0731-w8a8-dspark \
        --tp 8 \
        --dtype int8_w8a8 \
        --per-channel-quant \
        --tune \
        --search-space-file /tune/$SPACE \
        --batch-sizes $BS 2>&1 | tee /tune/tune_${MAXBS}.log"
echo "=== 产出 ==="
ls -la "$D"/*.json
