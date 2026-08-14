#!/bin/bash
# 阶段 3+4 重跑（修复：容器需挂 /opt/hyhal，否则 torch 导入报 librocm_smi64.so.2 缺失）
set -e
BF16=/data1/models/dsv4-0731-bf16
W8A8=${OUT:-/data1/models/dsv4-0731-w8a8}
EXCL=${EXCL-mtp.}
EXTRA=${EXTRA:-}
IMG=lzd/dsv4-flash-k100ai-sglang:0728-patched-v2

echo "=== [$(date +%H:%M:%S)] 量化 $BF16 -> $W8A8 (exclude='$EXCL' extra='$EXTRA') ==="
df -h /data1 | tail -1

docker run --rm \
  -v /data1:/data1 -v /home/user:/home/user -v /opt/hyhal:/opt/hyhal \
  --entrypoint python3 "$IMG" \
  /home/user/quant_w8a8_0731.py --input "$BF16" --output "$W8A8" \
    --exclude-prefix "$EXCL" $EXTRA

echo "=== [$(date +%H:%M:%S)] 产物校验 ==="
ls "$W8A8"/*.safetensors | wc -l
du -sh "$W8A8"
docker run --rm -v /data1:/data1 -v /opt/hyhal:/opt/hyhal --entrypoint python3 "$IMG" -c \
  "import json;c=json.load(open('$W8A8/config.json'));q=c['quantization_config'];print('ignore:',q['ignore']);print('dspark fields:',[k for k in c if k.startswith('dspark')]);print('layers:',c['num_hidden_layers'])"
df -h /data1 | tail -1
echo "=== [$(date +%H:%M:%S)] QUANT_DONE ==="
