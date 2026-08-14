#!/bin/bash
# 0731 权重量化管线：FP4/FP8 -> BF16 -> W8A8(INT8 experts)
# 全程 CPU（GPU 被生产服务占用），后台跑，日志 /tmp/pipeline_0731.log
set -e

SRC=/data1/models/DeepSeek-V4-Flash-0731
BF16=/data1/models/dsv4-0731-bf16
W8A8=/data1/models/dsv4-0731-w8a8
IMG=lzd/dsv4-flash-k100ai-sglang:0728-patched-v2

echo "=== [$(date +%H:%M:%S)] 阶段 0：前置检查 ==="
n=$(ls "$SRC"/*.safetensors 2>/dev/null | wc -l)
echo "shards: $n (期望 48)"
[ "$n" -eq 48 ] || { echo "分片不全，中止"; exit 1; }
[ -f "$SRC/model.safetensors.index.json" ] || { echo "缺 index.json，中止"; exit 1; }
[ -f "$SRC/tokenizer.json" ] || { echo "缺 tokenizer.json，中止"; exit 1; }
df -h /data1 | tail -1

echo "=== [$(date +%H:%M:%S)] 阶段 1：反量化 FP4/FP8 -> BF16（CPU，耗时较长）==="
docker run --rm -v /data1:/data1 -v /home/user:/home/user -v /opt/hyhal:/opt/hyhal \
  --entrypoint python3 "$IMG" \
  /data1/flagos_offline_ws/convert_weight.py \
    --input-fp8-hf-path "$SRC" \
    --output-bf16-hf-path "$BF16" \
    --device cpu
echo "[$(date +%H:%M:%S)] BF16 完成：$(du -sh $BF16 | cut -f1)"

echo "=== [$(date +%H:%M:%S)] 阶段 2：清理 BF16 的 config（去掉原 fp8 量化声明）==="
docker run --rm -v /data1:/data1 --entrypoint python3 "$IMG" -c \
  "import json;p='$BF16/config.json';c=json.load(open(p));r=c.pop('quantization_config',None);json.dump(c,open(p,'w'),indent=2);print('removed quantization_config:',r is not None)"

echo "=== [$(date +%H:%M:%S)] 阶段 3：量化 BF16 -> W8A8（排除 mtp.*，写入 sglang ignore 规则）==="
docker run --rm -v /data1:/data1 -v /home/user:/home/user --entrypoint python3 "$IMG" \
  /home/user/quant_w8a8_0731.py --input "$BF16" --output "$W8A8"

echo "=== [$(date +%H:%M:%S)] 阶段 4：产物校验 ==="
ls "$W8A8"/*.safetensors | wc -l
du -sh "$W8A8"
docker run --rm -v /data1:/data1 --entrypoint python3 "$IMG" -c \
  "import json;c=json.load(open('$W8A8/config.json'));q=c['quantization_config'];print('ignore:',q['ignore']);print('dspark fields left:',[k for k in c if k.startswith('dspark')]);print('layers:',c['num_hidden_layers'])"
df -h /data1 | tail -1
echo "=== [$(date +%H:%M:%S)] PIPELINE_DONE ==="
