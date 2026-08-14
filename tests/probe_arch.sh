#!/bin/bash
# gfx928 构建目标探针：在花时间做实验之前，先确认某个功能的内核到底有没有为本卡编译过。
#
# 用法：bash probe_arch.sh <镜像名>
#
# 背景：海光镜像里不同内核库的 offload-arch 覆盖并不一致——sgl_kernel 编了 gfx928，
# 而 aiter 与 deep_ep 只编了 gfx936/gfx938（BW 卡）。在 K100-AI（gfx928）上，
# 后两者会在初始化时报 "invalid kernel file (218)"，即胖二进制里没有匹配架构的设备代码。
#
# 这个探针在轮次 28 里用约 20 分钟否掉了两条原本各需半天到一天的调优路线。
set -uo pipefail
IMG=${1:?用法: bash probe_arch.sh <镜像名>}

echo "=== 一、各内核库编进了哪些 gfx 目标 ==="
docker run --rm -v /opt/hyhal:/opt/hyhal --entrypoint bash "$IMG" -c '
for m in sgl_kernel aiter lightop deep_ep; do
  D=$(python3 -c "import $m,os;print(os.path.dirname($m.__file__))" 2>/dev/null)
  [ -z "$D" ] && { echo "$m => 模块缺失"; continue; }
  for f in $(find $D -name "*.so" -size +1M 2>/dev/null | head -4); do
    A=$(strings $f 2>/dev/null | grep -oE "gfx[0-9]{3,4}" | sort -u | tr "\n" " ")
    echo "$m/$(basename $f | cut -c1-34) => ${A:-无gfx}"
  done
done'

echo
echo "=== 二、算子是否真被编进构建 ==="
echo "（torch.ops 用惰性 __getattr__，dir() 不枚举算子，只有直接取属性才作数）"
docker run --rm -v /opt/hyhal:/opt/hyhal --device=/dev/kfd --device=/dev/dri \
  --group-add video --entrypoint python3 "$IMG" -c '
import torch, sgl_kernel
ns = torch.ops.sgl_kernel
names = ["cutlass_w4a8_moe_mm", "qserve_w4a8_per_chn_gemm", "gptq_gemm",
         "awq_dequantize", "moe_align_block_size"]
print("架构:", torch.cuda.get_device_properties(0).gcnArchName)
for n in names:
    try:
        getattr(ns, n); print("  在   ", n)
    except Exception:
        print("  缺   ", n)
print("注：moe_align_block_size 是对照组，它若也显示“缺”，说明探针本身有问题。")'
