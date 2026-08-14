"""预检：确认 0731 的 DSpark(mtp) 张量在反量化脚本里会被如何处理。

convert_weight.py 的分流规则：
  - 名字带 experts 且非 shared_experts + 有 .scale  -> MXFP4 反量化
  - 其它 + 有 .scale                                -> 128x128 blockwise FP8 反量化
  - 无 .scale                                       -> 原样直通
本脚本只读 index.json + 已下载分片的元数据，不加载权重数据。
"""
import json, os, glob
from collections import defaultdict

IDX = "/tmp/idx0731.json"
DIR = "/data1/models/DeepSeek-V4-Flash-0731"

wm = json.load(open(IDX))["weight_map"]
mtp = {k: v for k, v in wm.items() if k.startswith("mtp.")}
print("mtp 张量总数:", len(mtp))

# 哪些 mtp 张量带 .scale（说明是量化存储，需要反量化）
scales = {k for k in mtp if k.endswith(".scale")}
weights = {k for k in mtp if k.endswith(".weight")}
paired = {w for w in weights if w[: -len("weight")] + "scale" in mtp}
print("mtp 中 .scale 数:", len(scales))
print("mtp 中 .weight 数:", len(weights))
print("mtp 中带 scale 的 weight（需反量化）:", len(paired))
for k in sorted(paired)[:10]:
    print("   ", k)

# 这些张量会走哪条分支？（是否被当作 expert）
def is_expert(name):
    return "experts" in name and "shared_experts" not in name

exp = [k for k in paired if is_expert(k)]
non = [k for k in paired if not is_expert(k)]
print("  -> 走 MXFP4(expert) 分支:", len(exp))
print("  -> 走 blockwise FP8 分支:", len(non))
for k in sorted(non)[:8]:
    print("       ", k)

# 直通的 mtp 张量（无 scale）
passthru = sorted(set(mtp) - paired - scales)
print("直通(无 scale)的 mtp 张量:", len(passthru))
for k in passthru[:12]:
    print("   ", k)

# 检查已下载分片里的实际 dtype/shape
print("\n=== 已下载分片中的 mtp 张量实测 dtype/shape ===")
have = {os.path.basename(p) for p in glob.glob(os.path.join(DIR, "model-*.safetensors"))}
by_shard = defaultdict(list)
for k, s in mtp.items():
    if s in have:
        by_shard[s].append(k)
if not by_shard:
    print("（含 mtp 的分片尚未下载完，稍后再查）")
else:
    from safetensors import safe_open
    shown = 0
    for s, keys in sorted(by_shard.items()):
        with safe_open(os.path.join(DIR, s), framework="pt", device="cpu") as f:
            for k in sorted(keys):
                sl = f.get_slice(k)
                print(f"  {k}: dtype={sl.get_dtype()} shape={sl.get_shape()}")
                shown += 1
                if shown >= 25:
                    break
        if shown >= 25:
            break
