"""DSV4 专家 AWQ int4 量化 v2：加 α-clip 搜索（每组 MSE 最优截断）。

v1（min-max RTN）的问题：目标模型分布偏移较大，DSpark 接受率从 0.68 崩到 0.25-0.36。
v2 对每个 (group, out_channel) 在 α∈{1.0,0.95,0.9,0.85,0.8} 中搜索使量化 MSE 最小的
截断范围（保持组中点，双侧收缩），离群值饱和、主体精度提升——AutoAWQ clip 搜索的
向量化简化版。MTP 专家保持 bf16。

用法（容器内）：挂载 /bf16（反量化源）、/refcfg（服务验证过的 config 来源）、/out。
流式、GPU 加速、.done 标记断点续跑。实测全量约 35-40 分钟。
"""
import glob
import json
import os
import re
import time

import torch
from safetensors.torch import load_file, save_file

SRC, OUT, REF = "/bf16", "/out", "/refcfg"
G = 128
ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
ALPHAS = [1.0, 0.95, 0.9, 0.85, 0.8]
EXP = re.compile(r"^(layers)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")

os.makedirs(OUT, exist_ok=True)
dev = "cuda"


def pack_int32(m):
    out = torch.zeros(m.shape[0], m.shape[1] // 8, dtype=torch.int32, device=m.device)
    for i, src in enumerate(ORDER):
        out |= (m[:, src::8] & 0xF) << (4 * i)
    return out


def awq_quant(w):
    wt = w.t().contiguous().to(dev, non_blocking=True).float()
    K, N = wt.shape
    v = wt.view(K // G, G, N)
    mx, mn = v.amax(1), v.amin(1)
    mid, half0 = (mx + mn) / 2, (mx - mn) / 2

    Ss, Zs, Es = [], [], []
    for a in ALPHAS:
        half = half0 * a
        lo = mid - half
        s = (2 * half).clamp_min(1e-8) / 15.0
        z = torch.round(-lo / s).clamp(0, 15)
        q = torch.round(v / s.unsqueeze(1) + z.unsqueeze(1)).clamp(0, 15)
        deq = (q - z.unsqueeze(1)) * s.unsqueeze(1)
        Es.append(((deq - v) ** 2).sum(1))
        Ss.append(s)
        Zs.append(z)
    idx = torch.stack(Es).argmin(0)                       # (g,N)
    s = torch.stack(Ss).gather(0, idx[None])[0]
    z = torch.stack(Zs).gather(0, idx[None])[0]
    q = torch.round(v / s.unsqueeze(1) + z.unsqueeze(1)).clamp(0, 15)
    q = q.to(torch.int32).view(K, N)
    return (pack_int32(q).cpu(), pack_int32(z.to(torch.int32)).cpu(),
            s.to(torch.bfloat16).cpu())


shards = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
print(f"输入分片: {len(shards)}  α搜索: {ALPHAS}", flush=True)

weight_map = {}
total = 0
t0 = time.time()

for si, sp in enumerate(shards):
    name = os.path.basename(sp)
    op = os.path.join(OUT, name)
    mark = op + ".done"
    if os.path.exists(mark):
        wm = json.load(open(mark))
        weight_map.update(wm["map"])
        total += wm["bytes"]
        print(f"[{si+1}/{len(shards)}] {name} 已完成，跳过", flush=True)
        continue

    sd = load_file(sp)
    out = {}
    for k, w in sd.items():
        if EXP.match(k):
            q, z, s = awq_quant(w)
            base = k[: -len(".weight")]
            out[base + ".qweight"] = q
            out[base + ".qzeros"] = z
            out[base + ".scales"] = s
        else:
            out[k] = w
    save_file(out, op, metadata={"format": "pt"})

    wm, b = {}, 0
    for k, v in out.items():
        wm[k] = name
        b += v.numel() * v.element_size()
    weight_map.update(wm)
    total += b
    json.dump({"map": wm, "bytes": b}, open(mark, "w"))
    el = time.time() - t0
    eta = el / (si + 1) * (len(shards) - si - 1)
    print(f"[{si+1}/{len(shards)}] {name} → {b/1e9:.2f}GB  累计 {total/1e9:.1f}GB"
          f"  ETA {eta/60:.1f}min", flush=True)
    del sd, out

json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"))

cfg = json.load(open(os.path.join(REF, "config.json")))
cfg["quantization_config"] = {
    "quant_method": "awq", "bits": 4, "group_size": G, "zero_point": True,
    "version": "gemm",
    "modules_to_not_convert": ["attn", "shared_experts", "ffn.gate",
                               "embed", "head", "hc_", "indexer", "mtp"],
}
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)

import shutil
for f in os.listdir(REF):
    if f.startswith("tokenizer") or f == "generation_config.json":
        shutil.copy(os.path.join(REF, f), os.path.join(OUT, f))

print(f"\n完成: {total/1e9:.1f}GB, 耗时 {(time.time()-t0)/60:.1f}min", flush=True)
