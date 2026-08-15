"""DSV4 精度混布量化（轮次 37）：int4 主体 + 三层 bf16。

在 v2（α-clip AWQ int4）的基础上，把 DSpark 取材相关的最后三层
（layers.40/41/42）的路由专家保留 bf16，其余 40 层维持 int4。
假设：DSpark 接受率崩塌（0.68→~0.3）主要来自靠近输出端、被 MTP 头取用的
隐状态特征漂移；把这几层还原到 bf16 精度，或可显著恢复接受率。
（实测结论：证伪——接受率不回升，漂移是全主干累积，非局部。见 docs/调优记录-轮次37.md）

实现上只与 v2 差两处：
  1. EXP 负向先行断言排除 layers.40/41/42 → 这些专家逐字复制为 bf16 .weight
  2. config.quantization_config.modules_to_not_convert 追加
     layers.{40,41,42}.ffn.experts → sglang 的 moe_wna16 对这些 FusedMoE
     返回 UnquantizedFusedMoEMethod（bf16，走已修复的 moe_align 路径）
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
# 排除 40/41/42：负向先行断言。layers.4./layers.14. 等仍匹配（量化），
# 仅 layers.40./41./42. 落空 → 保持 bf16。
EXP = re.compile(r"^layers\.(?!(40|41|42)\.)\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
BF16_LAYERS = ["layers.40.ffn.experts", "layers.41.ffn.experts",
               "layers.42.ffn.experts"]

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
    idx = torch.stack(Es).argmin(0)
    s = torch.stack(Ss).gather(0, idx[None])[0]
    z = torch.stack(Zs).gather(0, idx[None])[0]
    q = torch.round(v / s.unsqueeze(1) + z.unsqueeze(1)).clamp(0, 15)
    q = q.to(torch.int32).view(K, N)
    return (pack_int32(q).cpu(), pack_int32(z.to(torch.int32)).cpu(),
            s.to(torch.bfloat16).cpu())


shards = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
print(f"输入分片: {len(shards)}  α搜索: {ALPHAS}  bf16保留层: {BF16_LAYERS}", flush=True)

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
    nq = 0
    for k, w in sd.items():
        if EXP.match(k):
            q, z, s = awq_quant(w)
            base = k[: -len(".weight")]
            out[base + ".qweight"] = q
            out[base + ".qzeros"] = z
            out[base + ".scales"] = s
            nq += 1
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
    print(f"[{si+1}/{len(shards)}] {name} → {b/1e9:.2f}GB  量化专家权重 {nq}"
          f"  累计 {total/1e9:.1f}GB  用时 {el/60:.1f}min", flush=True)
    del sd, out

json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"))

cfg = json.load(open(os.path.join(REF, "config.json")))
cfg["quantization_config"] = {
    "quant_method": "awq", "bits": 4, "group_size": G, "zero_point": True,
    "version": "gemm",
    "modules_to_not_convert": ["attn", "shared_experts", "ffn.gate",
                               "embed", "head", "hc_", "indexer", "mtp"]
    + BF16_LAYERS,
}
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)

import shutil
for f in os.listdir(REF):
    if f.startswith("tokenizer") or f == "generation_config.json":
        shutil.copy(os.path.join(REF, f), os.path.join(OUT, f))

print(f"\n完成: {total/1e9:.1f}GB, 耗时 {(time.time()-t0)/60:.1f}min", flush=True)
