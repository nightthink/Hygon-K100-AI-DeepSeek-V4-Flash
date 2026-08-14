"""阶段二：把小 MoE 的专家权重量化为 AWQ int4，并在 CPU 上验证打包正确。

## 为什么要先在 CPU 上验证

上一个探针（tests/probe_int4_triton.py）直接手搓张量喂给内核，结果全零 / 段错误，
无法区分"内核有问题"和"我打包错了"。本脚本先用 **sglang 自己的 convert_awq_tensor 逻辑**
把打包结果还原回整数权重，与量化前的整数逐元素比对——**完全不碰 GPU**。
只有往返一致，才有资格去谈内核对不对。

实测结果：48/48 个专家线性层往返完全一致，量化最大相对误差 0.0636。

## AWQ 磁盘格式（取自 moe_wna16.py::convert_awq_tensor 的注释与实现）

    qweight: (in_features,               out_features // 8)  int32
    qzeros:  (in_features // group_size, out_features // 8)  int32
    scales:  (in_features // group_size, out_features)       bf16

打包置换 ORDER = [0, 2, 4, 6, 1, 3, 5, 7]。
校验：sglang 解包时用 REVERSE = [0,4,1,5,2,6,3,7]，二者互逆
（ORDER[REVERSE[j]] == j，故还原后第 j 个 nibble 正是第 j 个输出通道）。

## 只量化专家

非 MoE 的 Linear 在 moe_wna16 下会路由到 AWQConfig，需要 awq_dequantize 等算子，
而这些算子**没编进本镜像的 ROCm 构建**（轮次 28/32）。因此非专家层全部走
modules_to_not_convert 保持 bf16。这也正是 DSV4 的真实需求——只有专家是带宽瓶颈。

注意 is_layer_skipped_quant 是**纯子串匹配**：
    return any(module_name in prefix for module_name in modules_to_not_convert)
所以用 "mlp.gate" 而不是 "gate"，否则会误伤专家的 gate_proj。
"""
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SRC = "/work/tiny-moe-bf16"
DST = "/work/tiny-moe-awq"
GROUP = 128
ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
REVERSE = [0, 4, 1, 5, 2, 6, 3, 7]

os.makedirs(DST, exist_ok=True)


def quantize_group_asym(wt, group):
    """wt: (in, out) float。沿 in 维分组做非对称 int4 量化。

    返回 q (in,out) int32 in [0,15]、zeros (in/group,out) int32、scales (in/group,out) float
    """
    k, n = wt.shape
    assert k % group == 0, f"in_features {k} 不是 group {group} 的整数倍"
    g = k // group
    w = wt.reshape(g, group, n).float()
    wmax = w.amax(dim=1)
    wmin = w.amin(dim=1)
    scales = (wmax - wmin).clamp(min=1e-8) / 15.0
    zeros = torch.round(-wmin / scales).clamp(0, 15)
    q = torch.round(w / scales.unsqueeze(1) + zeros.unsqueeze(1)).clamp(0, 15)
    return q.reshape(k, n).to(torch.int32), zeros.to(torch.int32), scales


def pack_int32(mat):
    """mat: (a, b) int32，值域 [0,15]，沿 b 维每 8 个打包进一个 int32，使用 AWQ order。"""
    a, b = mat.shape
    assert b % 8 == 0
    out = torch.zeros(a, b // 8, dtype=torch.int32)
    for i, src in enumerate(ORDER):
        out |= (mat[:, src::8] & 0xF) << (4 * i)
    return out


def sglang_unpack(tensor, tensor_type):
    """逐字复刻 moe_wna16.py::convert_awq_tensor，用于往返校验。"""
    size0 = tensor.size(0)
    t = tensor.view(torch.uint8)
    shifter = torch.tensor([0, 4], dtype=torch.uint8)
    t = (t[:, :, None] >> shifter) & 0xF
    t = t.view(-1, 8)[:, REVERSE]
    t = t.view(size0, -1)
    t = t.T.contiguous()
    if tensor_type == "qweight":
        t = t[:, 1::2] * 16 + t[:, ::2]
    elif tensor_type == "qzeros":
        t = t[1::2, :] * 16 + t[::2, :]
    return t


src_file = glob.glob(os.path.join(SRC, "*.safetensors"))[0]
sd = load_file(src_file)
print(f"源张量 {len(sd)} 个")

new_sd = {}
n_quant = 0
max_rel_err = 0.0
roundtrip_ok = True

for name, w in sd.items():
    if not (".mlp.experts." in name and name.endswith(".weight")):
        new_sd[name] = w
        continue

    wt = w.t().contiguous()                      # (out,in) -> (in,out)
    q, zeros, scales = quantize_group_asym(wt, GROUP)

    qweight = pack_int32(q)
    qzeros = pack_int32(zeros)

    # ---- 往返校验：用 sglang 的还原逻辑取回 nibble，与 q 比对 ----
    back = sglang_unpack(qweight, "qweight")     # (out, in/2) uint8
    lo = (back & 0xF).to(torch.int32)
    hi = ((back >> 4) & 0xF).to(torch.int32)
    recovered = torch.stack([lo, hi], dim=-1).reshape(back.size(0), -1)
    if not torch.equal(recovered, q.T.contiguous()):
        roundtrip_ok = False
        print(f"  ✗ 往返不一致 {name}: {(recovered != q.T).sum().item()} 个元素不符")

    deq = (q.reshape(-1, GROUP, q.shape[1]).float()
           - zeros.unsqueeze(1).float()) * scales.unsqueeze(1)
    deq = deq.reshape(wt.shape)
    err = (deq - wt.float()).abs().max() / wt.float().abs().max().clamp(min=1e-8)
    max_rel_err = max(max_rel_err, err.item())

    base = name[: -len(".weight")]
    new_sd[base + ".qweight"] = qweight
    new_sd[base + ".qzeros"] = qzeros
    new_sd[base + ".scales"] = scales.to(torch.bfloat16)
    n_quant += 1

print(f"量化专家线性层: {n_quant} 个")
print(f"往返校验: {'✅ 全部一致' if roundtrip_ok else '❌ 存在不一致'}")
print(f"量化最大相对误差: {max_rel_err:.4f}")

if not roundtrip_ok:
    raise SystemExit("打包与 sglang 的还原逻辑不一致，先修打包再上 GPU")

save_file(new_sd, os.path.join(DST, "model.safetensors"), metadata={"format": "pt"})

cfg = json.load(open(os.path.join(SRC, "config.json")))
cfg["quantization_config"] = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": GROUP,
    "zero_point": True,
    "version": "gemm",
    # 纯子串匹配：用 mlp.gate 而非 gate，否则会误伤专家的 gate_proj
    "modules_to_not_convert": ["self_attn", "shared_expert", "mlp.gate", "lm_head"],
}
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=1)

for f in os.listdir(SRC):
    if f.startswith("tokenizer") or f in ("special_tokens_map.json", "generation_config.json"):
        shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))

print("已保存:", DST)
for k in sorted(new_sd):
    if "experts.0.gate_proj" in k:
        print(f"   {k}: {tuple(new_sd[k].shape)} {new_sd[k].dtype}")
