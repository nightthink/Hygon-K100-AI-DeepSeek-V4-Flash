"""Stream-quantize DeepSeek-V4-Flash-0731 BF16 -> compressed-tensors W8A8 (INT8 MoE experts).

在 /data1/quant_w8a8_stream.py 基础上针对 0731 版增加三点：
  1) --exclude-prefix：可排除 mtp.*（0731 的 MTP 是 DSpark 结构：main_norm / main_proj /
     markov_head / confidence_head）。给 0728 镜像用时要排除（其 sglang 0.5.12 只认旧式
     enorm/hnorm/e_proj/h_proj）；给 0811 镜像跑 DSpark 时要保留（--exclude-prefix ""）。
  2) --keep-dspark-config：保留 config.json 中的 dspark_* 字段（0811/DSpark 用）；
     默认删除（0728 用）。
  3) config.json 直接写入 sglang 兼容的 ignore 规则（负向先行断言，只量化 experts），
     省掉后续手工打补丁。

用法（容器内 CPU 即可，无需 GPU）：
  # 给 0728 镜像（无 DSpark）：
  python3 quant_w8a8_0731.py --input /data1/models/dsv4-0731-bf16 \
      --output /data1/models/dsv4-0731-w8a8
  # 给 0811 镜像（跑 DSpark）：
  python3 quant_w8a8_0731.py --input /data1/models/dsv4-0731-bf16 \
      --output /data1/models/dsv4-0731-w8a8-dspark --exclude-prefix "" --keep-dspark-config
"""

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def is_expert_weight(name: str, shape) -> bool:
    return ".experts." in name and name.endswith(".weight") and len(shape) == 2


def quantize_per_channel_int8(w: torch.Tensor):
    """对 [out, in] 权重做输出通道对称 INT8 量化，返回 (int8 权重, fp32 scale[out,1])"""
    w32 = w.to(torch.float32)
    scale = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(w32 / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--exclude-prefix",
        default="mtp.",
        help='逗号分隔的张量名前缀，命中即丢弃；传 "" 表示全部保留（DSpark 用）',
    )
    ap.add_argument(
        "--keep-dspark-config",
        action="store_true",
        help="保留 config.json 中的 dspark_* 字段（0811 镜像跑 DSpark 时需要）",
    )
    args = ap.parse_args()

    excludes = tuple(p for p in args.exclude_prefix.split(",") if p)
    os.makedirs(args.output, exist_ok=True)

    shard_files = sorted(glob.glob(os.path.join(args.input, "model-*.safetensors")))
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(args.input, "*.safetensors")))
    assert shard_files, f"no safetensors found in {args.input}"

    weight_map = {}
    total_size = 0
    n_quant = 0
    n_keep = 0
    n_drop = 0

    for i, shard_path in enumerate(shard_files):
        shard_name = os.path.basename(shard_path)
        out_tensors = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if excludes and name.startswith(excludes):
                    n_drop += 1
                    continue
                t = f.get_tensor(name)
                if is_expert_weight(name, t.shape):
                    q, scale = quantize_per_channel_int8(t)
                    out_tensors[name] = q
                    out_tensors[name.replace(".weight", ".weight_scale")] = scale
                    n_quant += 1
                else:
                    out_tensors[name] = t
                    n_keep += 1
        if not out_tensors:
            print(f"[{i+1}/{len(shard_files)}] {shard_name} 全部被排除，跳过", flush=True)
            continue
        save_file(
            out_tensors,
            os.path.join(args.output, shard_name),
            metadata={"format": "pt"},
        )
        for name, t in out_tensors.items():
            weight_map[name] = shard_name
            total_size += t.numel() * t.element_size()
        del out_tensors
        print(
            f"[{i+1}/{len(shard_files)}] {shard_name} done "
            f"(quantized: {n_quant}, kept: {n_keep}, dropped: {n_drop})",
            flush=True,
        )

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # 拷贝 config / tokenizer / encoding 等非权重文件
    for fname in os.listdir(args.input):
        src = os.path.join(args.input, fname)
        if os.path.isdir(src):
            if fname in ("encoding", "inference", "assets"):
                dst = os.path.join(args.output, fname)
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
            continue
        if fname.endswith(".safetensors"):
            continue
        if "safetensors" in fname and fname.endswith(".json"):
            continue
        shutil.copy(src, os.path.join(args.output, fname))

    cfg_path = os.path.join(args.output, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    if not args.keep_dspark_config:
        for k in (
            "dspark_block_size",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
            "dspark_markov_rank",
        ):
            cfg.pop(k, None)

    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["re:.*\\.experts\\..*"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
            }
        },
        # sglang 兼容：负向先行断言，非 experts 一律 ignore（否则报 wqkv_a target mismatch）
        "ignore": ["lm_head", "re:^(?!.*\\.experts\\.).*"],
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"\nDone. quantized expert weights: {n_quant}, kept bf16: {n_keep}, "
        f"dropped (excluded): {n_drop}"
    )
    print(f"output: {args.output} (total {total_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
