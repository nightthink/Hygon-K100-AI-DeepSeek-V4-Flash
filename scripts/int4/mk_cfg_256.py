"""为 DSV4 int4（E=256, N=128, TP8）生成分批量的 MoE 内核配置。

按批量档位分段选 tile：解码档小 M 用小 BM，预填档大 M 用大 tile 高并行。
文件名由服务日志报出的 miss 决定：
  E=256,N=128,device_name=K100_AI,dtype=int4_w4a16.json 及 _down。
服务时以 -v 挂载到镜像的 srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_0/。
"""
import json
import os

OUT = "/work/int4cfg256"
os.makedirs(OUT, exist_ok=True)

cfg = {}
for m in [1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096]:
    if m <= 16:
        c = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
             "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    elif m <= 64:
        c = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
             "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    else:
        c = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2}
    c["waves_per_eu"] = 0
    cfg[str(m)] = c

for n in ["E=256,N=128,device_name=K100_AI,dtype=int4_w4a16.json",
          "E=256,N=128,device_name=K100_AI,dtype=int4_w4a16_down.json"]:
    json.dump(cfg, open(os.path.join(OUT, n), "w"), indent=1)
    print("已写:", n)
