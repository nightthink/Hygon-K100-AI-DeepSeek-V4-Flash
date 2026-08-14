"""合并两轮调优结果。

调优器每次运行都会用"本次 batch 列表"整体覆盖同名 json，所以：
  第一轮（解码批量 1-16，space_small）结果先备份为 decode_configs_backup.json
  第二轮（预填批量 32-4096，space_large）结果留在正式文件名里
本脚本把两者合并成一份完整配置，并放进 configs/ 供挂载。
"""

import json
import os

D = os.path.dirname(os.path.abspath(__file__))
NAME = "E=256,N=256,device_name=K100_AI,dtype=int8_w8a8,per_channel_quant=True.json"

large = json.load(open(os.path.join(D, NAME)))
small = json.load(open(os.path.join(D, "decode_configs_backup.json")))

merged = {}
merged.update(small)
merged.update(large)
merged = {k: merged[k] for k in sorted(merged, key=int)}

overlap = set(small) & set(large)
if overlap:
    print(f"警告：两轮有重叠 batch {sorted(overlap, key=int)}，以大批量轮为准")

os.makedirs(os.path.join(D, "configs"), exist_ok=True)
out = os.path.join(D, "configs", NAME)
with open(out, "w") as f:
    json.dump(merged, f, indent=4)
    f.write("\n")

print(f"合并完成，共 {len(merged)} 个批量档位：{sorted(merged, key=int)}")
print(f"输出：{out}")
