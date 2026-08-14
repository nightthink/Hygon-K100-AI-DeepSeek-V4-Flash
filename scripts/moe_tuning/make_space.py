"""生成裁剪后的 MoE triton 搜索空间（解码档，batch 1-16）。

sglang 默认的 ROCm 搜索空间是 4(block_m) x 5(block_n) x 4(block_k) x 4(num_warps)
x 5(group_size) = 1600 个配置，其中大量组合对我们的场景毫无意义，而且
num_warps=1 配上大 tile 会产生单个耗时数十秒的内核（实测出现过 85 s/config，
整体 ETA 被拖到 17 小时）。

裁剪依据（我方负载：解码为主，batch 1-16，MoE 每专家的有效行数很少）：
- num_warps=1 全部剔除：小 M 下并行度不足，正是拖慢搜索的元凶。
- BLOCK_SIZE_M 只留 32/64：M 本来就小，128/256 纯属浪费。
- BLOCK_SIZE_K 剔除 32：K=7168，块太小会产生过多 K 轮次。
- GROUP_SIZE_M 只留 1/8：swizzle 分组对小 batch 影响很小。

结果：2 x 3 x 3 x 3 x 2 = 108 个配置，约为原来的 1/15，每个批量档位约 3.5 分钟。
"""

import json
import os

configs = []
for block_m in [32, 64]:
    for block_n in [32, 64, 128]:
        for block_k in [64, 128, 256]:
            for num_warps in [2, 4, 8]:
                for group_size in [1, 8]:
                    configs.append(
                        {
                            "BLOCK_SIZE_M": block_m,
                            "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_K": block_k,
                            "GROUP_SIZE_M": group_size,
                            "num_warps": num_warps,
                            "num_stages": 2,
                            "waves_per_eu": 0,
                        }
                    )

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "space_small.json")
with open(out, "w") as f:
    json.dump(configs, f, indent=2)
print(f"wrote {len(configs)} configs to {out}")
