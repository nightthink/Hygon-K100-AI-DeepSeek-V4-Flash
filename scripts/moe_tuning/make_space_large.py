"""大批量（预填充）搜索空间。

小批量那份空间把 BLOCK_SIZE_M 限制在 32/64，对 M=数百到数千的预填充场景不合适。
这里换一组：M 方向放大，K 方向收窄（大 M 下 K 轮次不再是瓶颈），仍然剔除 num_warps=1。

3(block_m) x 4(block_n) x 2(block_k) x 2(num_warps) x 2(group_size) = 96 个配置，
每个批量档位约 9.5 分钟。

注意：调优器每次运行会用"本次的 batch 列表"整体覆盖同名 json，
所以小批量与大批量两轮的结果必须事后合并（见 merge_configs.py）。
"""

import json
import os

configs = []
for block_m in [64, 128, 256]:
    for block_n in [32, 64, 128, 256]:
        for block_k in [64, 128]:
            for num_warps in [4, 8]:
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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "space_large.json")
with open(out, "w") as f:
    json.dump(configs, f, indent=2)
print(f"wrote {len(configs)} configs to {out}")
