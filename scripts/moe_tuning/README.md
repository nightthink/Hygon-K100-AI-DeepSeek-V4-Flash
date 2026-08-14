# 为 K100-AI 现场调优 MoE triton 内核配置

> **先说结论**：这条路我们走通了，但**对 DeepSeek-V4-Flash 的解码/预填没有收益**
> （解码 +1% 在噪声内、预填 0%）。原因与完整数据见 `docs/调优记录-轮次26.md`。
> 本目录保留全部工具与产物，是为了让后来者不必重走这半天，同时把配置文件贡献出来。

## 为什么会想到这条路

0811 镜像里 `sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_0/`
**是空目录——0 个文件，任何型号的卡都没有调优配置**。于是每次 MoE 计算都走启发式默认配置，
日志固定刷：

```
Using default MoE kernel config. Performance might be sub-optimal!
Config file not found at .../E=256,N=256,device_name=K100_AI,dtype=int8_w8a8,per_channel_quant=True.json
```

而 MoE 是解码单步（约 130 ms）里的大头，这条又**完全不依赖上游**，所以优先级最高。

## 复现步骤

```bash
# 1. 取上游调优器（镜像里没有）
mkdir -p moe_tuning && cd moe_tuning
B=https://raw.githubusercontent.com/HYGON-AI/sglang-das/main/benchmark/kernels/fused_moe_triton
curl -sSLO $B/tuning_fused_moe_triton.py
curl -sSLO $B/common_utils.py

# 2. 打补丁（per-channel 量化会让调优器崩溃，见下）
python3 patch_tuner.py

# 3. 生成裁剪后的搜索空间
python3 make_space.py        # 解码档，108 个配置
python3 make_space_large.py  # 预填档，96 个配置

# 4. 分两轮调优（需独占全部 GPU，先停掉推理服务）
bash run_tune.sh "1 2 4 6 8 16"                      # 约 3.5 分钟
bash run_tune.sh "32 64 128 256 512 1024 2048 4096"  # 约 9.5 分钟

# 5. 合并两轮结果（调优器每轮都会整体覆盖同名文件）
python3 merge_configs.py

# 6. 以只读方式挂进镜像
#    -v $PWD/configs:/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/moe/\
#       moe_runner/triton_utils/configs/triton_3_5_0:ro
```

生效验证：日志里 `Using default MoE kernel config` 消失，改为
`Down MoE config file not found ... reusing the tuned up-projection config`
——下投影配置是可选的，缺失时复用已调优的上投影配置，不会退回启发式。

## 两个必须知道的坑

### 1. 调优器对 per-channel 量化直接崩溃

`common_utils.get_model_config` 见到 compressed-tensors 的 `config_groups` 就无条件取
`weights.group_size` 组成 `block_shape=[0, group_size]`。per-channel 量化的 `group_size`
是 `null`，于是下游 `block_k % config["BLOCK_SIZE_K"]` 拿 `None` 取模：

```
TypeError: unsupported operand type(s) for %: 'NoneType' and 'int'
```

`patch_tuner.py` 修的就是这个（正确语义：per-channel 无 block 概念，`block_shape` 应为 None）。

### 2. ROCm 默认搜索空间里有病态组合，会把调优拖到 17 小时

`get_rocm_configs_compute_bound()` 是 4×5×4×4×5 = **1600 个配置**。实测跑到中段时
单个配置耗时飙到 **85 秒**，ETA 从 40 分钟变成 **16:58:08**。元凶是 `num_warps=1` 配大 tile。

`make_space*.py` 的裁剪依据：

| 维度 | 裁剪 | 理由 |
|---|---|---|
| `num_warps` | 去掉 1 | 小 M 下并行度不足，正是拖慢搜索的元凶 |
| `BLOCK_SIZE_M` | 解码档只留 32/64 | M 本来就小（每步约 6 token × topk 8） |
| `BLOCK_SIZE_K` | 解码档去掉 32 | K=7168，块太小会产生过多 K 轮次 |
| `GROUP_SIZE_M` | 只留 1/8 | swizzle 分组对小 batch 影响很小 |

裁剪后 108 / 96 个配置，每个批量档位 3.5 / 9 分钟跑完，结果质量无损。

## 文件

| 文件 | 说明 |
|---|---|
| `patch_tuner.py` | per-channel 量化崩溃补丁 |
| `make_space.py` / `make_space_large.py` | 裁剪后的搜索空间生成器 |
| `run_tune.sh` | 容器化调优入口 |
| `merge_configs.py` | 合并两轮结果 |
| `configs/E=256,N=256,device_name=K100_AI,...json` | **调优产物**，14 个批量档位 |

---

Copyright © 2026 DaoTech Team. Licensed under the MIT License.
