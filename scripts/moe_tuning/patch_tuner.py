"""修正 sglang MoE 调优器对"每通道量化"的处理。

问题：`common_utils.get_model_config` 见到 compressed-tensors 的 `config_groups` 就无条件
取 `weights.group_size` 组成 `block_shape=[0, group_size]`。我方是 **per-channel 对称量化**，
`group_size` 为 null，于是 `block_shape=[0, None]`，随后

    tuning_fused_moe_triton.py:
        if block_k % config["BLOCK_SIZE_K"] == 0

拿 None 做取模，抛：

    TypeError: unsupported operand type(s) for %: 'NoneType' and 'int'

正确语义：per-channel 量化没有 block/group 概念，`block_shape` 应为 None（不做搜索空间过滤）。

用法（在放有 tuning_fused_moe_triton.py / common_utils.py 的目录里）：
    python3 patch_tuner.py

对应上游文件：
    benchmark/kernels/fused_moe_triton/common_utils.py
    benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
"""

import os

D = os.path.dirname(os.path.abspath(__file__))

CU = os.path.join(D, "common_utils.py")
MARK = "PATCH_PERCHANNEL_BLOCK_SHAPE"

src = open(CU, encoding="utf-8").read()
if MARK in src:
    print("common_utils.py already patched")
else:
    old = """        group_size = weights_config.get("group_size")
        block_shape = [0, group_size]
        assert len(block_shape) == 2"""
    new = """        group_size = weights_config.get("group_size")
        # PATCH_PERCHANNEL_BLOCK_SHAPE: per-channel 量化没有 group，group_size 为 None，
        # 此时 block_shape 必须保持 None，否则下游用 None 做取模会 TypeError。
        if group_size is None:
            block_shape = None
        else:
            block_shape = [0, group_size]
            assert len(block_shape) == 2"""
    assert old in src, "pattern not found in common_utils.py"
    open(CU, "w", encoding="utf-8").write(src.replace(old, new))
    print("common_utils.py patched")

TU = os.path.join(D, "tuning_fused_moe_triton.py")
src = open(TU, encoding="utf-8").read()
if "PATCH_BLOCK_K_GUARD" in src:
    print("tuning_fused_moe_triton.py already patched")
else:
    old = """        if block_shape is not None:
            block_k = block_shape[1]"""
    new = """        # PATCH_BLOCK_K_GUARD: 双保险，block_shape[1] 为 None 时不过滤搜索空间。
        if block_shape is not None and block_shape[1] is not None:
            block_k = block_shape[1]"""
    assert old in src, "pattern not found in tuning_fused_moe_triton.py"
    open(TU, "w", encoding="utf-8").write(src.replace(old, new))
    print("tuning_fused_moe_triton.py patched")
