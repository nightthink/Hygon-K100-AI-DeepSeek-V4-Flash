"""补丁：修复 gfx928 上 moe_align_block_size 产出垃圾导致 fused_moe_triton 全家族 VM fault。

根因（轮次35，经验二分定位）：
  1. 默认的 sgl_moe_align_block_size（sgl_kernel C++ 版）在本卡输出全垃圾
     （num_post 为负数、expert_ids 为未初始化内存）——即长期刷屏的
     "Launch params (1024,1,1) > launch bounds (256)" 警告，在本卡是真 UB。
  2. lightop.op.moe_align_block_size 排序正确，但不填充 pad 槽位；
     残留负垃圾会骗过内核的 token_mask（负数 < num_valid_tokens），
     使内核按负偏移读 A → VM fault。

修复：默认分支改用 lightop 算子，且 sorted_ids 预填充为 topk_ids.numel()
（内核以该值识别 pad 槽）。已验证：直调 fused_moe_kernel_gptq_awq
相对误差 0.003（与手工 torch 排序的黄金结果一致）。

统一解释并修复三处旧故障：轮次28 EP 朴素 dispatcher 段错误、
AWQ 工装 bf16 基线崩溃、轮次33 int4 内核 fault。
"""

P = ("/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/moe/"
     "moe_runner/triton_utils/moe_align_block_size.py")

s = open(P).read()

if "PATCH_MOE_ALIGN_GFX928" in s:
    print("moe_align 补丁已存在，跳过")
    raise SystemExit(0)

old = """    else:
        sgl_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
            ignore_invalid_expert,
        )
    return sorted_ids, expert_ids, num_tokens_post_pad"""

new = """    else:
        # PATCH_MOE_ALIGN_GFX928: sgl_kernel 版在 gfx928 上输出垃圾
        # （launch bounds 超限 → UB）。改用 lightop 版并预填充 pad 槽，
        # 否则残留负值会骗过内核 token_mask 造成 VM fault。
        import lightop.op as _lop

        sorted_ids.fill_(topk_ids.numel())
        _lop.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
    return sorted_ids, expert_ids, num_tokens_post_pad"""

assert old in s, "moe_align 补丁匹配失败——文件内容与预期不符"
open(P, "w").write(s.replace(old, new, 1))
print("moe_align gfx928 补丁已应用")
