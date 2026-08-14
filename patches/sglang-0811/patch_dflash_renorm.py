"""移植上游修复：DSpark 在 temperature>0 时 top_k/top_p_renorm_prob 为 None 崩溃。

来源：HYGON-AI/sglang-das 提交 18e167b73（2026-08-13）
      "[BugFix] fix the client inference None errors in temperature>0 with a torch implementation"

现象（我方 gfx928 实测）：
    File ".../speculative/dflash_utils.py", line 784, in build_dflash_verify_target_probs
        target_probs = top_k_renorm_prob(...)
    TypeError: 'NoneType' object is not callable

根因：ROCm/HCU 上 sgl_kernel 只暴露 Python 包装器、不含 native 的
top_{k,p}_renorm_probs 算子，模块顶部的 import 失败后把两者置为 None，
DSpark 的非贪心验证路径一旦 temperature != 0 就直接崩。

修复：补上 torch 版实现（复用 pytorch 采样后端已有的逻辑），按"能力"而非"平台"判定。
tree_speculative_sampling_target_only 无 torch 等价实现，仍由
_DFLASH_SAMPLING_VERIFY_AVAILABLE 单独把关，行为不变。
"""

P = "/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_utils.py"
MARKER = "PATCH_DFLASH_RENORM"

ANCHOR = "def is_dflash_sampling_verify_available() -> bool:"

ADDITION = '''def _top_p_renorm_prob_torch(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of sgl_kernel.top_p_renorm_prob."""
    from sglang.srt.layers.sampler import top_p_normalize_probs_torch

    return top_p_normalize_probs_torch(probs, top_ps)


def _top_k_renorm_prob_torch(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of sgl_kernel.top_k_renorm_prob."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


# PATCH_DFLASH_RENORM: ROCm/HCU 上 sgl_kernel 无 native top_{k,p}_renorm_probs，
# 上面的 import 会把它们留成 None，DSpark 非贪心验证路径 temperature != 0 即崩。
# 回退到 torch 实现（与 pytorch 采样后端一致）。按能力判定，不按平台判定。
if top_p_renorm_prob is None:
    top_p_renorm_prob = _top_p_renorm_prob_torch
if top_k_renorm_prob is None:
    top_k_renorm_prob = _top_k_renorm_prob_torch


'''


def main():
    src = open(P).read()
    if MARKER in src:
        print("dflash renorm patch: already patched")
        return
    if ANCHOR not in src:
        raise SystemExit("PATCH FAILED: anchor not found in " + P)
    open(P, "w").write(src.replace(ANCHOR, ADDITION + ANCHOR, 1))
    print("dflash renorm patch applied")


if __name__ == "__main__":
    main()
