"""可选补丁：把 DSpark accept 内核的分发从 triton 强制切到 torch 参考实现。

动机（我方 gfx928 实测）：DSpark 非贪心接受路径的 triton 内核在批量增大时触发
    HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
随后 watchdog 300s 超时、服务不可用。贪心路径（纯 torch）完全稳定。

镜像里每个 accept 内核（AcceptSampling / SelectMixedAccept / AcceptGreedy /
FinalizeAcceptLens / CapCorrectLen / SoftmaxTemp ...）都自带 torch 参考实现，
由 kernels/dispatch.py::inputs_on_cuda 按"张量在不在 GPU 上"分发——GPU 张量一律走
triton。本补丁让它恒返 False，从而全部走 torch 实现。

代价：accept 这一步从融合内核退化为若干 torch 算子，单步开销上升（这些张量都很小：
bs × (gamma+1)，相对整个 forward 可忽略，但仍需实测确认）。
收益：预期绕开崩溃的 triton 内核，让 temperature>0 的并发请求可用。

【实测结论（2026-08-14，8×K100-AI）】启用本补丁后 **仍然崩溃**，现象不变。
因此诱因不限于 accept_sampling_triton 这一个内核，本补丁仅作为诊断手段保留，
不建议作为生产规避方案。详见 docs/Bug报告-DSpark与triton路由.md 第二节。

启用方式：设环境变量 SGLANG_DSPARK_FORCE_TORCH_ACCEPT=1（默认不启用）。
"""

import os

P = "/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dspark_components/kernels/dispatch.py"
MARKER = "PATCH_DSPARK_FORCE_TORCH"

ADDITION = '''
# PATCH_DSPARK_FORCE_TORCH: gfx928 上 DSpark 非贪心 accept 的 triton 内核在批量增大时
# 触发 HSA_STATUS_ERROR_EXCEPTION(0x1016)。恒返 False 使全部 accept 内核走 torch 参考实现。
import os as _os

_force_torch = _os.environ.get("SGLANG_DSPARK_FORCE_TORCH_ACCEPT", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

if _force_torch:
    _orig_inputs_on_cuda = inputs_on_cuda

    def inputs_on_cuda(*args, **kwargs) -> bool:  # noqa: F811
        return False
'''


def main():
    if os.environ.get("SGLANG_DSPARK_FORCE_TORCH_ACCEPT", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        print("dspark torch-accept patch: 未启用（SGLANG_DSPARK_FORCE_TORCH_ACCEPT 未设）")
        return
    src = open(P).read()
    if MARKER in src:
        print("dspark torch-accept patch: already patched")
        return
    open(P, "w").write(src + ADDITION)
    print("dspark torch-accept patch applied (accept 内核走 torch 参考实现)")


if __name__ == "__main__":
    main()
