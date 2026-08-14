"""从 run_0811_probe.sh 派生 int4 启动器 run_0811_int4.sh。

改动：
  1. 启动前追加应用 4 个 wna16/align 补丁（patches/sglang-0811/）
  2. --quantization compressed-tensors → moe_wna16
  3. 追加 env SGLANG_DSV4_FP4_EXPERTS=0（int32 qweight 下强制关闭 fp4 专家自动检测）
其余（含全部 gfx928 env 与既有三补丁）逐字保持。

服务时另需挂载 int4 调优配置（mk_cfg_256.py 产出）到镜像的 configs/triton_3_5_0/ 路径。
"""

SRC = "/data1/sglang_patches/launchers_0811/run_0811_probe.sh"
DST = "/data1/sglang_patches/launchers_0811/run_0811_int4.sh"

s = open(SRC).read()

anchor = "python3 /data1/sglang_patches/launchers_0811/patch_dspark_torch_accept.py\n"
assert anchor in s, "找不到既有补丁锚点"
extra = (
    "python3 /data1/sglang_patches/launchers_0811/patch_moe_align.py\n"
    "python3 /data1/sglang_patches/launchers_0811/patch_wna16_guard.py\n"
    "python3 /data1/sglang_patches/launchers_0811/patch_wna16_skip.py\n"
    "python3 /data1/sglang_patches/launchers_0811/patch_wna16_linear.py\n"
    "export SGLANG_DSV4_FP4_EXPERTS=0\n"
)
s = s.replace(anchor, anchor + extra, 1)

old_q = "  --quantization compressed-tensors \\\n"
assert old_q in s, "找不到 quantization 参数"
s = s.replace(old_q, "  --quantization moe_wna16 \\\n", 1)

s = s.replace("/tmp/logs/probe_0811.log", "/tmp/logs/int4_0811.log")

open(DST, "w").write(s)
print("已生成", DST)
