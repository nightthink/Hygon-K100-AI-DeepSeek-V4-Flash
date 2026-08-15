"""派生 run_0811_prodfix.sh：在生产 w8a8 启动器基础上加 moe_align 根修复补丁。

轮次38 发现：生产线 run_0811_probe.sh 从不应用 patch_moe_align.py，
DSpark verify 图捕获会触发 sgl C++ moe_align 的 launch-bounds 未定义行为，
之前 4 小时是 UB 侥幸没崩，重启后同一处崩（launch bounds 1024>256）。
补上该补丁（与 int4 线同款）使该路径确定性正确，重启不再刷 launch-bounds。
只在补丁段末尾插一行，其余逐字不动，仅改日志名。

注：这只修掉"坑 A"（moe_align UB）。w8a8 int8 验证图捕获还有独立的
hipErrorInvalidValue（坑 B，见 docs/调优记录-轮次38.md），需配置绕开。
"""
SRC = "/data1/sglang_patches/launchers_0811/run_0811_probe.sh"
DST = "/data1/sglang_patches/launchers_0811/run_0811_prodfix.sh"

s = open(SRC).read()
anchor = "python3 /data1/sglang_patches/launchers_0811/patch_dspark_torch_accept.py\n"
assert anchor in s, "找不到补丁段锚点"
add = "python3 /data1/sglang_patches/launchers_0811/patch_moe_align.py\n"
if add not in s:
    s = s.replace(anchor, anchor + add, 1)
s = s.replace("/tmp/logs/probe_0811.log", "/tmp/logs/prodfix_0811.log")

open(DST, "w").write(s)
print("已生成", DST)
for line in s.splitlines():
    if "patch_" in line:
        print("   ", line)
