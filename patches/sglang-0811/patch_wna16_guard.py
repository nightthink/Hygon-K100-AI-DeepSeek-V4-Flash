"""补丁：把 moe_wna16 加入 ROCm 支持的量化方法白名单。

原拦截（model_config.py::_verify_quantization 对 rocm_supported_quantization 的检查）
与海光自己在 moe_wna16.py 中为 HCU 放行 AWQ 的代码（`if not _is_hcu:`）直接矛盾
（轮次32）。配合 patch_moe_align.py（修复 align 垃圾输出）后，
Triton use_int4_w4a16 路径在 gfx928 上已验证数值正确（轮次35 L6/L4）。
"""

P = "/usr/local/lib/python3.10/dist-packages/sglang/srt/configs/model_config.py"

s = open(P).read()

if "PATCH_WNA16_GUARD" in s:
    print("wna16 闸门补丁已存在，跳过")
    raise SystemExit(0)

old = '''        rocm_supported_quantization = [
            "awq",
            "gptq",'''
new = '''        rocm_supported_quantization = [
            "moe_wna16",  # PATCH_WNA16_GUARD: gfx928 Triton wna16 路径已验证（配合 patch_moe_align）
            "awq",
            "gptq",'''

assert old in s, "白名单匹配失败"
open(P, "w").write(s.replace(old, new, 1))
print("wna16 闸门补丁已应用")
