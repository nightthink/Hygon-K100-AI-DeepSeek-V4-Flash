"""补丁：moe_wna16 下非专家线性层一律走 UnquantizedLinearMethod。

原逻辑把未被 modules_to_not_convert 覆盖的 LinearBase 路由到 AWQConfig，
需要 awq_dequantize/awq_gemm 等未编入本镜像 ROCm 构建的算子（轮次28/32）。
我方 int4 权重方案只量化路由专家（约 97% 字节），全部非专家线性层保持 bf16，
因此对 LinearBase 直接返回 Unquantized——也免去为 DSV4 每个模块维护
modules_to_not_convert 子串清单的脆弱性。
"""

P = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/quantization/moe_wna16.py"

s = open(P).read()

if "PATCH_WNA16_LINEAR" in s:
    print("wna16 linear 补丁已存在，跳过")
    raise SystemExit(0)

old = """        elif isinstance(layer, LinearBase):

            if self.linear_quant_method == "gptq":"""
new = """        elif isinstance(layer, LinearBase):
            # PATCH_WNA16_LINEAR: 非专家线性层全部保持 bf16（AWQ 线性算子
            # 未编入本镜像 ROCm 构建；本方案只量化路由专家）
            return UnquantizedLinearMethod()

            if self.linear_quant_method == "gptq":"""

assert old in s, "linear 分支匹配失败"
open(P, "w").write(s.replace(old, new, 1))
print("wna16 linear 补丁已应用")
