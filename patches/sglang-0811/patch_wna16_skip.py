"""补丁：moe_wna16 的 modules_to_not_convert 分支对非线性层返回错误的方法类。

当 "self_attn" 出现在 modules_to_not_convert 中时，RadixAttention（非 LinearBase）
也会命中 skip 分支并拿到 UnquantizedLinearMethod，随后
RadixAttention.__init__ 按注意力约定调用 create_weights(self) →
TypeError: missing 5 required positional arguments。

修复：skip 分支仅对 LinearBase 返回 UnquantizedLinearMethod、
对 FusedMoE 返回 UnquantizedFusedMoEMethod，其余层返回 None（不做量化处理）。
"""

P = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/quantization/moe_wna16.py"

s = open(P).read()

if "PATCH_WNA16_SKIP" in s:
    print("wna16 skip 补丁已存在，跳过")
    raise SystemExit(0)

old = """        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            if isinstance(layer, FusedMoE):
                return UnquantizedFusedMoEMethod()
            return UnquantizedLinearMethod()"""
new = """        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            # PATCH_WNA16_SKIP: 仅线性层可返回 UnquantizedLinearMethod，
            # RadixAttention 等其它层必须返回 None，否则 create_weights 签名不匹配
            if isinstance(layer, FusedMoE):
                return UnquantizedFusedMoEMethod()
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None"""

assert old in s, "skip 分支匹配失败"
open(P, "w").write(s.replace(old, new, 1))
print("wna16 skip 补丁已应用")
