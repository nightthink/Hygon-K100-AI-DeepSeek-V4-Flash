"""把 triton 稀疏解码路径接入 0811 镜像的 dsv4 注意力后端（gfx928 可用）。

背景：0.5.15 中 HCU 平台走 DeepseekV4AttnBackend，其 _call_flash_mla_with_kvcache
调用的是 debug_flash_mla_adapter.flash_mla_with_kvcache_entrypoint，而该入口只认
"kernel"（flash_mla 内核，运行时拒绝 gfx928：Dense decode MLA only on gfx936/938）
与 "torch"（纯 PyTorch，实测 1.26 tok/s 不可用）。

镜像里其实自带 triton 稀疏解码内核（sglang/srt/layers/attention/nsa/triton_decode），
其入口 triton_fp8_attention_fwd 显式声明"接受与 flash_mla 相同的 kwargs，多余键忽略"，
是 drop-in 设计。本补丁在 debug 适配器里加一条 triton 分支即可。
"""

P = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/debug_flash_mla_adapter.py"

MARKER = "PATCH_TRITON_BACKEND"
OLD = '''def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if backend in {"torch", "native", "torch_native"}:
        return torch_native_flash_mla_with_kvcache(**kwargs)

    assert backend == "kernel", f"unsupported backend {backend!r}"'''

NEW = '''def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if backend in {"torch", "native", "torch_native"}:
        return torch_native_flash_mla_with_kvcache(**kwargs)

    # PATCH_TRITON_BACKEND: gfx928 上 flash_mla 内核拒绝 dense decode MLA，
    # 改走镜像自带的 triton 稀疏解码内核（同一 kwargs 约定）。
    if backend in {"triton", "triton_logic"}:
        from sglang.srt.layers.attention.nsa.triton_decode import (
            triton_fp8_attention_fwd,
        )

        return triton_fp8_attention_fwd(**kwargs)

    assert backend == "kernel", f"unsupported backend {backend!r}"'''


def main():
    src = open(P).read()
    if MARKER in src:
        print("already patched")
        return
    if OLD not in src:
        raise SystemExit("PATCH FAILED: anchor not found in " + P)
    open(P, "w").write(src.replace(OLD, NEW))
    print("triton backend patch applied")


if __name__ == "__main__":
    main()
