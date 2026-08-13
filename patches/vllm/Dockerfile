FROM <internal-harbor>/dcu/vllm-ubuntu22.04-dtk26.04-hy3-0706:latest
# DeepSeek-V4-Flash K100-AI(gfx928) 6 补丁直接固化（原以 -v 挂载方式使用）
# 根因与 diff 见随包文档《在8张K100-AI上运行DeepSeek-V4-Flash：从零到成果》第 2.2 节
COPY mhc_patched.py             /usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/mhc.py
COPY dsv4_attn_patched.py       /usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/deepseek_v4_attention.py
COPY mla_sparse_patched.py      /usr/local/lib/python3.10/dist-packages/vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py
COPY vllm_mla_sparse_patched.py /usr/local/lib/python3.10/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
COPY kv_cache_utils_patched.py  /usr/local/lib/python3.10/dist-packages/vllm/v1/core/kv_cache_utils.py
COPY worker_utils_patched.py    /usr/local/lib/python3.10/dist-packages/vllm/v1/worker/utils.py
LABEL maintainer="lzd" \
      base="vllm-ubuntu22.04-dtk26.04-hy3-0706" \
      patches="dsv4-6patch-20260812" \
      description="DeepSeek-V4-Flash on 8xK100-AI, 6 patches baked (mhc PDL / attn C-op bypass / indexer FP8 / wo_a transposed view / logging fixes)"
