#!/bin/bash

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ENABLE_MTP=false
export SGLANG_DSV4_INT8_KV_CACHE=false
export SGLANG_HACK_FLASHMLA_BACKEND=torch_native
export SGLANG_USE_TRITON_MQA_LOGITS=false
export KV_CACHE_DTYPE=bfloat16
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-1}"

exec bash "$SCRIPT_DIR/run_ds_mtp.sh" "$@"
