#!/bin/bash

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ENABLE_MTP=false
export SGLANG_DSV4_INT8_KV_CACHE=false
export SGLANG_HACK_FLASHMLA_BACKEND=triton_logic
export SGLANG_USE_TRITON_MQA_LOGITS=false
export KV_CACHE_DTYPE=bfloat16

exec bash "$SCRIPT_DIR/run_ds_mtp.sh" "$@"
