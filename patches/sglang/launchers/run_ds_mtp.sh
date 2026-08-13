#!/bin/bash

set -o pipefail

export SGLANG_TORCH_PROFILER_DIR="/profiling"

ulimit -l unlimited
######## 20260511更新 ##########
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3

# 算子优化
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_USE_FP8_W8A8_MOE=false # 模型专家权重是 INT8，不启用 FP8 MoE 分支
unset SGLANG_DEEPEP_BF16_DISPATCH
export SGLANG_GROUPGEMM=true

export SGLANG_USE_LIGHTOP=1 # 开启 lightop的 topK算子(fused_moe_gate)/moe_align算子/moe_sum算子
export SGLANG_ROCM_USE_AITER_MOE=false  # AITER moe_sorting_ck 未包含 gfx928 device function
######## 20260511更新 ##########

export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_LIGHTOP_TOPK=true
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_NSA_FUSE_TOPK=${NSA_FUSE_TOPK:-false}  # temp: bypass lightop fused topk to isolate crash
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

MODEL_PATH=/models
PORT=${PORT:-30001}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-64}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.725}
PREFILL_CHUNK=${PREFILL_CHUNK:-8192}
SCHED_POLICY=${SCHED_POLICY:-fcfs}
ENABLE_MTP=${ENABLE_MTP:-true}
SPECULATIVE_NUM_STEPS=${SPECULATIVE_NUM_STEPS:-3}
SPEC_TOPK=${SPEC_TOPK:-1}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}

SPECULATIVE_ARGS=()
case "${ENABLE_MTP,,}" in
  1|true|yes|on)
    ENABLE_MTP=true
    if ! [[ "$SPECULATIVE_NUM_STEPS" =~ ^[1-9][0-9]*$ ]]; then
      printf 'Invalid SPECULATIVE_NUM_STEPS=%q; expected a positive integer.\n' \
        "$SPECULATIVE_NUM_STEPS" >&2
      exit 2
    fi
    SPECULATIVE_NUM_DRAFT_TOKENS=${SPEC_DRAFT:-$((SPECULATIVE_NUM_STEPS * SPEC_TOPK + 1))}
    SPECULATIVE_ARGS=(
      --speculative-algorithm EAGLE
      --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
      --speculative-eagle-topk "$SPEC_TOPK"
      --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
    )
    DECODE_MODE="mtp_steps${SPECULATIVE_NUM_STEPS}"
    ;;
  0|false|no|off)
    ENABLE_MTP=false
    DECODE_MODE=target_only
    ;;
  *)
    printf 'Invalid ENABLE_MTP=%q; expected true or false.\n' "$ENABLE_MTP" >&2
    exit 2
    ;;
esac

export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_INT8_KV_CACHE="${SGLANG_DSV4_INT8_KV_CACHE:-true}"
export SGLANG_HACK_FLASHMLA_BACKEND="${SGLANG_HACK_FLASHMLA_BACKEND:-triton}"
export SGLANG_USE_TRITON_MQA_LOGITS="${SGLANG_USE_TRITON_MQA_LOGITS:-false}"

case "$KV_CACHE_DTYPE" in
  bf16|bfloat16)
    if [[ "${SGLANG_DSV4_INT8_KV_CACHE,,}" =~ ^(1|true|yes|on)$ ]]; then
      printf 'Invalid KV cache combination: KV_CACHE_DTYPE=%s requires SGLANG_DSV4_INT8_KV_CACHE=false.\n' \
        "$KV_CACHE_DTYPE" >&2
      exit 2
    fi
    ;;
esac

# Prefer locally tuned MoE configs when they have been generated.
MOE_CONFIG_ROOT="$PWD/moe_configs"
if [[ -d "$MOE_CONFIG_ROOT/configs" ]]; then
  export SGLANG_MOE_CONFIG_DIR="$MOE_CONFIG_ROOT"
fi

export SGLANG_DSV4_DEEPEP_TP_SHARD_QUANT=0
export FORCE_TORCH_AR=1
# DeepSeekV4 CP uses attention TP ranks as context-parallel ranks.
# With --tp 8 and --enable-nsa-prefill-context-parallel, cp_size is 8.

mkdir -p "$PWD/logs"
printf 'MoE configuration: quantization=INT8-W8A8-per-channel, aiter=%s, fp8_moe=%s\n' \
  "$SGLANG_ROCM_USE_AITER_MOE" "$SGLANG_USE_FP8_W8A8_MOE"
printf 'Attention configuration: flashmla_backend=%s, kv_cache_dtype=%s, int8_kv=%s\n' \
  "$SGLANG_HACK_FLASHMLA_BACKEND" "$KV_CACHE_DTYPE" "$SGLANG_DSV4_INT8_KV_CACHE"
printf 'Indexer configuration: triton_mqa_logits=%s\n' \
  "$SGLANG_USE_TRITON_MQA_LOGITS"
if [[ "$ENABLE_MTP" == true ]]; then
  printf 'Speculative configuration: algorithm=EAGLE, steps=%s, topk=1, draft_tokens=%s\n' \
    "$SPECULATIVE_NUM_STEPS" "$SPECULATIVE_NUM_DRAFT_TOKENS"
else
  printf 'Speculative configuration: disabled (target model only)\n'
fi

sglang serve \
  --port "$PORT" \
  --trust-remote-code \
  --model-path $MODEL_PATH \
  --tp 8 \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --served-model-name deepseek-v4-flash \
  --schedule-policy "$SCHED_POLICY" \
  --nsa-prefill-cp-mode round-robin-split \
  --moe-a2a-backend none \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --chunked-prefill-size "$PREFILL_CHUNK" \
  --quantization compressed-tensors \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --disable-flashinfer-autotune \
  "${SPECULATIVE_ARGS[@]}" \
  2>&1 | tee -a "$PWD/logs/sglang_test_${DECODE_MODE}_${SGLANG_HACK_FLASHMLA_BACKEND}_${KV_CACHE_DTYPE}_$(date +%Y%m%d_%H%M%S).log"
