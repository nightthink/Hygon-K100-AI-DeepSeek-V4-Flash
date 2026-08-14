#!/bin/bash
# 0811 镜像（sglang 0.5.15 + DSpark）在 gfx928 上的探针启动器
# 关键：关闭 tilelang MHC / deepgemm prenorm，走纯 torch 回退路径（与 0728 线一致）
set -o pipefail

# --- 同步加载补丁（修多线程 H2D 死锁），直接在容器内改 ---
UTILS=/usr/local/lib/python3.10/dist-packages/sglang/srt/model_loader/utils.py
if ! grep -q "PATCH_SYNC_LOAD" "$UTILS"; then
  python3 - <<'PY'
p = "/usr/local/lib/python3.10/dist-packages/sglang/srt/model_loader/utils.py"
s = open(p).read()
old = '    device = getattr(weight, "device", None)\n    if device is None:\n        return False\n    return device.type == "cpu"'
new = '    # PATCH_SYNC_LOAD: DTK 26.04 HIP multi-thread H2D deadlock workaround\n    return False'
assert old in s, "pattern not found"
open(p, "w").write(s.replace(old, new))
print("sync-load patch applied")
PY
fi

python3 /data1/sglang_patches/launchers_0811/patch_triton_backend.py
python3 /data1/sglang_patches/launchers_0811/patch_dflash_renorm.py
ulimit -l unlimited

# --- 官方必需 env（沿用 0728 线实测有效的一整套）---
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
export SGLANG_USE_JIT_ALL_REDUCE=0
export FORCE_TORCH_AR=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1

# --- gfx928 关键回退：两条 MHC 加速路径全关，走 hc_pre_torch_impl ---
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_USE_AITER_MHC_PRE=0
export SGLANG_OPT_USE_AITER_MHC_POST=0
export SGLANG_OPT_USE_TILELANG_INDEXER=0
export SGLANG_DSV4_MHC_PREWARM=0
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# --- MoE / 算子 ---
export SGLANG_USE_FP8_W8A8_MOE=false
export SGLANG_GROUPGEMM=true
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_LIGHTOP_TOPK=true
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_NSA_FUSE_TOPK=false
export SGLANG_APPLY_CONFIG_BACKUP=none

# --- DSv4 注意力 ---
export SGLANG_DSV4_MODE=2604
export SGLANG_DSV4_INT8_KV_CACHE=true
export SGLANG_HACK_FLASHMLA_BACKEND=${FLASHMLA_BACKEND:-triton}
export SGLANG_USE_TRITON_MQA_LOGITS=false
export SGLANG_DSV4_DEEPEP_TP_SHARD_QUANT=0

MODEL_PATH=${MODEL_PATH:-/models}
PORT=${PORT:-8000}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
PREFILL_CHUNK=${PREFILL_CHUNK:-4096}
SPEC_ALGO=${SPEC_ALGO:-none}

MAXRUN_ARGS=()
[ -n "$MAX_RUNNING" ] && MAXRUN_ARGS=(--max-running-requests "$MAX_RUNNING")

SPEC_ARGS=()
case "${SPEC_ALGO,,}" in
  dspark)
    SPEC_ARGS=(--speculative-algorithm DSPARK)
    [ -n "$DSPARK_BLOCK" ] && SPEC_ARGS+=(--speculative-dspark-block-size "$DSPARK_BLOCK")
    ;;
  eagle)
    SPEC_ARGS=(--speculative-algorithm EAGLE
               --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}"
               --speculative-eagle-topk 1
               --speculative-num-draft-tokens "$(( ${SPECULATIVE_NUM_STEPS:-3} + 1 ))")
    ;;
esac

echo "=== probe config: spec=$SPEC_ALGO backend=$SGLANG_HACK_FLASHMLA_BACKEND model=$MODEL_PATH ==="
mkdir -p /tmp/logs

sglang serve \
  --port "$PORT" \
  --trust-remote-code \
  --model-path "$MODEL_PATH" \
  --tp 8 \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --served-model-name deepseek-v4-flash \
  --schedule-policy fcfs \
  --moe-a2a-backend none \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --chunked-prefill-size "$PREFILL_CHUNK" \
  --quantization compressed-tensors \
  --kv-cache-dtype auto \
  --disable-flashinfer-autotune \
  --disable-custom-all-reduce \
  "${SPEC_ARGS[@]}" \
  "${MAXRUN_ARGS[@]}" \
  2>&1 | tee -a /tmp/logs/probe_0811.log
