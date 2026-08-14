#!/bin/bash
# 顺序跑多组配置对照，结果写到 /tmp/three_result.txt。轮次 31 用它一次跑完四组。
#
# 设计要点（都是踩过坑之后加的）：
#   1. **必带基线对照组**（不加任何额外参数）——确认当日机器状态，否则无从判断差异来源
#   2. **贪心热身两次，且用完整长度的请求**——8 token 的热身覆盖不了 400 token 生成路径
#      上的全部内核；本项目已四次因 JIT 污染误读数据（首测约 12 tok/s，恰好接近"无投机"
#      的 12.3，极易误判为投机没生效）
#   3. **预填每组换不同前缀**——避开前缀缓存（曾测得 2144 tok/s 的假值）
#   4. **记录 accept rate**——它是数值正确性的敏感指标：轮次 31 强开共享专家融合后
#      accept rate 从 0.62 崩到 0.03，是输出乱码的先行信号
#
# ⚠️ 本脚本只测速度。任何显示出增益的配置都必须再过一遍数学 / Think / Tool Call 验收
#    才能采信——本项目已两次遇到"启动正常、无报错、数值静默错误"。
#
# 用法：编辑末尾的 run_one 调用，然后 nohup bash run_ab_experiments.sh &
set -uo pipefail
OUT=${OUT:-/tmp/three_result.txt}
IMG=${IMG:-harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk2604-py3.10-20260804-0006-deepseekV4-0811}
MODEL=${MODEL:-/data1/models/dsv4-0731-w8a8-dspark}
LAUNCHER=${LAUNCHER:-run_0811_flex.sh}   # 需支持 EXTRA_ARGS，构造方式见 scripts/moe_tuning/run_0811_ep.md
: > $OUT

run_one () {
  local name="$1"; local extra="$2"
  echo "" >> $OUT
  echo "########## $name ##########" >> $OUT
  echo "EXTRA_ARGS=[$extra]" >> $OUT
  echo "[$(date '+%H:%M:%S')] 启动中" >> $OUT

  docker rm -f sglang-dsv4 >/dev/null 2>&1
  sleep 5
  docker run -d --name sglang-dsv4 --network=host --ipc=host --ulimit memlock=-1 \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /opt/hyhal:/opt/hyhal -v /data1:/data1 -v "$MODEL":/models \
    -e NCCL_P2P_DISABLE=1 -e PORT=8000 -e SPEC_ALGO=dspark \
    -e MEM_FRACTION_STATIC=0.85 -e CUDA_GRAPH_MAX_BS=16 -e PREFILL_CHUNK=4096 \
    -e EXTRA_ARGS="$extra" \
    -w /data1/sglang_patches/launchers_0811 --entrypoint bash \
    "$IMG" "$LAUNCHER" >/dev/null 2>&1

  local code=""
  for i in $(seq 1 70); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:8000/health 2>/dev/null)
    [ "$code" = "200" ] && break
    sleep 20
  done
  echo "[$(date '+%H:%M:%S')] health=$code" >> $OUT
  if [ "$code" != "200" ]; then
    echo "结果: 启动失败 / 未就绪" >> $OUT
    docker logs --tail 400 sglang-dsv4 2>&1 | tr '\r' '\n' \
      | grep -iE 'error|unrecognized|invalid|Traceback|not support' \
      | grep -viE 'multimem' | sort -u | head -4 >> $OUT
    return
  fi

  # 贪心热身两次，完整长度
  for i in 1 2; do
    curl -s -o /dev/null --max-time 120 http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"用 Python 写一个 LRU 缓存类，带注释。"}],"max_tokens":400,"temperature":0}'
  done

  STAMP="$name" python3 - >> $OUT 2>&1 <<'PY'
import json, os, time, urllib.request
U = "http://127.0.0.1:8000/v1/chat/completions"

def call(p, to=120):
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        U, json.dumps(p).encode(), {"Content-Type": "application/json"}), timeout=to))
    return r, time.time() - t

r, d = call({"model": "deepseek-v4-flash",
             "messages": [{"role": "user", "content": "用 Python 写一个 LRU 缓存类，带注释。"}],
             "max_tokens": 400, "temperature": 0})
print(f"解码单流: {r['usage']['completion_tokens']/d:.2f} tok/s")

# 每组换前缀，避开前缀缓存；max_tokens=1 让耗时基本等于 prefill
tag = os.environ["STAMP"]
fill = " ".join([f"{tag}第{i}节 分布式系统的一致性协议需要在可用性与分区容忍之间权衡。"
                 for i in range(400)])
r, d = call({"model": "deepseek-v4-flash",
             "messages": [{"role": "user", "content": fill + " 请只回答OK。"}],
             "max_tokens": 1, "temperature": 0})
pt = r["usage"]["prompt_tokens"]
print(f"预填充: {pt} tok / {d:.2f}s = {pt/d:.2f} tok/s")
PY

  echo "-- accept rate（数值正确性的敏感指标）--" >> $OUT
  docker logs --since 4m sglang-dsv4 2>&1 | tr '\r' '\n' \
    | grep -oE 'accept rate: [0-9.]+' | tail -2 >> $OUT
  echo "-- 关键日志 --" >> $OUT
  docker logs sglang-dsv4 2>&1 | tr '\r' '\n' \
    | grep -iE 'shared expert|moe_runner_backend|Disable prefill CUDA graph|Capture prefill' \
    | sed 's/^\(.\{125\}\).*/\1/' | sort -u | head -4 >> $OUT
}

# 轮次 31 实际跑的四组（结论：后三组全否，详见 docs/调优记录-轮次31.md）
run_one "A_基线" ""
run_one "B_共享专家融合" "--enforce-shared-experts-fusion"
run_one "C_MoE_runner_triton_kernel" "--moe-runner-backend triton_kernel"
run_one "D_预填充cuda_graph" "--cuda-graph-backend-prefill full"

echo "" >> $OUT
echo "[$(date '+%H:%M:%S')] 全部完成" >> $OUT
