#!/bin/bash
# 验证假设：DSpark 非贪心路径需要先单请求预热（触发 triton JIT 编译），再上并发才安全。
# 此前所有非贪心并发测试，8 个请求都是该路径的首批请求，等于 8 路并发同时 JIT。
#
# 【实测结果，2026-08-14，8×K100-AI / gfx928 / 0811 镜像 + 0731 W8A8 + DSpark】
#   步骤 3（2 并发）：✅ 通过（此前冷启动也能过）
#   步骤 4（4 并发）：✅ 通过，32.4 tok/s —— 冷启动时同配置必崩，预热确实有效
#   步骤 5（8 并发）：❌ 仍失败，但失效形态从 HSA_STATUS_ERROR_EXCEPTION + watchdog 超时
#                     降级为软挂起（进程存活、无内核异常、请求不返回）
# 结论：崩溃与该路径的"首次并发触发"强相关，而非稳态计算的越界访存。
#
# 依赖：bench_conc_param.py（同目录），服务已在 127.0.0.1:8000 就绪。
set -x

echo "=== 步骤 1：贪心预热（走已知稳定路径）==="
timeout 300 curl -s -o /dev/null -w 'greedy warmup: %{http_code}\n' \
  http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"temperature":0}'

echo "=== 步骤 2：非贪心单请求预热（关键：让 JIT 串行完成）==="
timeout 300 curl -s -o /dev/null -w 'nongreedy warmup 1: %{http_code}\n' \
  http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"介绍一下人工智能"}],"max_tokens":48,"temperature":0.7}'

timeout 300 curl -s -o /dev/null -w 'nongreedy warmup 2: %{http_code}\n' \
  http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"解释一下数据库索引"}],"max_tokens":48,"temperature":0.7,"top_p":0.95}'

echo "=== 步骤 3：非贪心 2 并发（渐进升温）==="
python3 -u "$(dirname "$0")/bench_conc_param.py" 2 60 0.7 120

echo "=== 步骤 4：非贪心 4 并发 ==="
python3 -u "$(dirname "$0")/bench_conc_param.py" 4 60 0.7 120

echo "=== 步骤 5：非贪心 8 并发（冷启动时必崩）==="
python3 -u "$(dirname "$0")/bench_conc_param.py" 8 60 0.7 120

echo "=== 步骤 6：健康检查 ==="
curl -s -o /dev/null -w 'health: %{http_code}\n' --max-time 20 http://127.0.0.1:8000/health
echo WARMUP_TEST_DONE
