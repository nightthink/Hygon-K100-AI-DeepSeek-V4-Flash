#!/bin/bash
# DeepSeek-V4-Flash @ 8x K100-AI vLLM 服务验收脚本
# 用法: bash test_dv4.sh [host]   默认 127.0.0.1
HOST="${1:-127.0.0.1}"
BASE="http://${HOST}:8000"
M="deepseek-v4-flash"
 
echo "========== 1. 健康检查 =========="
curl -s -o /dev/null -w "GET /health -> HTTP %{http_code}\n" ${BASE}/health
 
echo ""
echo "========== 2. 模型列表 =========="
curl -s ${BASE}/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin); print('models:', [m['id'] for m in d['data']])"
 
echo ""
echo "========== 3. 基础对话 =========="
curl -s ${BASE}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$M"'",
  "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
  "max_tokens": 128
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print('---'); print('usage:', d['usage'])"
 
echo ""
echo "========== 4. 数学正确性抽查（INT8 量化精度）=========="
curl -s ${BASE}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$M"'",
  "messages": [{"role": "user", "content": "计算 37 * 89 - 156，只给出最终数字。"}],
  "max_tokens": 512,
  "temperature": 0
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('回答:', d['choices'][0]['message']['content'].strip()); print('期望包含: 3137')"
 
echo ""
echo "========== 5. 代码能力抽查 =========="
curl -s ${BASE}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$M"'",
  "messages": [{"role": "user", "content": "用Python写一个判断素数的函数，只给代码。"}],
  "max_tokens": 256,
  "temperature": 0
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
 
echo ""
echo "========== 6. 流式输出 =========="
curl -sN ${BASE}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$M"'",
  "messages": [{"role": "user", "content": "从1数到10，用顿号分隔。"}],
  "max_tokens": 64,
  "stream": true
}' | head -20
echo ""
 
echo ""
echo "========== 7. 解码速度粗测 =========="
python3 - <<PYEOF
import json, time, urllib.request
req = urllib.request.Request(
    "${BASE}/v1/chat/completions",
    data=json.dumps({
        "model": "$M",
        "messages": [{"role": "user", "content": "写一段200字左右介绍人工智能发展史的文字。"}],
        "max_tokens": 256, "temperature": 0.7,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
t0 = time.time()
resp = json.load(urllib.request.urlopen(req, timeout=600))
dt = time.time() - t0
u = resp["usage"]
print(f"prompt_tokens={u['prompt_tokens']}  completion_tokens={u['completion_tokens']}  耗时={dt:.1f}s")
print(f"解码速度约 {u['completion_tokens']/dt:.1f} tokens/s（单请求）")
PYEOF
 
echo ""
echo "========== 完成 =========="
 
