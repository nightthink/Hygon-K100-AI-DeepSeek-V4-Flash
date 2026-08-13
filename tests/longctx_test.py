import json, time, urllib.request, sys

# 用法: python3 longctx_test.py [重复次数]  每份约 40 token, 400 次约 16K token
n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
para = "人工智能的发展经历了符号主义、连接主义与深度学习三个主要阶段，每个阶段都伴随着算力、数据与算法的协同进步。"
long_text = (para * n) + "\n\n请用一句话总结上文的核心观点，并数一下上文大约重复了多少遍同一段话。"

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": long_text}],
        "max_tokens": 128, "temperature": 0,
    }).encode(),
    headers={"Content-Type": "application/json"})

t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=1800))
dt = time.time() - t0
u = r["usage"]
print("prompt_tokens=%d completion=%d 总耗时=%.1fs prefill速率≈%.0f tok/s" % (
    u["prompt_tokens"], u["completion_tokens"], dt, u["prompt_tokens"]/dt))
print("回答:", r["choices"][0]["message"]["content"].strip()[:150])
