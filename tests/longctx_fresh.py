"""长上下文 prefill 测速，每次用不同前缀避免命中前缀缓存。

用法: python3 longctx_fresh.py [重复次数] [标签]
"""
import json, time, urllib.request, sys, hashlib

n = int(sys.argv[1]) if len(sys.argv) > 1 else 575
tag = sys.argv[2] if len(sys.argv) > 2 else "x"

# 用标签生成唯一前缀，确保前缀缓存不命中
salt = hashlib.md5(tag.encode()).hexdigest()
para = (
    "在编号 %s 的技术评审记录中，工程团队讨论了模型推理框架的算子调度、"
    "显存分配与并行策略，并对不同硬件后端的表现做了横向对比。" % salt
)
long_text = (para * n) + "\n\n请用一句话总结上文的核心主题。"

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": long_text}],
            "max_tokens": 64,
            "temperature": 0,
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)

t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=1800))
dt = time.time() - t0
u = r["usage"]
gen = u["completion_tokens"]
# 粗略扣掉解码时间估算 prefill 速率（解码速率按实测单流 33 tok/s 折算）
decode_est = gen / 33.0
prefill_dt = max(dt - decode_est, 0.01)
print(
    "prompt_tokens=%d completion=%d 总耗时=%.1fs 估算prefill=%.1fs prefill速率≈%.0f tok/s"
    % (u["prompt_tokens"], gen, dt, prefill_dt, u["prompt_tokens"] / prefill_dt)
)
