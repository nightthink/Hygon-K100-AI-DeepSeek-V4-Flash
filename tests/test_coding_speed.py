"""编程题单流速度测试（用户真实负载画像）。跑两次取均值。"""
import json, time, urllib.request, sys

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT = "请写一个Python函数，实现快速排序，并附带简要注释。"


def run(max_tokens=600):
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t0
    u = r["usage"]
    return u["completion_tokens"], dt, u["completion_tokens"] / dt


tag = sys.argv[1] if len(sys.argv) > 1 else ""
rates = []
for i in range(2):
    tok, dt, rate = run()
    rates.append(rate)
    print("  %s run%d: %d tok / %.1fs = %.2f tok/s" % (tag, i + 1, tok, dt, rate), flush=True)
print("  %s 均值: %.2f tok/s" % (tag, sum(rates) / len(rates)))
