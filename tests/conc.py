"""并发聚合吞吐测试。用法: python3 conc.py <并发数> <标签>

标签拼进 prompt 以避开前缀缓存；每路 120 token 贪心生成。
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

C = int(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "x"


def req(i):
    p = {"model": "deepseek-v4-flash",
         "messages": [{"role": "user",
                       "content": f"{TAG}写一段关于消息队列编号{i}的技术说明"}],
         "max_tokens": 120, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions", json.dumps(p).encode(),
        {"Content-Type": "application/json"}), timeout=280))
    return r["usage"]["completion_tokens"]


t = time.time()
with cf.ThreadPoolExecutor(C) as ex:
    ns = list(ex.map(req, range(C)))
d = time.time() - t
print(f"{C}并发: {len(ns)}/{C} 成功, 用时 {d:.1f}s, 聚合 {sum(ns)/d:.2f} tok/s", flush=True)
