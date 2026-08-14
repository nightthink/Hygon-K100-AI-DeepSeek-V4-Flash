"""参数化并发压测：并发数 / max_tokens / 温度 / top_k，用于隔离 DSpark 并发崩溃。

用法: python3 bench_conc_param.py <conc> <max_tokens> <temp> <timeout> [top_k]
"""
import json, time, urllib.request, concurrent.futures, sys

BASE = "http://127.0.0.1:8000/v1/chat/completions"
M = "deepseek-v4-flash"
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 100
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
TIMEOUT = int(sys.argv[4]) if len(sys.argv) > 4 else 180
TOPK = int(sys.argv[5]) if len(sys.argv) > 5 else 0

PROMPTS = [
    "写一段150字介绍人工智能发展史的文字。",
    "解释一下什么是注意力机制，用通俗的语言。",
    "用Python写一个快速排序函数。",
    "简述量子计算与经典计算的区别。",
    "写一首关于秋天的五言绝句并解释。",
    "什么是数据库索引？举例说明。",
    "介绍一下太阳系的八大行星。",
    "解释TCP三次握手的过程。",
]


def one(i):
    body = {
        "model": M,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": MAXTOK,
        "temperature": TEMP,
    }
    if TOPK > 0:
        body["top_k"] = TOPK
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
        return r["usage"]["completion_tokens"], time.time() - t0, None
    except Exception as e:
        return 0, time.time() - t0, repr(e)[:80]


print("conc=%d max_tokens=%d temp=%.2f top_k=%s" % (CONC, MAXTOK, TEMP, TOPK or "default"), flush=True)
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
    results = list(ex.map(one, range(CONC)))
wall = time.time() - t0
ok = [r for r in results if r[2] is None]
bad = [r for r in results if r[2] is not None]
total = sum(c for c, _, _ in ok)
print("成功 %d/%d, 失败 %d" % (len(ok), CONC, len(bad)), flush=True)
if bad:
    print("  首个失败:", bad[0][2])
if total:
    print("总 tok=%d wall=%.1fs 聚合=%.1f tok/s" % (total, wall, total / wall))
