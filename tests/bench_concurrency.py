import json, time, urllib.request, concurrent.futures, sys

BASE = "http://127.0.0.1:8000/v1/chat/completions"
M = "deepseek-v4-flash"
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 128

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
    req = urllib.request.Request(
        BASE,
        data=json.dumps({
            "model": M,
            "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            "max_tokens": MAXTOK, "temperature": 0.7,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    dt = time.time() - t0
    u = r["usage"]
    return u["completion_tokens"], dt

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
    results = list(ex.map(one, range(CONC)))
wall = time.time() - t0
total = sum(c for c, _ in results)
print(f"concurrency={CONC} max_tokens={MAXTOK}")
print(f"total completion tokens={total}  wall={wall:.1f}s")
print(f"aggregate throughput={total/wall:.1f} tok/s   per-request avg={total/CONC/wall:.2f} tok/s")
