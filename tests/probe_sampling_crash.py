"""判别实验：DSpark 采样崩溃到底卡在代码路径，还是拒绝采样的残差重采样分支。

背景：本线在 temperature=0 下全并发稳定，temperature>0 且并发 >=8 会触发
HSA_STATUS_ERROR_EXCEPTION 0x1016，或 detokenizer 卡死（同一缺陷的两种表现）。

判别思路：
  temperature=0.7 + top_k=1  在数学上等价于贪心（候选集只有 1 个 token，
  无论怎么缩放 logits，采出来的必然是 argmax），但它**走的是采样代码路径**
  ——do_sample 分支、接受判定、残差缓冲全都要过一遍，唯独不会发生"拒绝后重采样"。

  top_k=1 不崩、top_k=2 崩 → 触发条件是拒绝重采样分支被真正执行，
  与采样代码路径本身、与随机性本身都无关。

实测结论（轮次 29）：
  temperature=0        并发8 → 8/8 通过，accept rate 0.60-0.77
  temperature=0.7 k=1  并发8 → 8/8 通过，accept rate 0.60-0.77
  temperature=0.7 k=2  并发8 → HSA_STATUS_ERROR_EXCEPTION 0x1016，accept rate 0.25
  temperature=0.7 全词表 并发8 → detokenizer 卡死，accept rate 0.26

用法：
  python3 probe_sampling_crash.py <并发数> <temperature> <top_k|0表示不传>

注意：
  1. 不要预热采样路径。已知预热会把安全并发从 2 抬到 4，会污染结论。
  2. 触发后服务需重启（约 14 分钟），请勿在有实际流量时运行。
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TEMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def one(i):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": f"简述分布式系统中的一致性问题，编号{i}。"}],
        "max_tokens": 100,
        "temperature": TEMP,
    }
    if TOPK > 0:
        payload["top_k"] = TOPK
    req = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        r = json.load(urllib.request.urlopen(req, timeout=40))
        return ("ok", r["usage"]["completion_tokens"])
    except Exception as e:
        return ("ERR", type(e).__name__)


label = f"并发={CONC} temperature={TEMP} top_k={TOPK if TOPK else '未传'}"
print(f"=== {label} ===")
t = time.time()
with cf.ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(one, range(CONC)))
d = time.time() - t

ok = [r for r in res if r[0] == "ok"]
err = [r for r in res if r[0] != "ok"]
toks = sum(r[1] for r in ok)
print(f"成功 {len(ok)}/{CONC}   用时 {d:.2f}s   聚合 {toks/d:.2f} tok/s" if ok else
      f"成功 0/{CONC}   用时 {d:.2f}s")
if err:
    kinds = {}
    for _, k in err:
        kinds[k] = kinds.get(k, 0) + 1
    print("失败类型:", kinds)
print("提示：同时观察服务端日志里的 accept rate——从 0.6+ 掉到 0.25 即为重采样分支被密集触发。")
