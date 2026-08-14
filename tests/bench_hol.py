"""混合负载下的队头阻塞（head-of-line blocking）测试。

## 测什么

编程助手场景的真实痛点不是单流 tok/s，而是：
**当某个请求正在预填一段长上下文时，其它交互式请求要等多久才吐出第一个字。**

本脚本先发一个长 prompt 请求占住预填，1.5 秒后并发发若干短请求，
用流式接口量每个短请求的 TTFT（首 token 时间）与端到端耗时，
并给出"队头阻塞放大倍数" = 受扰 TTFT 中位数 / 无干扰 TTFT。

## 为什么这个指标比 tok/s 更重要

轮次 34 实测：无干扰时短请求 TTFT 0.64s，而一个约 8K 上下文正在预填时，
同样的短请求要等 **54.79s** —— 放大 **86 倍**。

相比之下，单流解码从 33 提到 50 tok/s（且已被证明卡在权重带宽上，见轮次 26/33）
对用户体感的改善远不如把这 55 秒压下来。

用法：python3 bench_hol.py <标签>

注意：标签会拼进长 prompt 的内容，因此不同标签天然避开前缀缓存——
每组实验请用不同标签，否则第二次跑会命中缓存、预填瞬间完成而失去意义。
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def post_stream(payload, timeout=180):
    """返回 (TTFT, 总耗时, 产出 token 数)。"""
    req = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                d = json.loads(body)
            except Exception:
                continue
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return ttft, time.time() - t0, n


def long_request():
    # 约 8K token 的长上下文，制造一段可观的预填
    fill = " ".join(
        f"{TAG}第{i}节 分布式系统的一致性协议需要在可用性与分区容忍之间权衡。"
        for i in range(430)
    )
    return post_stream({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": fill + " 请用一句话总结。"}],
        "max_tokens": 200, "temperature": 0, "stream": True,
    })


def short_request(i):
    return post_stream({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": f"用一句话说明什么是哈希表（编号{i}）。"}],
        "max_tokens": 50, "temperature": 0, "stream": True,
    })


print(f"=== {TAG} ===", flush=True)

base_ttft, base_tot, base_n = short_request(999)
print(f"无干扰短请求: TTFT={base_ttft:.2f}s 总耗时={base_tot:.2f}s tokens={base_n}", flush=True)

ex = cf.ThreadPoolExecutor(8)
fut_long = ex.submit(long_request)
time.sleep(1.5)                      # 让长请求先进入预填
futs = [ex.submit(short_request, i) for i in range(4)]

shorts = [f.result() for f in futs]
lttft, ltot, ln = fut_long.result()

ttfts = [s[0] for s in shorts if s[0] is not None]
tots = [s[1] for s in shorts]
print(f"长请求:      TTFT={lttft:.2f}s 总耗时={ltot:.2f}s tokens={ln}", flush=True)
print(f"受扰短请求:  TTFT 最小/中位/最大 = "
      f"{min(ttfts):.2f} / {sorted(ttfts)[len(ttfts)//2]:.2f} / {max(ttfts):.2f} s", flush=True)
print(f"             端到端最大 = {max(tots):.2f}s", flush=True)
print(f"队头阻塞放大倍数（中位 TTFT / 无干扰 TTFT）= "
      f"{sorted(ttfts)[len(ttfts)//2] / base_ttft:.1f}x", flush=True)
