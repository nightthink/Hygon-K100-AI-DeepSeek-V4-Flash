# 专家并行（EP）启动器的构造方式

轮次 28 做 EP 对照实验时，需要一个与生产启动器**只差 EP 本身**的变体，
避免把别的变量一起改掉。做法是从 `run_0811_probe.sh` 派生，只改三处：

1. 去掉硬编码的 `--moe-a2a-backend none`
2. 新增 `EP_ARGS`，由环境变量 `EP_SIZE` / `A2A` / `DEEPEP_MODE` 驱动
3. 把 `EP_ARGS` 插进 `sglang serve` 参数列表（日志分流到 `ep_0811.log`）

```bash
EP_ARGS=(--moe-a2a-backend "${A2A:-none}")
if [ -n "${EP_SIZE:-}" ]; then
  EP_ARGS+=(--ep-size "$EP_SIZE")
  [ -n "${DEEPEP_MODE:-}" ] && EP_ARGS+=(--deepep-mode "$DEEPEP_MODE")
fi
```

用法：

```bash
# DeepEP 低延迟模式（实测：RocSHMEM 无 gfx928 设备代码，初始化即 abort）
-e EP_SIZE=8 -e A2A=deepep -e DEEPEP_MODE=low_latency

# 朴素 dispatcher（实测：fused_moe_triton 前向段错误）
-e EP_SIZE=8 -e A2A=none
```

两条路径的失败根因见 `docs/调优记录-轮次28.md`。

## 为什么值得先跑 `tests/probe_arch.sh`

上面两次启动各花约 5–8 分钟才撞到墙，而 `tests/probe_arch.sh` 在几十秒内
就能给出"`deep_ep` 只编了 gfx936/gfx938"这个结论。**先探针，再实验。**
