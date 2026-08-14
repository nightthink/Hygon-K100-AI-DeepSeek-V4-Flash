"""探针：sglang 的 Triton int4 MoE 内核能否在 gfx928 上编译并算出正常数值。

## 背景

轮次 28/31 查明低比特 C++ 算子（cutlass_w4a8_moe_mm 等）没编进这个 ROCm 构建。
但 fused MoE 还有一条 **Triton** 路径带 use_int4_w4a16（4-bit 权重 / 16-bit 激活）。
Triton 是运行时 JIT，按当前架构现场生成代码——理论上不需要任何 gfx928 移植。

且 moe_wna16.py 的 AWQ 分支写有 `if not _is_hcu:` 才做能力检查，
即海光已为 HCU 放行这条路；本机 `is_hcu()` 返回 True。

## 当前状态（轮次 32，未完成）

- ✅ 内核**确实为 gfx928 编译并执行**：配置查找打出 `dtype=int4_w4a16`，无异常，返回正常形状张量
- ❌ 数值验证**未成功**：全零权重与随机权重都得到全零输出；常数权重触发 VM fault

张量形状已按 moe_wna16.py 的 create_weights 对齐（含"零点也按每字节两值打包"这个坑），
但 AWQ 的 nibble 排列有特定顺序（不是简单的高低半字节），手工构造等于盲写一遍打包逻辑。

**结论：这是探针的局限，不是平台的结论。** 要验证这条路，必须走真实加载器 + 真实 AWQ 权重。

## 张量形状（取自 moe_wna16.py create_weights，权威来源）

    w13_qweight (E, 2N,   K/2)        uint8   权重每字节两值
    w2_qweight  (E, K,    N/2)        uint8
    w13_scales  (E, 2N,   K/group)    bf16
    w2_scales   (E, K,    N/group)    bf16
    w13_qzeros  (E, 2N/2, K/group)    uint8   零点同样每字节两值
    w2_qzeros   (E, K/2,  N/group)    uint8
    block_shape [0, group_size]

## ⚠️ 使用注意

**必须先停掉推理服务再跑。** 本探针可能触发 VM fault，而 VM fault 会把同一批 GPU 上
正在服务的进程一起带崩（表现为 detokenizer 卡死）。轮次 32 因此多重启了一次。
"""
import torch

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="/models"))

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl

torch.manual_seed(0)
dev = "cuda"
print("架构:", torch.cuda.get_device_properties(0).gcnArchName)

E, M, K, N = 8, 16, 512, 256
TOPK = 2
G = 128
PF = 2  # pack factor = 8 // 4 bits

topk_w = torch.rand(M, TOPK, dtype=torch.float32, device=dev)
topk_w = topk_w / topk_w.sum(-1, keepdim=True)
topk_i = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=dev)
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.1


def build(random_w):
    if random_w:
        w1q = torch.randint(0, 256, (E, 2 * N, K // PF), dtype=torch.uint8, device=dev)
        w2q = torch.randint(0, 256, (E, K, N // PF), dtype=torch.uint8, device=dev)
    else:
        w1q = torch.zeros(E, 2 * N, K // PF, dtype=torch.uint8, device=dev)
        w2q = torch.zeros(E, K, N // PF, dtype=torch.uint8, device=dev)
    w1s = torch.full((E, 2 * N, K // G), 0.01, dtype=torch.bfloat16, device=dev)
    w2s = torch.full((E, K, N // G), 0.01, dtype=torch.bfloat16, device=dev)
    # 零点也按每字节两值打包：0x88 = 两个 8
    w1z = torch.full((E, 2 * N // PF, K // G), 0x88, dtype=torch.uint8, device=dev)
    w2z = torch.full((E, K // PF, N // G), 0x88, dtype=torch.uint8, device=dev)
    return w1q, w2q, w1s, w2s, w1z, w2z


def run(tag, random_w):
    w1q, w2q, w1s, w2s, w1z, w2z = build(random_w)
    out = fused_experts_impl(
        x, w1q, w2q, topk_w, topk_i,
        inplace=False,
        use_int4_w4a16=True,
        w1_scale=w1s, w2_scale=w2s,
        w1_zp=w1z, w2_zp=w2z,
        block_shape=[0, G],
    )
    torch.cuda.synchronize()
    nan = bool(torch.isnan(out).any().item() or torch.isinf(out).any().item())
    mx = out.abs().max().item()
    print(f"  [{tag}] 形状={tuple(out.shape)} "
          f"范围=[{out.min().item():.5f}, {out.max().item():.5f}] NaN/Inf={nan}")
    return nan, mx


try:
    print("健全性检查（全零权重，期望输出全零）:")
    nan0, mx0 = run("zeros", False)
    print("数值检查（随机权重，期望非零且无 NaN）:")
    nan1, mx1 = run("random", True)

    ok = (not nan0) and (not nan1) and mx0 == 0.0 and mx1 > 0.0
    print()
    print("结论:", "Triton int4 路径在 gfx928 上可用" if ok
          else f"跑通但数值可疑（零权重 max={mx0}, 随机权重 max={mx1}, NaN={nan0 or nan1}）")
except Exception as e:
    print("结果:", type(e).__name__)
    print("  ", str(e).replace("\n", " ")[:600])
