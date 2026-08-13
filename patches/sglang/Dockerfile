FROM harbor.sourcefind.cn:5443/dcu/admin/base/custom:dsv4-flash-k100ai-sglang0.5.12-20260728
# 1) 同步加载补丁：多线程 H2D 拷贝在 DTK/HIP 栈死锁（Bug 报告 A-1），should_async_load 恒返 False
COPY model_loader_utils_patched_0728.py /usr/local/lib/python3.10/dist-packages/sglang/srt/model_loader/utils.py
# 2) 官方启动器（含我方 2 处修改：+reasoning/tool-call 双解析器、启用前缀缓存、删不存在的 chat-template 引用）
#    放镜像内备用，不设默认入口——启动仍由用户显式指定脚本与参数
COPY launchers/ /opt/dsv4/launchers/
LABEL maintainer="lzd" \
      base="dsv4-flash-k100ai-sglang0.5.12-20260728" \
      patches="syncload+param-launchers-20260812-v2" \
      description="DeepSeek-V4-Flash on 8xK100-AI sglang line: sync weight-load patch baked, official launchers (parser/radix mods + MEM_FRACTION_STATIC/PREFILL_CHUNK/NSA_FUSE_TOPK params, tuned defaults doc: mf0.85 chunk4096 graphbs16) at /opt/dsv4/launchers. No default entrypoint."
