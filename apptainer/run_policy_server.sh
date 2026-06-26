#!/usr/bin/env bash
# 在 droid.sif 内起 diffusion-policy WebSocket server（解耦评测的策略端）。
# benchmark 自动识别（libero/robocasa），sim client 在 libero.sif/robocasa.sif 里跑（见各 benchmarks/*/apptainer/run_eval.sh）。
#
# 已验证的关键 flag（本机 RTX4090 + apptainer 1.5.2 实测）：
#   --nv                                : 策略推理需要 GPU。
#   LD_LIBRARY_PATH=容器库优先          : --nv 会注入宿主(Ubuntu24.04, glibc2.39)的 GL 库(libGLdispatch)，
#                                         与容器(ubuntu20.04, glibc2.31)不兼容，凡加载 GL 的 import(cv2)即崩。
#                                         把容器自带库目录放最前 → cv2 用容器 GL，torch 仍走宿主 libcuda（cuda True）。
#   --writable-tmpfs                    : entrypoint 要 `uv pip install -e .` 并把 policy_websocket 更新到 git HEAD，
#                                         以与 client 协议一致(__command__)；只读 sif 需可写覆盖层。
#                                         注意：droid.sif 与 robocasa.sif build 时 pin 的 policy_websocket 版本不同，
#                                         靠 entrypoint 在线更到 HEAD 对齐 —— 集群无外网时应改为 build 时 pin 同一 commit。
#   HF_HOME=可写目录                    : robocasa 模型在 server 端编码 DistilBERT 语言，需下载 tokenizer；
#                                         apptainer 默认挂宿主 $HOME，其 ~/.cache/huggingface 可能 root 属主不可写 → 显式指可写盘。
# 集群上：APPTAINER_CACHEDIR/HF cache/数据集都指向 $WORK；代理用 APPTAINERENV_http_proxy 透传。
# 源码已烤入 droid.sif（见 docker/Dockerfile §7），集群上**不需要 bind 源码树**，只 bind 存放 ckpt 的 outputs 目录。
# 默认把宿主 CKPT_DIR 挂到容器 /workspace/droid_policy_learning/outputs；CKPT 给容器内路径。
# 要 live 改 robomimic 代码：设 DROID_CODE_DIR=/path/to/droid_policy_learning 覆盖整棵源码（editable 路径一致）。
set -euo pipefail
SIF="${DROID_SIF:-/mnt/ssd2T/yjin/sif/droid.sif}"
CKPT_DIR="${CKPT_DIR:-/home/yjin/localdisk/vla-code/droid_policy_learning/outputs}"   # 宿主 ckpt(outputs) 目录
CKPT="${CKPT:?必须设 CKPT=/workspace/droid_policy_learning/outputs/<run>/<ts>/models/model_epoch_N.pth (容器内路径)}"
PORT="${PORT:-8765}"
HOST_FAKEROOT="${APPTAINER_FAKE_ROOT:-/mnt/ssd2T/yjin/.fakeroot_server}"
HOST_HF="${HF_CACHE_DIR:-/mnt/ssd2T/yjin/.hf_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/mnt/ssd2T/yjin/.apptainer_cache}"
mkdir -p "$HOST_FAKEROOT" "$HOST_HF"

# 默认 bind：仅 ckpt 目录 + fakeroot + hf；DROID_CODE_DIR 设了则改为覆盖整棵源码（dev/live）。
if [ -n "${DROID_CODE_DIR:-}" ]; then
  CODE_BIND=(-B "${DROID_CODE_DIR}:/workspace/droid_policy_learning")
else
  CODE_BIND=(-B "${CKPT_DIR}:/workspace/droid_policy_learning/outputs")
fi

exec apptainer run --nv --writable-tmpfs \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  --env HF_HOME=/hf_cache \
  "${CODE_BIND[@]}" \
  -B "${HOST_FAKEROOT}:/root" \
  -B "${HOST_HF}:/hf_cache" \
  "$SIF" \
  python -m robomimic.scripts.policy_server --ckpt "$CKPT" --port "$PORT"
