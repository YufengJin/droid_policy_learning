#!/bin/bash -l
#SBATCH --job-name=eval_decoupled
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100_40
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yufeng.jin@tu-darmstadt.de
#SBATCH --export=NONE   # 避免把登陆节点的脏环境带入

################################################################################
# 解耦跨容器评测（同一节点）：
#   policy server（droid.sif，GPU）  +  sim client（libero.sif / robocasa.sif，osmesa CPU 渲染）
#   两者经 apptainer 默认共享的宿主网络走 localhost:$PORT 通信。
#
# 用法：
#   sbatch --export=ALL,BENCHMARK=libero,CKPT=/home/atuin/.../model_epoch_N.pth   slurm/eval_app.slurm
#   sbatch --export=ALL,BENCHMARK=robocasa,CKPT=/home/atuin/.../model_epoch_N.pth slurm/eval_app.slurm
# 可覆盖：TASK / TASK_SUITE / NUM_TRIALS / PORT / *_SIF / *_DIR（见下）。
#
# 前置：三个 .sif 已 build 好放 $WORK/sif/（见各 repo 的 apptainer/build.sh）。镜像已自包含
#       （源码/素材烤入；policy_websocket pin 到 main），集群无外网亦可运行。
################################################################################
set -euo pipefail
mkdir -p logs
echo "[$(date +'%F %T')] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  NODE=${SLURM_JOB_NODELIST:-NA}"
trap 'status=$?; echo "[EXIT $(date +%T)] code=$status"' EXIT

# ---- 2) 代理（构建已 baked，运行一般不需要；wandb/HF 上传时透传给容器）----
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
# server↔client 走 localhost；必须 no_proxy，否则 websocket 客户端把本地连接也走代理被 403 拒
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ---- 3) 参数 ----
BENCHMARK="${BENCHMARK:?必须指定 BENCHMARK=libero 或 robocasa}"
CKPT="${CKPT:?必须指定 CKPT=宿主机上的 model_epoch_N.pth 绝对路径}"
PORT="${PORT:-8765}"
[[ -f "$CKPT" ]] || { echo "[ERR] ckpt 不存在: $CKPT"; exit 2; }

# PROJECT_ROOT 解析：sbatch 下脚本被拷到节点 spool（/var/tmp/slurmd_spool/...），BASH_SOURCE
#   不再指向仓库内，dirname 推导会得到 spool 目录（eval_logs mkdir 会 Permission denied）。
#   优先级：DROID_PROJECT_DIR > SLURM_SUBMIT_DIR（从仓库根提交即仓库根）> 硬编码默认
#   （与 train_*_app.slurm 一致）。
DEFAULT_PROJECT_ROOT="/home/hpc/g108ea/g108ea10/vault/vla-wam/droid_policy_learning"
PROJECT_ROOT="${DROID_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$DEFAULT_PROJECT_ROOT}}"
[[ -d "$PROJECT_ROOT" ]] || { echo "[ERR] PROJECT_ROOT 不存在: $PROJECT_ROOT（用 DROID_PROJECT_DIR 覆盖）"; exit 2; }
cd "$PROJECT_ROOT"; mkdir -p logs   # 让 logs/ 等相对路径与 PROJECT_ROOT 一致
SIF_DIR="${SIF_DIR:-${WORK:?未设 \$WORK}/sif}"
DROID_SIF="${DROID_SIF:-$SIF_DIR/droid.sif}"
BENCH_SIF="${BENCH_SIF:-$SIF_DIR/${BENCHMARK}.sif}"
for f in "$DROID_SIF" "$BENCH_SIF"; do
  [[ -f "$f" ]] || { echo "[ERR] 镜像不存在: $f（先 build：<repo>/apptainer/build.sh）"; exit 1; }
done

# benchmark 数据集（libero 评测可能用到演示数据；robocasa 素材已烤入镜像，无需挂）
LIBERO_DATA="${LIBERO_DATASET_DIR:-/home/hpc/g108ea/g108ea10/atuin/datasets/libero}"

# ---- 4) apptainer / 缓存 / fakeroot / HF / wandb（全部指向 $WORK，勿用 $HOME）----
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORK}/.apptainer_cache}"
HOST_FAKE_ROOT_SRV="${APPTAINER_FAKE_ROOT:-${WORK}/.apptainer_fake_root_server}"
HOST_HF="${HF_CACHE_DIR:-${WORK}/huggingface}"          # server 端 DistilBERT tokenizer 下载/缓存
HOST_CKPT_DIR="$(cd "$(dirname "$CKPT")" && pwd)"; CKPT_NAME="$(basename "$CKPT")"
HOST_EVAL_LOGS="${EVAL_LOGS_DIR:-${PROJECT_ROOT}/eval_logs}"   # 评测日志/结果落盘（持久）
HOST_TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$HOST_FAKE_ROOT_SRV" "$HOST_HF" "$HOST_EVAL_LOGS"

# wandb（如评测要 log）：-C 隔离下需显式透传 key+proxy
WANDB_ENV_FILE="${PROJECT_ROOT}/slurm/.env.wandb"
[[ -f "$WANDB_ENV_FILE" ]] && source "$WANDB_ENV_FILE"
[[ -n "${WANDB_API_KEY:-}" ]] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
export APPTAINERENV_http_proxy="${http_proxy}" APPTAINERENV_https_proxy="${https_proxy}"
export APPTAINERENV_no_proxy="${no_proxy}" APPTAINERENV_NO_PROXY="${NO_PROXY}"

# ---- 5) 启动 policy server（droid.sif，后台）----
# 源码已烤入 droid.sif，只 bind 存放 ckpt 的目录；--nv + LD 修复（容器自带 GL 优先，规避宿主 GL 库 glibc 冲突，
# torch 仍 cuda True）；--writable-tmpfs（entrypoint 幂等检查 + 只读 sif 的运行时写）；HF_HOME 指可写盘。
SRV_LOG="logs/${SLURM_JOB_ID:-local}_server.log"
echo "[INFO] 起 policy server: $DROID_SIF  ckpt=$CKPT  port=$PORT"
# websocket_server.py 补丁（ping_interval=None）：镜像内 policy_websocket 默认 20s keepalive ping，
#   同步推理(首次 CUDA warmup)阻塞事件循环会被判超时断连(1011)。若 patch 文件在就 bind 覆盖安装路径。
#   长期正解：在 droid.sif/robocasa.sif build 时更新 policy_websocket pin 或注入此 patch。
WS_PATCH_BIND=()
WS_PATCH_SRC="${PROJECT_ROOT}/slurm/patches/websocket_server.py"
[[ -f "$WS_PATCH_SRC" ]] && WS_PATCH_BIND=(-B "${WS_PATCH_SRC}:/opt/venv/lib/python3.10/site-packages/policy_websocket/websocket_server.py:ro")
apptainer run --nv --writable-tmpfs \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  --env HF_HOME=/hf_cache \
  -B "${HOST_CKPT_DIR}:/ckpts:ro" \
  -B "${HOST_FAKE_ROOT_SRV}:/root" \
  -B "${HOST_HF}:/hf_cache" \
  -B "${HOST_TMPDIR}:/tmp" \
  "${WS_PATCH_BIND[@]}" \
  "$DROID_SIF" \
  python -m robomimic.scripts.policy_server --ckpt "/ckpts/${CKPT_NAME}" --port "$PORT" \
  > "$SRV_LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

# 等 server ready（最多 ~10 分钟：policy_server 冷启动要 import TF+torch、载模型、首次可能下 DistilBERT；
#   慢节点上 >5 分钟常见，故放宽到 200×3s=600s，避免刚好在 ready 前超时）
echo "[INFO] 等 server 就绪 ..."
for i in $(seq 1 200); do
  grep -q "listening on ws://" "$SRV_LOG" 2>/dev/null && { echo "[INFO] server ready"; break; }
  kill -0 $SRV_PID 2>/dev/null || { echo "[ERR] server 进程退出，日志："; tail -20 "$SRV_LOG"; exit 1; }
  sleep 3
  [[ $i -eq 200 ]] && { echo "[ERR] server 等待超时"; tail -20 "$SRV_LOG"; exit 1; }
done

# ---- 6) 跑 sim client（benchmark.sif，osmesa；连 localhost:$PORT）----
# 源码/资源已烤入；MUJOCO_GL=osmesa（CPU 软件渲染，不需 --nv，规避 --nv 的 GL glibc 冲突）；--log_dir 指可写绑入目录。
# 注意：client 运行时往只读镜像内写（robocasa 每个物体在 asset 同目录写 *_pid.xml、editable 装包写 /opt/venv、
#   numba/matplotlib 缓存）。--writable-tmpfs 的内存 overlay 默认仅 16MiB，跑几个物体就 ENOSPC。
#   故改用建在大盘（$HOST_TMPDIR，节点本地 ~TB）上的 overlay 镜像，给足可写空间。
EVAL_OVERLAY="${HOST_TMPDIR}/eval_overlay_${SLURM_JOB_ID:-$$}.img"
apptainer overlay create --size 4096 "$EVAL_OVERLAY" >/dev/null 2>&1 || { echo "[ERR] 建 overlay 失败"; exit 1; }
trap 'kill $SRV_PID 2>/dev/null || true; rm -f "$EVAL_OVERLAY" 2>/dev/null || true' EXIT
echo "[INFO] 起 ${BENCHMARK} sim client，连 localhost:$PORT（overlay=$EVAL_OVERLAY）"

# run_eval.py 覆盖绑定：镜像内烤入的 run_eval 是 build 时的旧版（不写 results.json）。
#   把宿主 sibling repo 里的最新版 bind 覆盖，使其额外产出 results.json（供后面回写 wandb）。
#   重建 sif 后可去掉此 bind。文件不在就跳过（logger 会回退解析 eval.log）。
VLA_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
RUN_EVAL_BIND=()
if [[ "$BENCHMARK" == "libero" ]]; then
  _re="${LIBERO_REPO:-$VLA_ROOT/LIBERO}/scripts/run_eval.py"
  [[ -f "$_re" ]] && RUN_EVAL_BIND=(-B "${_re}:/workspace/libero/scripts/run_eval.py:ro")
elif [[ "$BENCHMARK" == "robocasa" ]]; then
  _re="${ROBOCASA_REPO:-$VLA_ROOT/robocasa}/scripts/run_eval.py"
  [[ -f "$_re" ]] && RUN_EVAL_BIND=(-B "${_re}:/workspace/robocasa/scripts/run_eval.py:ro")
fi

if [[ "$BENCHMARK" == "libero" ]]; then
  SUITE="${TASK_SUITE:-libero_10}"; NTRIALS="${NUM_TRIALS:-20}"
  DATA_BIND=(); [[ -d "$LIBERO_DATA" ]] && DATA_BIND=(-B "${LIBERO_DATA}:/workspace/libero/datasets:ro")
  apptainer run --overlay "$EVAL_OVERLAY" --env MUJOCO_GL=osmesa \
    --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    --env NUMBA_CACHE_DIR=/tmp/numba_cache --env XDG_CACHE_HOME=/tmp/xdg_cache \
    -B "${PROJECT_ROOT}/slurm/patches/websocket_client.py:/opt/venv/lib/python3.8/site-packages/policy_websocket/websocket_client.py:ro" \
    "${RUN_EVAL_BIND[@]}" \
    -B "${HOST_EVAL_LOGS}:/eval_logs" -B "${HOST_TMPDIR}:/tmp" "${DATA_BIND[@]}" \
    "$BENCH_SIF" \
    bash -lc "cd /workspace/libero && python scripts/run_eval.py \
      --task_suite_name '${SUITE}' --policy_server_addr localhost:${PORT} \
      --num_trials_per_task '${NTRIALS}' --log_dir /eval_logs/${SLURM_JOB_ID:-local}"
elif [[ "$BENCHMARK" == "robocasa" ]]; then
  TASK="${TASK:-PnPCounterToCab}"; NTRIALS="${NUM_TRIALS:-5}"
  apptainer run --overlay "$EVAL_OVERLAY" --env MUJOCO_GL=osmesa \
    --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    --env NUMBA_CACHE_DIR=/tmp/numba_cache --env XDG_CACHE_HOME=/tmp/xdg_cache \
    -B "${PROJECT_ROOT}/slurm/patches/websocket_client.py:/opt/venv/lib/python3.10/site-packages/policy_websocket/websocket_client.py:ro" \
    "${RUN_EVAL_BIND[@]}" \
    -B "${HOST_EVAL_LOGS}:/eval_logs" -B "${HOST_TMPDIR}:/tmp" \
    "$BENCH_SIF" \
    bash -lc "cd /workspace/robocasa && python scripts/run_eval.py \
      --task_name '${TASK}' --policy_server_addr localhost:${PORT} \
      --num_trials '${NTRIALS}' --no_save_video --log_dir /eval_logs/${SLURM_JOB_ID:-local}"
else
  echo "[ERR] 未知 BENCHMARK=$BENCHMARK（应为 libero / robocasa）"; exit 2
fi

echo "[INFO] 评测完成 $(date +'%F %T')；日志在 $HOST_EVAL_LOGS"

# ---- 7) 回写 wandb：用 ckpt 里的 wandb_run_id resume 训练那个 run，append eval/* 指标 ----
# 在 droid.sif 内跑（装了 wandb + macros + API key）。找本次评测最新的 run 目录传给 logger。
# 整步非致命：失败不影响评测作业的退出码（logger 自身对无 run_id/无网也是 exit 0）。
RUN_DIR_HOST="$(ls -dt ${HOST_EVAL_LOGS}/${SLURM_JOB_ID:-local}/*/ 2>/dev/null | head -1)"
if [[ -n "$RUN_DIR_HOST" ]]; then
  RUN_DIR_NAME="$(basename "$RUN_DIR_HOST")"
  echo "[INFO] 回写 wandb：run_dir=$RUN_DIR_NAME ckpt=$CKPT_NAME"
  apptainer run --writable-tmpfs \
    --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    --env HF_HOME=/hf_cache \
    -B "${HOST_CKPT_DIR}:/ckpts:ro" \
    -B "${HOST_FAKE_ROOT_SRV}:/root" \
    -B "${HOST_HF}:/hf_cache" \
    -B "${HOST_EVAL_LOGS}:/eval_logs" \
    -B "${HOST_TMPDIR}:/tmp" \
    "$DROID_SIF" \
    python -m robomimic.scripts.log_eval_to_wandb \
      --ckpt "/ckpts/${CKPT_NAME}" --run-dir "/eval_logs/${SLURM_JOB_ID:-local}/${RUN_DIR_NAME}" \
    || echo "[WARN] wandb 回写失败（不影响评测结果）"
else
  echo "[WARN] 未找到评测 run 目录，跳过 wandb 回写"
fi
