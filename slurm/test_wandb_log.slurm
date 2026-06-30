#!/bin/bash -l
#SBATCH --job-name=test_wandb_log
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100_40
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=NONE

# 隔离测试 log_eval_to_wandb.py：不起 server/不跑仿真，直接用一个已有评测 run 目录
# （含 eval.log 或 results.json）+ ckpt，验证 wandb_run_id 读取与 resume 回写是否成功。
# 用法：
#   sbatch --export=ALL,CKPT=<abs ckpt>,RUN_DIR=<abs eval run dir> slurm/test_wandb_log.slurm
set -euo pipefail
DEFAULT_PROJECT_ROOT="/home/hpc/g108ea/g108ea10/vault/vla-wam/droid_policy_learning"
PROJECT_ROOT="${DROID_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$DEFAULT_PROJECT_ROOT}}"
cd "$PROJECT_ROOT"; mkdir -p logs

CKPT="${CKPT:?need CKPT}"; RUN_DIR="${RUN_DIR:?need RUN_DIR}"
[[ -f "$CKPT" ]] || { echo "[ERR] ckpt 不存在: $CKPT"; exit 2; }
[[ -d "$RUN_DIR" ]] || { echo "[ERR] run dir 不存在: $RUN_DIR"; exit 2; }

export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
SIF_DIR="${SIF_DIR:-${WORK:?}/sif}"; DROID_SIF="${DROID_SIF:-$SIF_DIR/droid.sif}"
WANDB_ENV_FILE="${PROJECT_ROOT}/slurm/.env.wandb"; [[ -f "$WANDB_ENV_FILE" ]] && source "$WANDB_ENV_FILE"
[[ -n "${WANDB_API_KEY:-}" ]] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
export APPTAINERENV_http_proxy="$http_proxy" APPTAINERENV_https_proxy="$https_proxy"
export APPTAINERENV_no_proxy="$no_proxy" APPTAINERENV_NO_PROXY="$NO_PROXY"

HOST_CKPT_DIR="$(cd "$(dirname "$CKPT")" && pwd)"; CKPT_NAME="$(basename "$CKPT")"
HOST_HF="${HF_CACHE_DIR:-${WORK}/huggingface}"
HOST_FAKE_ROOT="${WORK}/.apptainer_fake_root_server"
RUN_DIR_ABS="$(cd "$RUN_DIR" && pwd)"; RUN_PARENT="$(dirname "$RUN_DIR_ABS")"; RUN_NAME="$(basename "$RUN_DIR_ABS")"
mkdir -p "$HOST_HF" "$HOST_FAKE_ROOT"

echo "[INFO] ckpt=$CKPT_NAME run_dir=$RUN_NAME"
apptainer run --writable-tmpfs \
  --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  --env HF_HOME=/hf_cache \
  -B "${HOST_CKPT_DIR}:/ckpts:ro" \
  -B "${HOST_FAKE_ROOT}:/root" \
  -B "${HOST_HF}:/hf_cache" \
  -B "${RUN_PARENT}:/eval_logs" \
  "$DROID_SIF" \
  python -m robomimic.scripts.log_eval_to_wandb \
    --ckpt "/ckpts/${CKPT_NAME}" --run-dir "/eval_logs/${RUN_NAME}"
echo "[INFO] done"
